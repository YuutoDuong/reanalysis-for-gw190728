"""The injection tables, generated from the results files.

Appendix B: one row per noise segment: the source parameters both
campaigns share (from injection_lnb.json; the first campaign injected the
masses as they are, the second multiplies them by the redshift factor
1+z), and per campaign the network signal-to-noise ratio
(injections.network_snr) and the Bayes factors as the paper quotes them
(injections.background: full-grid likelihood once script 30 has run,
mass-cut offset included).

Section V.G: one row per environmental signal (script 31): its loudness,
the exact ln B of its vacuum twin and of itself under both priors, and
its shifts with Virgo and with Virgo and the catalogue's spectra, with
GW190728's values in the last row.

Section V.H: one row per vacuum signal made with SEOBNRv4PHM (script 32):
the norm <dh|dh> of its difference from its XPHM twin after alignment,
its exact ln B under both priors, and its shifts and the twin's, with
GW190728's values in the last row.

Writes paper_appB_table.tex, paper_cloud_table.tex and paper_eob_table.tex,
so no number in any of them is transcribed by hand.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.injections import (EVENT_SNR, background, matched_keys, network_snr,
                                   select_sr_subset)

CAMPAIGNS = ("", "_det")
records = {tag: json.loads(Path(f"injection_lnb{tag}.json").read_text()) for tag in CAMPAIGNS}
lnb = {tag: background(tag) for tag in CAMPAIGNS}
snr = {tag: network_snr(tag) for tag in CAMPAIGNS}
superradiance = {   # the injections re-analysed under the superradiance prior
    "": {key for key, _ in select_sr_subset("injection_lnb.json")},
    "_det": matched_keys("_det") & set(json.loads(Path("injection_sr_lnb_det.json").read_text())),
}


def cell(value):
    return f"${value:+.2f}$" if abs(value) < 100 else f"${value:+.0f}$"


lines = []
for key in sorted(records[""]):
    p = records[""][key]["parameters"]
    row = [key.replace("inj", ""), f"{p['mass_1']:.2f}", f"{p['mass_2']:.2f}", f"{p['a_1']:.2f}",
           f"{p['luminosity_distance']:.0f}", f"{p['theta_jn']:.2f}"]
    for tag in CAMPAIGNS:
        if tag:
            row.append(f"{records[tag][key]['parameters']['mass_1'] / p['mass_1']:.3f}")
        agnostic, sr = lnb[tag][key]
        row += [f"{snr[tag][key]:.1f}", cell(agnostic), cell(sr) if key in superradiance[tag] else "---"]
    lines.append("  " + " & ".join(row) + r" \\")

out = Path("paper_appB_table.tex")
out.write_text("\n".join(lines) + "\n")
print(f"{out}: {len(lines)} rows")

# ------------------------------------------------- the environmental signals
NETWORKS = ("H1+L1+V1", "H1+L1+V1, BayesWave")
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2", "gw190728_env_rho_max2e8"]
event = json.loads(Path("reweighted_lnb.json").read_text())["lnb"]
cloud, twin = (json.loads(Path(f"injection_virgo{tag}.json").read_text()) for tag in ("_det_cloud", "_det"))


def loudness(label):
    """Network optimal SNR of the signal injected for a run."""
    ifos = json.loads(Path(f"outdir/{label}/{label}_result.json").read_text())[
        "meta_data"]["likelihood"]["interferometers"]
    return math.hypot(*(ifo["optimal_SNR"] for ifo in ifos.values()))


def shift(entry, network):
    """Agnostic shift from H1+L1, daggered where the reweighting rests on
    fewer than 30 effective samples."""
    value = entry[network]["agnostic"]["lnb"] - entry["H1+L1"]["agnostic"]["lnb"]
    return f"${value:+.2f}" + (r"^{\dagger}$" if entry[network]["agnostic"]["ess"] < 30 else "$")


lines = []
for key in sorted(cloud):
    values = [entry["H1+L1"][prior]["lnb"] for entry in (twin[key], cloud[key])
              for prior in ("agnostic", "superradiance")]
    lines.append("  " + " & ".join([key.replace("inj", ""), f"{loudness(f'{key}_det_cloud_env'):.1f}"]
                                   + [cell(v) for v in values]
                                   + [shift(cloud[key], n) for n in NETWORKS]) + r" \\")
base = sum(event[label]["exact"] for label in BASE) / len(BASE)
observed = [base, event["gw190728_env_sr_eq9_taud1e6_rho2e8"]["exact"],
            event["gw190728_env_hlv"]["exact"] - base, event["gw190728_env_hlv_gwtc21psd"]["exact"] - base]
lines += [r"  \hline",
          "  " + " & ".join(["GW190728", f"{EVENT_SNR:.1f}", "---", "---"] + [cell(v) for v in observed])
          + r" \\"]

out = Path("paper_cloud_table.tex")
out.write_text("\n".join(lines) + "\n")
print(f"{out}: {len(cloud)} injections and GW190728")

# ------------------------------------------- the second waveform family
eob = json.loads(Path("injection_virgo_det_seobnrv4phm.json").read_text())
alignment = json.loads(Path("injection_alignment_det_seobnrv4phm.json").read_text())

lines = []
for key in sorted(eob):
    rho = loudness(f"{key}_det_seobnrv4phm_vac"), loudness(f"{key}_det_vac")
    norm = rho[0] ** 2 + rho[1] ** 2 - 2 * rho[0] * rho[1] * alignment[key]["overlap"]
    lines.append("  " + " & ".join([key.replace("inj", ""), f"{norm:.1f}"]
                                   + [cell(eob[key]["H1+L1"][prior]["lnb"])
                                      for prior in ("agnostic", "superradiance")]
                                   + [shift(entry[key], n) for entry in (eob, twin) for n in NETWORKS])
                 + r" \\")
lines += [r"  \hline",
          "  " + " & ".join(["GW190728", "---"] + [cell(v) for v in observed] + ["---", "---"]) + r" \\"]

out = Path("paper_eob_table.tex")
out.write_text("\n".join(lines) + "\n")
print(f"{out}: {len(eob)} injections and GW190728")
