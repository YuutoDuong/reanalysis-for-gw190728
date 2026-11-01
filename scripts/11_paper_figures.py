"""Paper versions of the robustness and background figures (no sampling).

Scripts 04-06 draw quick-look figures with raw run labels, a fixed
x-range and raw ln B; these are the versions the paper uses: grouped,
labelled rows, a histogram whose range always covers every clean
injection, and every ln B for the full-grid likelihood with the mass-cut
offset included.  Reads reweighted_lnb.json (script 17) and, for the
injections, injections.background (full grid once script 30 has run);
writes figures/.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.analysis import compare_with_catalog, environment_corner, vacuum_vs_env_overlay
from scalar_env.data import CATALOGUE
from scalar_env.inference import exact_posterior, mass_cut_offset
from scalar_env.injections import (EVENT_SNR, GLITCH, MATCH, background, matched_keys,
                                   network_snr, select_sr_subset)

OFFSET = mass_cut_offset()
LNB = {label: (v["exact"], v["err"]) for label, v in   # full grid, offset included
       json.loads(Path("reweighted_lnb.json").read_text())["lnb"].items()}
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
        "gw190728_env_rho_max2e8"]
ROY = {5: 3.4, 6: 3.5, 7: 3.5, 8: 2.8}          # Roy et al., SR priors
# the two vacuum campaigns: the first injected the source-frame masses
# unredshifted, so its binaries are lighter than GW190728
CAMPAIGNS = (("", "lighter binaries", dict(color="0.55", mfc="none")),
             ("_det", "masses as GW190728's", dict(color="C0")))
CLOUD = dict(color="C1", marker="^", ls="none", ms=5)   # the environmental signals (script 31)
EOB = dict(color="C2", marker="s", ls="none", ms=4)     # vacuum signals made with SEOBNRv4PHM (script 32)
# the campaigns of scripts 31-32 in the ten segments: (tag, a run of each
# injection, legend label, style)
INJECTED = (("_det_seobnrv4phm", "vac", "made with SEOBNRv4PHM", EOB),
            ("_det_cloud", "env", "with a cloud", CLOUD))


def injected(tag, run):
    """{key: (network SNR, agnostic ln B, superradiance ln B)} of a campaign
    of scripts 31-32, exact likelihood, offset included; the SNR is the
    injected signal's, read from one of its runs."""
    out = {}
    for key, entry in json.loads(Path(f"injection_virgo{tag}.json").read_text()).items():
        label = f"{key}{tag}_{run}"
        ifos = json.loads(Path(f"outdir/{label}/{label}_result.json").read_text())[
            "meta_data"]["likelihood"]["interferometers"]
        out[key] = (float(np.hypot(*[ifo["optimal_SNR"] for ifo in ifos.values()])),
                    *(entry["H1+L1"][prior]["lnb"] for prior in ("agnostic", "superradiance")))
    return out

# (group, [(row label, run label), ...]); agnostic rows use rho_max = 2e8
GROUPS = [
    ("Density bound", [(r"$\rho_{\max}=10^{6}$", "gw190728_env_rho_max1e6"),
                       (r"$\rho_{\max}=10^{7}$", "gw190728_env_rho_max1e7"),
                       (r"$\rho_{\max}=10^{8}$", "gw190728_env_rho_max1e8")]),
    ("Prior on $\\alpha$", [("log-uniform", "gw190728_env_log_alpha")]),
    ("Sampler", [("MultiNest", "gw190728_env_pymultinest"),
                 ("nessai", "gw190728_env_nessai")]),
    ("Waveform", [("IMRPhenomPv2", "gw190728_env_imrphenompv2"),
                  ("IMRPhenomXP", "gw190728_env_imrphenomxp")]),
    ("Noise / network", [("Welch 8 s", "gw190728_env_psd8s"),
                         ("Welch 16 s", "gw190728_env_psd16s"),
                         ("H1 only", "gw190728_env_only_H1"),
                         ("L1 only", "gw190728_env_only_L1"),
                         ("H1+L1+V1", "gw190728_env_hlv"),
                         ("H1+L1+V1, BayesWave", "gw190728_env_hlv_gwtc21psd")]),
    ("Phase", [("sampled", "gw190728_env_phase_sampled"),
               ("sampled, H1+L1+V1, BayesWave", "gw190728_env_hlv_gwtc21psd_phase_sampled")]),
    ("Superradiance, $2\\times10^{8}$",
     [(rf"$\tau_d=10^{e}$ yr", f"gw190728_env_sr_eq9_taud1e{e}_rho2e8") for e in (5, 6, 7, 8)]),
    ("Superradiance, $10^{7}$",
     [(rf"$\tau_d=10^{e}$ yr", f"gw190728_env_sr_eq9_taud1e{e}") for e in (5, 6, 7, 8)]),
    ("Superradiance, H1+L1+V1, BayesWave",
     [(r"$\tau_d=10^{6}$ yr", "gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8"),
      (r"$\tau_d=10^{6}$ yr, phase sampled",
       "gw190728_env_hlv_gwtc21psd_phase_sampled_sr_eq9_taud1e6_rho2e8")]),
]


def robustness_figure(name="lnb_robustness_paper"):
    base = np.array([LNB[l][0] for l in BASE])
    rows = [("Baseline (4 runs)", base.mean(), base.std(ddof=1), None, None)]
    for group, members in GROUPS:
        rows.append((group, None, None, None, None))
        for text, label in (m for m in members if m[1] in LNB):   # skip failed runs
            roy = ROY[int(label.split("taud1e")[1][0])] if "sr_eq9" in label else None
            rows.append(("    " + text, *LNB[label], roy, label))

    fig, ax = plt.subplots(figsize=(5.2, 0.27 * len(rows) + 0.8))
    ax.axvspan(-1, 1, color="0.9", zorder=0, label=r"$|\ln B|<1$")
    ax.axvline(base.mean(), color="C0", lw=0.8, ls="--", zorder=1)
    for y, (text, lnb, err, roy, _) in enumerate(rows):
        if lnb is None:
            continue
        ax.errorbar(lnb, y, xerr=err, fmt="o", color="C0", ms=4, capsize=2, zorder=3)
        if roy is not None:
            ax.plot(roy, y, marker="D", color="C3", ms=4, zorder=3)
    ax.plot([], [], "D", color="C3", ms=4, label="Roy et al.")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row[1] is None:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\ln\mathcal{B}^{\rm env}_{\rm vac}$")
    ax.legend(fontsize=7, loc="upper right")
    _save(fig, name)


def background_figure(name="injection_background_paper", hi=8.0):
    """ln B of the vacuum injections of both campaigns against the loudness
    of the signal: the event's background is the injections as loud and as
    heavy as it is.  The signals made with SEOBNRv4PHM show what an error
    of the vacuum waveform gives, and the environmental ones what a real
    cloud of the size the event's posterior suggests gives."""
    observed = np.mean([LNB[l][0] for l in BASE])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.axvspan(EVENT_SNR - MATCH, EVENT_SNR + MATCH, color="0.9", zorder=0,
               label=f"within {MATCH} of the event's SNR")
    off = []                                   # the glitch segment, far above
    for tag, label, style in CAMPAIGNS:
        snr = network_snr(tag)
        lnb = {key: agnostic for key, (agnostic, _) in background(tag).items()}
        on = [key for key in lnb if lnb[key] <= hi]
        ax.plot([snr[k] for k in on], [lnb[k] for k in on], "o", ms=4,
                label=f"{label} ($N={len(lnb)}$)", **style)
        off += [(snr[k], lnb[k]) for k in set(lnb) - set(on)]
    for tag, run, label, style in INJECTED:
        points = injected(tag, run).values()
        ax.plot([p[0] for p in points], [p[1] for p in points],
                label=f"{label} ($N={len(points)}$)", **style)
    ax.plot(EVENT_SNR, observed, "*", color="C3", ms=12, label=f"GW190728 ({observed:+.2f})")
    if off:
        x = np.mean([s for s, _ in off])
        values = ", ".join(f"{v:+.0f}" for _, v in sorted(off))
        ax.annotate(f"$\\ln\\mathcal{{B}}={values}$\n(glitch segment)", xy=(x, hi),
                    xytext=(x - 0.4, hi - 1.6), ha="left", fontsize=7, color="0.3",
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.set_ylim(top=hi)
    ax.set_xlabel("network signal-to-noise ratio of the injected signal")
    ax.set_ylabel(r"$\ln\mathcal{B}^{\rm env}_{\rm vac}$")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    _save(fig, name)


def vacuum_figure(name="vacuum_vs_gwtc_paper"):
    """Fig. 2: the vacuum posterior for the exact likelihood, as the text
    quotes it, over the GWTC-2.1 release's (script 01 draws it from the
    run as sampled)."""
    with h5py.File(Path("data_cache") / CATALOGUE["GW190728_064510"], "r") as f:
        catalogue = f["C01:IMRPhenomXPHM/posterior_samples"][()]
    compare_with_catalog(SimpleNamespace(posterior=exact_posterior("gw190728_vacuum")),
                         catalogue, name=name)


def posterior_figures():
    """Figs. 3-4 from the four baseline runs combined, as the text quotes
    them, for the exact likelihood (script 02 draws them from the first
    run alone, as sampled)."""
    result = lambda *labels: SimpleNamespace(posterior=pd.concat(
        [exact_posterior(l) for l in labels], ignore_index=True))
    env = result(*BASE)
    environment_corner(env, name="env_corner_paper")
    vacuum_vs_env_overlay(result("gw190728_vacuum"), env, name="mc_chieff_shift_paper")


def sr_comparison_figure(name="injection_sr_comparison_paper"):
    """ln B under the superradiance prior (tau_d = 1e6 yr) against agnostic
    ln B for the injections re-analysed with it: the first campaign's
    subset, chosen by rank, the second campaign's injections as loud as
    the event, and the signals of scripts 31-32, next to GW190728."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    points = []
    for tag, label, style in CAMPAIGNS:
        errors = json.loads(Path(f"injection_sr_lnb{tag}.json").read_text())
        keys = (matched_keys(tag) & set(errors) if tag
                else {key for key, _ in select_sr_subset("injection_lnb.json")})
        values = background(tag)
        shown = sorted(k for k in keys if abs(values[k][0]) < 100)   # the glitch segment is off scale
        x, y = ([values[k][i] for k in shown] for i in (0, 1))
        ax.errorbar(x, y, yerr=[errors[k]["err"] for k in shown], fmt="o", ms=4, capsize=2,
                    label=f"{label} ($N={len(shown)}$)", **style)
        points += x + y
    for tag, run, label, style in INJECTED:
        errors = json.loads(Path(f"injection_sr_lnb{tag}.json").read_text())
        values = injected(tag, run)
        x, y = ([values[k][i] for k in sorted(values)] for i in (1, 2))
        ax.errorbar(x, y, yerr=[errors[k]["err"] for k in sorted(values)], capsize=2,
                    label=f"{label} ($N={len(values)}$)", **style)
        points += x + y
    event = (np.mean([LNB[l][0] for l in BASE]), LNB["gw190728_env_sr_eq9_taud1e6_rho2e8"][0])
    ax.plot(*event, "*", color="C3", ms=12, label="GW190728", zorder=4)
    ax.axhline(ROY[6], ls=":", color="C3", lw=0.8, label=f"Roy et al. ({ROY[6]})")
    lim = (min(points) - 0.7, max(points + [ROY[6]]) + 0.7)
    ax.plot(lim, lim, color="grey", lw=0.8, zorder=0)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"$\ln\mathcal{B}^{\rm env}_{\rm vac}$, agnostic priors")
    ax.set_ylabel(r"$\ln\mathcal{B}^{\rm env}_{\rm vac}$, superradiance priors")
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, name)


def shift_figure(name="injection_shifts_paper", min_ess=30):
    """How much ln B rises with Virgo (x) and with Virgo and the catalogue's
    spectra (y), agnostic priors, for the vacuum signals of both campaigns,
    those made with a second waveform family and the environmental ones,
    next to GW190728: the rise that takes the event to the published value.
    Reweightings on fewer than min_ess effective samples are left out."""
    networks = ("H1+L1+V1", "H1+L1+V1, BayesWave")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(0, color="grey", lw=0.6, zorder=0)
    ax.axvline(0, color="grey", lw=0.6, zorder=0)
    series = [(tag, label, dict(marker="o", ls="none", ms=4, **style)) for tag, label, style in CAMPAIGNS]
    for tag, label, style in series + [(t, l, s) for t, _, l, s in INJECTED]:
        entries = json.loads(Path(f"injection_virgo{tag}.json").read_text())
        shifts = [[e[n]["agnostic"]["lnb"] - e["H1+L1"]["agnostic"]["lnb"] for n in networks]
                  for key, e in entries.items()
                  if key != GLITCH and min(e[n]["agnostic"]["ess"] for n in networks) >= min_ess]
        ax.plot(*zip(*shifts), label=f"{label} ($N={len(shifts)}$)", **style)
    base = np.mean([LNB[l][0] for l in BASE])
    ax.plot(LNB["gw190728_env_hlv"][0] - base, LNB["gw190728_env_hlv_gwtc21psd"][0] - base,
            "*", color="C3", ms=12, label="GW190728")
    ax.set_xlabel(r"$\Delta\ln\mathcal{B}$, adding Virgo")
    ax.set_ylabel(r"$\Delta\ln\mathcal{B}$, adding Virgo and the catalogue's spectra")
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, name)


def _save(fig, name):
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    print(f"full-grid ln B, mass-cut offset {OFFSET:+.4f} included")
    robustness_figure()
    background_figure()
    sr_comparison_figure()
    shift_figure()
    vacuum_figure()
    posterior_figures()
    print("figures/*_paper.* written")
