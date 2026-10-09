"""Every ln B quoted in the text, from the results files (no sampling).

Sections V-VI, the abstract and the conclusion quote these numbers; this
script prints them with the mass-cut offset (inference.mass_cut_offset)
applied, so none is computed by hand.  Table II's identity column comes
from script 08, Appendix B from script 14, the figures from script 11.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env.inference import ln_bayes_factor, mass_cut_offset, OUTDIR
from scalar_env.injections import false_alarm_probability, select_sr_subset

bilby.core.utils.logger.setLevel("WARNING")
OFFSET = mass_cut_offset()
LNB = {l: (b + OFFSET, e) for l, b, e in
       json.loads(Path("robustness_lnb.json").read_text())}
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
        "gw190728_env_rho_max2e8"]
GLITCH = "inj028"


def lnb_of(env_label, vac_label):
    load = lambda l: bilby.result.read_in_result(str(OUTDIR / l / f"{l}_result.json"))
    lnb, err = ln_bayes_factor(load(env_label), load(vac_label))
    return lnb + OFFSET, err


print(f"mass-cut offset {OFFSET:+.4f}\n")
base = np.array([LNB[l][0] for l in BASE])
observed = base.mean()
print(f"agnostic, 2e8: runs {np.round(base, 2)}, mean {observed:+.2f}, "
      f"sd {base.std(ddof=1):.2f}")
for label, (b, e) in LNB.items():
    if label not in BASE:
        print(f"  {label:44s} {b:+6.2f} +- {e:.2f}")

print("\nsuperradiance, tau_d = 1e6 yr, 2e8, four direct estimates:")
direct = {"dynesty rwalk (production)": LNB["gw190728_env_sr_eq9_taud1e6_rho2e8"],
          "nessai": lnb_of("gw190728_env_sr_eq9_taud1e6_rho2e8_nessai",
                           "gw190728_vacuum_nessai"),
          "dynesty acceptance-walk 60": lnb_of(
              "gw190728_env_sr_eq9_taud1e6_rho2e8_accwalk60", "gw190728_vacuum"),
          "dynesty rwalk, shared reference": lnb_of(
              "gw190728_env_fixfid_sr_eq9_taud1e6_rho2e8", "gw190728_vacuum")}
for name, (b, e) in direct.items():
    print(f"  {name:32s} {b:+.2f} +- {e:.2f}")
values = np.array([b for b, _ in direct.values()])
print(f"  mean {values.mean():+.2f}, sd {values.std(ddof=1):.2f}")
print("  shared-reference agnostic:",
      "{:+.2f} +- {:.2f}".format(*lnb_of("gw190728_env_fixfid", "gw190728_vacuum")))

agn = json.loads(Path("injection_lnb.json").read_text())
lnb = np.array([v["lnb"] for v in agn.values()]) + OFFSET
clean = np.array([v["lnb"] + OFFSET for k, v in agn.items() if k != GLITCH])
p, e = false_alarm_probability(lnb, observed)
print(f"\nbackground: N = {lnb.size}, median {np.median(lnb):+.2f} "
      f"(clean {np.median(clean):+.2f}), 90% of values in "
      f"[{np.quantile(lnb, 0.05):+.1f}, {np.quantile(lnb, 0.95):+.1f}], "
      f"largest clean {clean.max():+.2f}, {GLITCH} {agn[GLITCH]['lnb'] + OFFSET:+.0f}")
print(f"  P(lnB >= {observed:+.2f}) = {p:.2f} +- {e:.2f}; "
      f"clean injections >= 3.5: {(clean >= 3.5).sum()}")

sr = json.loads(Path("injection_sr_lnb.json").read_text())
print("\nsuperradiance subset (tau_d = 1e6 yr):")
rises = []
for key, _ in select_sr_subset("injection_lnb.json"):
    a, s = sr[key]["lnb_agnostic"] + OFFSET, sr[key]["lnb_sr"] + OFFSET
    print(f"  {key}: agnostic {a:+7.2f} -> SR {s:+6.2f} +- {sr[key]['err']:.2f}"
          f"   (rise {s - a:+.2f})")
    if key != GLITCH:
        rises.append(s - a)
print(f"  clean: median rise {np.median(rises):+.2f}, max {max(rises):+.2f}")
