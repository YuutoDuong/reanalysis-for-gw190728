"""Paired comparison of the two injection campaigns (no sampling).

Both use the same noise segments and source parameters; the second
redshifts the masses (1 + z of each injection's own distance), which the
first omitted.  Prints the per-injection shift in ln B, the new
background statistics and GW190728's place in it, and the superradiance
subset.  Runs on partial results too, so it can be used mid-campaign.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import mass_cut_offset
from scalar_env.injections import false_alarm_probability

OFFSET = mass_cut_offset()
GLITCH = "inj028"
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
        "gw190728_env_rho_max2e8"]
rows = {l: b for l, b, _ in json.loads(Path("robustness_lnb.json").read_text())}
observed = np.mean([rows[l] for l in BASE]) + OFFSET
load = lambda name: json.loads(Path(name).read_text()) if Path(name).exists() else {}

old, new = load("injection_lnb.json"), load("injection_lnb_det.json")
paired = sorted(set(old) & set(new))
print(f"{len(new)} of 50 redshifted injections done; {len(paired)} paired with the first campaign")
if paired:
    shift = np.array([new[k]["lnb"] - old[k]["lnb"] for k in paired if k != GLITCH])
    print(f"  clean pairs: ln B shift median {np.median(shift):+.2f}, "
          f"mean {shift.mean():+.2f} +- {shift.std(ddof=1) / np.sqrt(shift.size):.2f}, "
          f"range [{shift.min():+.2f}, {shift.max():+.2f}]")

for name, data in (("first (unredshifted)", old), ("second (redshifted)", new)):
    if not data:
        continue
    lnb = np.array([v["lnb"] for v in data.values()]) + OFFSET
    clean = np.array([v["lnb"] + OFFSET for k, v in data.items() if k != GLITCH])
    p, e = false_alarm_probability(lnb, observed)
    print(f"{name:22s} N={lnb.size}: median {np.median(lnb):+.2f}, 90% in "
          f"[{np.quantile(lnb, 0.05):+.1f}, {np.quantile(lnb, 0.95):+.1f}], largest clean "
          f"{clean.max():+.2f}; P(ln B >= {observed:+.2f}) = {p:.2f} +- {e:.2f}")

for name in ("injection_sr_lnb.json", "injection_sr_lnb_det.json"):
    sr = load(name)
    if sr:
        rise = [v["lnb_sr"] - v["lnb_agnostic"] for k, v in sr.items() if k != GLITCH]
        top = max(sr, key=lambda k: sr[k]["lnb_sr"] if k != GLITCH else -np.inf)
        print(f"{name}: {len(sr)} entries; clean median rise {np.median(rise):+.2f}; "
              f"largest {top}: {sr[top]['lnb_sr'] + OFFSET:+.2f} "
              f"(agnostic {sr[top]['lnb_agnostic'] + OFFSET:+.2f})")
