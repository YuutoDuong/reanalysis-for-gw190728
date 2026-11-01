"""How robust is the tail of the Virgo time slides (script 33)?

Reads virgo_time_slides.json and prints, for each prior and spectrum, the
fraction of slides whose shift of ln B on adding Virgo reaches the event's
zero-lag shift under each selection of slides (all; no glitch; effective
sample size >= 30; both), the same for slides within an hour of the event,
how many of the slides that reach it have a low effective sample size, and
the chance that at least one of 28 events shows such a rise.
"""
import json
from pathlib import Path

import numpy as np

N_EVENTS = 28
T0 = 1248331528.5
r = json.loads(Path("virgo_time_slides.json").read_text())
starts = np.array(r["starts"])
glitch = (np.array(r["max_whitened"]) > 6.0) | (np.abs(np.array(r["kurtosis"])) > 0.5)
near = np.abs(starts + 4 - T0) <= 3600

for name, s in r["shifts"].items():
    x, ess, zero = np.array(s["slides"]), np.array(s["ess"]), s["zero_lag"]
    print(f"{name}: zero lag {zero:+.2f} (ESS {s['ess_zero_lag']:.0f})")
    for title, keep in (("all", np.ones_like(glitch)), ("no glitch", ~glitch),
                        ("ESS >= 30", ess >= 30), ("both", ~glitch & (ess >= 30)),
                        ("both, within 1 h", ~glitch & (ess >= 30) & near)):
        k, n = int(np.sum(x[keep] >= zero)), int(keep.sum())
        p = k / n
        print(f"  {title:17s} {k:3d} of {n:4d}: p = {p:.4f} (1 in {1 / p if k else np.inf:.0f}); "
              f"sd {x[keep].std(ddof=1):.2f}; at least one of {N_EVENTS} events {1 - (1 - p) ** N_EVENTS:.2f}")
    reach = x >= zero
    print(f"  of the {reach.sum()} slides reaching it, {np.sum(reach & (ess < 30))} have ESS < 30 "
          f"and {np.sum(reach & glitch)} a glitch")
