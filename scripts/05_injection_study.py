"""Sections IV.F and V.F -- Injection-recovery / false-alarm study.

Injects 50 vacuum-GR signals with GW190728-like parameters into real
off-source noise, runs the environmental vs vacuum comparison on each,
and reports how often pure-vacuum data produces ln B above the observed
GW190728 value.  Runs the redshifted (second) campaign; the first, whose
masses were left in the source frame, is kept in injection_lnb.json.

Expected wall time: ~10 days; fully restartable (per-injection caching).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.injections import (run_campaign, false_alarm_probability,
                                   plot_background)

OBSERVED_LNB = -1.29   # measured GW190728 baseline (script 02); the
                       # false-alarm question is how often pure-vacuum data
                       # reaches this.  Roy et al.'s 3.5 is reported too.
                       # Raw, like the campaign values it is compared with
                       # (the paper adds mass_cut_offset() = +0.18 to both).

lnb_values = run_campaign(n_injections=50)
p_fa, p_err = false_alarm_probability(lnb_values, OBSERVED_LNB)

print(f"\n{len(lnb_values)} vacuum injections analysed.")
print(f"False-alarm probability P(ln B >= {OBSERVED_LNB}) = "
      f"{p_fa:.3f} +/- {p_err:.3f}")
plot_background(lnb_values, OBSERVED_LNB)
print("figures/injection_background.png written.")
