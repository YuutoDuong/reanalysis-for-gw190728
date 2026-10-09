"""Section V.F -- Superradiance-prior background subset.

The 50-injection campaign (script 05) calibrates the *agnostic* ln B.
Roy et al.'s headline ln B ~ 3.5 instead uses their superradiance priors
(SM Eqs. 9-10), so the background for that statistic is untested.  This
script re-analyses a selected subset of the same vacuum injections --
the highest agnostic ln B draws plus a stratified sample -- under the
superradiance prior at tau_d = 10^6 yr (their headline configuration).

The question it answers is mechanistic rather than a rate: does the SR
prior amplify or suppress noise fluctuations that already look
environmental?  For the real GW190728 it raises ln B from -1.29
(agnostic) to -0.03 (tau_d = 1e6 yr, rho_max = 2e8), versus their 3.5.

Runs the redshifted (second) campaign's subset; the first campaign's is
kept in injection_sr_lnb.json.  Cached vacuum runs are reused, so only ~13
environmental runs execute.
Expected wall time: ~1.5-2.5 days; restartable at any point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.injections import (run_sr_campaign, plot_sr_comparison,
                                   false_alarm_probability)

TAU_D_YR = 1e6          # Roy et al.'s headline delay time
OBSERVED_SR = -0.03     # our GW190728 ln B under the same SR prior and box
                        # (script 04, gw190728_env_sr_eq9_taud1e6_rho2e8);
                        # raw, like the campaign values (paper: +0.18 to all,
                        # figure via script 11)

results = run_sr_campaign(tau_d_yr=TAU_D_YR)

sr = [v["lnb_sr"] for v in results.values()]
agnostic = [v["lnb_agnostic"] for v in results.values()]
print(f"\n{len(sr)} injections re-analysed with SR priors "
      f"(tau_d = {TAU_D_YR:.0e} yr).")
for label, values in (("SR", sr), ("agnostic", agnostic)):
    p, e = false_alarm_probability(values, 3.5)
    print(f"  {label:9s}: P(ln B >= 3.5) = {p:.2f} +/- {e:.2f} "
          f"(subset only, not the full-campaign rate)")

plot_sr_comparison(results, observed_sr=OBSERVED_SR)
print("figures/injection_sr_comparison.png written.")
