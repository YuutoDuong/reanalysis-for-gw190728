"""Direct superradiance runs at the most favourable density bounds.

Script 16 predicts, from the 2e8 runs through the box identity, that the
superradiance ln B peaks for rho_max ~ 2-3e7 (at most +2.5 for tau_d =
1e6 yr, +2.9 for 1e8 yr, against Roy et al.'s 3.5 and 2.8); below ~1e7
its prediction rests on a sparsely sampled posterior tail.  Run the peaks,
and one low bound, directly.  ~7 h.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig, run_pe, ln_bayes_factor, mass_cut_offset

NPOOL = max(1, (os.cpu_count() or 2) - 2)
SCAN = [(6, 2.2e7, "2p2e7"), (6, 3e6, "3e6"), (8, 2.7e7, "2p7e7")]  # (log10 tau_d, rho_max, tag)

vac = run_pe(RunConfig(label="gw190728_vacuum", model="vacuum", npool=NPOOL))
offset = mass_cut_offset()
print()
for exp, rho_max, tag in SCAN:
    tau = 10.0 ** exp
    cfg = RunConfig(label=f"gw190728_env_sr_eq9_taud1e{exp}_rho{tag}",
                    model="environment", npool=NPOOL,
                    prior_kwargs=dict(rho_max_gcm3=rho_max, superradiance_tau_d_yr=tau))
    lnb, err = ln_bayes_factor(run_pe(cfg), vac)
    print(f"superradiance, tau_d = {tau:.0e} yr, rho_max = {rho_max:.1e}: "
          f"ln B = {lnb + offset:+.2f} +- {err:.2f}")
