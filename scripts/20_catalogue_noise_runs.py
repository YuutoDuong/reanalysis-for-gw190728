"""The noise model Roy et al. most likely used.

Their network and PSD method are not stated.  The standard LVK choice for
GW190728 is all three detectors with the BayesWave PSDs released with
GWTC-2.1; analyse the event that way under the vacuum, agnostic and
superradiance (tau_d = 1e6 yr) models, all at rho_max = 2e8.  ~8 h.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig, run_pe, ln_bayes_factor, mass_cut_offset

NPOOL = max(1, (os.cpu_count() or 2) - 2)
NOISE = dict(detectors=("H1", "L1", "V1"), psd_source="gwtc21", npool=NPOOL)

vac = run_pe(RunConfig(label="gw190728_vacuum_hlv_gwtc21psd", model="vacuum", **NOISE))
agn = RunConfig(label="gw190728_env_hlv_gwtc21psd", model="environment",
                prior_kwargs=dict(rho_max_gcm3=2e8), **NOISE)
sr = agn.variant("sr_eq9_taud1e6_rho2e8",
                 prior_kwargs=dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6))
offset = mass_cut_offset()
print()
for name, cfg in (("agnostic", agn), ("superradiance, tau_d = 1e6", sr)):
    lnb, err = ln_bayes_factor(run_pe(cfg), vac)
    print(f"H1+L1+V1, GWTC-2.1 PSDs, {name:28s} ln B = {lnb + offset:+.2f} +- {err:.2f}")
