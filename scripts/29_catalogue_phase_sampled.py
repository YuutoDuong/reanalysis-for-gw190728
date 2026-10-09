"""The catalogue noise model with the phase sampled.

bilby's analytic phase marginalisation assumes the strain rotates as a
whole with the reference phase, which IMRPhenomXPHM's higher multipoles
and precession break: for GW190728 it lowers the vacuum evidence by
~1.9 and the environmental one by ~1.4, inflating ln B by ~0.6, and it
pushes the mass ratio toward 1 (median 0.80 against 0.64 when the phase
is sampled, and 0.60 in GWTC-2.1, whose analysis sampled it).  Repeat the
catalogue-noise-model runs of script 20 (H1+L1+V1, BayesWave PSDs,
rho_max = 2e8) sampling the phase: vacuum, superradiance at
tau_d = 1e6 yr, agnostic.  Runs beside the injection campaign on 5 cores
at low priority: ~2 days.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig, run_pe, ln_bayes_factor, mass_cut_offset

NOISE = dict(detectors=("H1", "L1", "V1"), psd_source="gwtc21",
             phase_marginalization=False, npool=5)

vac = run_pe(RunConfig(label="gw190728_vacuum_hlv_gwtc21psd_phase_sampled",
                       model="vacuum", **NOISE))
agn = RunConfig(label="gw190728_env_hlv_gwtc21psd_phase_sampled", model="environment",
                prior_kwargs=dict(rho_max_gcm3=2e8), **NOISE)
sr = agn.variant("sr_eq9_taud1e6_rho2e8",
                 prior_kwargs=dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6))
offset = mass_cut_offset()
print()
for name, cfg in (("superradiance, tau_d = 1e6", sr), ("agnostic", agn)):
    lnb, err = ln_bayes_factor(run_pe(cfg), vac)
    print(f"H1+L1+V1, GWTC-2.1 PSDs, phase sampled, {name:28s} "
          f"ln B = {lnb + offset:+.2f} +- {err:.2f} (relative binning; reweight with 17)",
          flush=True)
