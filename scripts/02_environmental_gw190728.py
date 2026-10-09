"""Section V.B -- Environmental PE on GW190728 (the central result).

Runs the scalar-cloud model with baseline (Roy et al.) priors, computes
ln B^env_vac against the vacuum run from script 01, cross-checks it with
the Savage-Dickey ratio, and produces the headline figures.

Expected wall time: ~12-48 h (laptop).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import (RunConfig, run_pe, ln_bayes_factor,
                                  savage_dickey_lnb, build_priors)
from scalar_env.analysis import (environment_corner, vacuum_vs_env_overlay,
                                 phase_difference_plot, mu_s_credible_interval)

NPOOL = max(1, (os.cpu_count() or 2) - 2)

# Density prior bound.  Roy et al.'s GW190728 posterior (their Figs. 5-6)
# occupies 3e7-1.1e8 g/cm^3, and 91.5% of their superradiance prior's mass
# lies above 1e7 -- so a 1e7 box cannot test their claim, it excludes the
# region under test.  The 1e7 and 1e6 boxes are retained as prior-
# sensitivity variants in robustness.py.
RHO_MAX = 2e8
TRIGGER = 1248331528.5      # only sets the geocent_time prior

vacuum = run_pe(RunConfig(label="gw190728_vacuum", model="vacuum"))
env_config = RunConfig(label="gw190728_env", model="environment",
                       npool=NPOOL, prior_kwargs=dict(rho_max_gcm3=RHO_MAX))
environ = run_pe(env_config)

lnb, err = ln_bayes_factor(environ, vacuum)
print(f"ln B^env_vac (GW190728) = {lnb:+.2f} +/- {err:.2f}   "
      f"[Roy et al.: +3.5]")
print("Savage-Dickey cross-check: ln B ~ "
      f"{savage_dickey_lnb(environ, build_priors(env_config, TRIGGER)):+.2f}")

mid, lo, hi = mu_s_credible_interval(environ)
print(f"mu_s = {mid:.2e} eV, 90% CI [{lo:.2e}, {hi:.2e}]   "
      f"[Roy et al.: ~1e-12 eV]")

environment_corner(environ)               # Fig 1
vacuum_vs_env_overlay(vacuum, environ)    # Fig 2  (Mc / chi_eff shifts)
phase_difference_plot(environ)            # Fig 4  (best-fit dephasing)
print("Figures written to figures/.")
