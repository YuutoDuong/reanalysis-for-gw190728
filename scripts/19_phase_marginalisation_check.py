"""Is analytic phase marginalisation adequate for IMRPhenomXPHM?

bilby's phase marginalisation treats the coalescence phase as an overall
rotation of the strain, which is exact only for a single (2,2) mode; for a
precessing, higher-multipole waveform it is an approximation.  Repeat the
baseline vacuum and agnostic environmental runs sampling the phase instead
and compare ln B.  ~6-8 h.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig, run_pe, ln_bayes_factor, mass_cut_offset

NPOOL = max(1, (os.cpu_count() or 2) - 2)
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
        "gw190728_env_rho_max2e8"]

vac = RunConfig(label="gw190728_vacuum_phase_sampled", model="vacuum",
                npool=NPOOL, phase_marginalization=False)
env = RunConfig(label="gw190728_env_phase_sampled", model="environment",
                npool=NPOOL, prior_kwargs=dict(rho_max_gcm3=2e8),
                phase_marginalization=False)

lnb, err = ln_bayes_factor(run_pe(env), run_pe(vac))
offset = mass_cut_offset()
rows = {l: b for l, b, _ in json.loads(Path("robustness_lnb.json").read_text())}
baseline = np.mean([rows[l] for l in BASE]) + offset
print(f"\nagnostic ln B, rho_max = 2e8: phase marginalised {baseline:+.2f} (4 runs), "
      f"phase sampled {lnb + offset:+.2f} +- {err:.2f}")
