"""Section V.D (tests 1-4) -- Robustness suite on GW190728.

Prior sweeps, sampler comparison, waveform systematics and PSD/detector
variations, all resumable.  Produces the ln B robustness figure (Fig 3)
and a machine-readable table.

Expected wall time: ~2-4 weeks total; run in batches (each finished run
is cached, so the script can be interrupted and restarted freely).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig
from scalar_env.robustness import full_suite, run_suite, plot_lnb_robustness

# npool does not enter the run label, so already-cached single-core
# results load unchanged and only the remaining runs go parallel.
NPOOL = max(1, (os.cpu_count() or 2) - 2)

# The baseline density box MUST match script 02's RHO_MAX: every
# non-prior variant (sampler, waveform, PSD, network) inherits it, and a
# forest plot mixing prior volumes is not interpretable, because ln B
# carries an explicit -ln(rho_max) Occam term.
RHO_MAX = 2e8

env_base = RunConfig(label="gw190728_env", model="environment", npool=NPOOL,
                     prior_kwargs=dict(rho_max_gcm3=RHO_MAX))
vac_base = RunConfig(label="gw190728_vacuum", model="vacuum", npool=NPOOL)

# Repeat seeds of the baseline: the quoted dynesty ln Z error appears to
# understate run-to-run scatter for the environmental model (two
# nominally identical runs differed by 0.49), so measure it directly.
seeds = [env_base.variant(f"seed{s}", seed=s) for s in (1, 2)]

rows = run_suite(full_suite(env_base, vac_base)
                 + [(c, vac_base) for c in seeds])

Path("robustness_lnb.json").write_text(json.dumps(rows, indent=2))
plot_lnb_robustness(rows)
print("robustness_lnb.json and figures/lnb_robustness.png written.")
