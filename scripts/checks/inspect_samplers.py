"""ln Z by sampler: dynesty and nessai normalise to the constrained prior,
MultiNest to the full prior box (constraints enter only via the likelihood)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import bilby
from scalar_env.inference import OUTDIR

bilby.core.utils.logger.setLevel("ERROR")
for label in ("gw190728_vacuum", "gw190728_vacuum_seed0", "gw190728_vacuum_seed1",
              "gw190728_vacuum_seed2", "gw190728_vacuum_nessai", "gw190728_vacuum_pymultinest",
              "gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
              "gw190728_env_rho_max2e8", "gw190728_env_fixfid",
              "gw190728_env_nessai", "gw190728_env_pymultinest"):
    f = OUTDIR / label / f"{label}_result.json"
    if not f.exists():
        print(f"{label:32s} missing"); continue
    r = bilby.result.read_in_result(str(f))
    print(f"{label:32s} lnZ {r.log_evidence:12.3f} +- {r.log_evidence_err:.3f}   "
          f"min m2 {r.posterior['mass_2'].min():.2f}  frac m2<5.5 {np.mean(r.posterior['mass_2'] < 5.5):.4f}")

# injections: injected masses and the agnostic lnB values
for name in ("injection_lnb.json", "injection_results.json"):
    p = Path(name)
    if p.exists():
        print("found", name)
inj = sorted(Path(".").glob("injection*.json"))
print([p.name for p in inj])
