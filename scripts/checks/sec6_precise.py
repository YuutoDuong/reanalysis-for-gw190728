"""Posterior quantiles and the prior/posterior mass below 1e6 g/cm^3
quoted in Sec. V.B (four baseline runs combined)."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import bilby
bilby.core.utils.logger.setLevel("ERROR")
from scalar_env.inference import RunConfig, build_priors

res = lambda l: bilby.result.read_in_result(f"outdir/{l}/{l}_result.json")
vp = res("gw190728_vacuum").posterior
ep = pd.concat([res(l).posterior for l in ("gw190728_env", "gw190728_env_seed1",
                "gw190728_env_seed2", "gw190728_env_rho_max2e8")], ignore_index=True)
for name, p in (("vac", vp), ("env", ep)):
    for c in ("chirp_mass", "chi_eff", "mass_ratio"):
        lo, mid, hi = np.quantile(p[c], [0.05, 0.5, 0.95])
        print(f"{name} {c:11s} {mid:.4f} +{hi - mid:.4f} -{mid - lo:.4f}")
pri = build_priors(RunConfig(label="_", model="environment",
                             prior_kwargs=dict(rho_max_gcm3=2e8)), 1248331528.5)
draws = pri.sample(200_000)          # constrained prior draws
print(f"constrained prior P(rho < 1e6) = {np.mean(np.asarray(draws['rho_phi']) < 1e6):.4f}; "
      f"posterior {np.mean(ep.rho_phi < 1e6):.4f}")
