"""Does each SR run sample the agnostic posterior restricted to the SR region?

Same likelihood, same prior shape: the SR posterior must equal the agnostic
posterior truncated to the SR region.  A mismatch (KS p-value ~ 0) means
the constrained run did not explore its region, which would also explain
a negative truncation-identity residual (script 08).
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, ".")
import bilby
bilby.core.utils.logger.setLevel("ERROR")
from scalar_env.inference import RunConfig, build_priors, OUTDIR

KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
COLS = ("log_likelihood", "rho_phi", "alpha_cloud", "chirp_mass")


def post(*labels):
    return pd.concat([bilby.result.read_in_result(
        str(OUTDIR / l / f"{l}_result.json")).posterior for l in labels],
        ignore_index=True)


AGN = {2e8: post("gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
                 "gw190728_env_rho_max2e8"),
       1e7: post("gw190728_env_rho_max1e7")}

for box, suffix in ((2e8, "_rho2e8"), (1e7, "")):
    agn = AGN[box]
    print(f"\n=== box {box:.0e}   agnostic samples: {len(agn)}   "
          f"max lnL {agn.log_likelihood.max():.2f}")
    for e in (5, 6, 7, 8):
        lab = f"gw190728_env_sr_eq9_taud1e{e}{suffix}"
        sr = post(lab)
        pri = build_priors(RunConfig(label="_", model="environment", prior_kwargs=dict(
            rho_max_gcm3=box, superradiance_tau_d_yr=10.0 ** e)), 1248331528.5)
        inside = np.asarray(pri.evaluate_constraints(
            {k: agn[k].to_numpy() for k in KEYS}), dtype=bool)
        ref = agn[inside]
        print(f"tau_d 1e{e}: SR n={len(sr)}, agnostic-inside n={len(ref)}; "
              f"max lnL SR {sr.log_likelihood.max():.2f} vs ref {ref.log_likelihood.max():.2f}")
        for c in COLS:
            p = ks_2samp(sr[c], ref[c]).pvalue
            q = lambda x: np.quantile(x, [0.05, 0.5, 0.95])
            s, r = q(sr[c]), q(ref[c])
            fmt = (lambda v: f"{v:.3g}")
            print(f"   {c:15s} SR [{', '.join(map(fmt, s))}]  "
                  f"ref [{', '.join(map(fmt, r))}]  KS p={p:.2g}")
