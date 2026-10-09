"""Quantify three assumptions found in the audit, from existing posteriors only.

1. alpha prior box: ours U[0.01, 0.75] (inherited from the discarded v1
   model) vs Roy et al.'s quoted bounds 0.008 and 0.5.  Narrowing the box
   is a truncation of our prior, so ln B changes by exactly ln(P/f):
   P = posterior mass with alpha <= 0.5, f = constrained-prior mass there.
2. Frame of the scalar mass: mu_s from detector- vs source-frame M.
3. What bilby 2.8.1 does with sample="rwalk" (Roy et al. used
   acceptance-walk, naccept = 60).
"""
import inspect
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import bilby
bilby.core.utils.logger.setLevel("ERROR")
from scalar_env.inference import RunConfig, build_priors, OUTDIR
from scalar_env.waveform import scalar_mass_ev

np.random.seed(2)
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
LABELS = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
          "gw190728_env_rho_max2e8"]

post = pd.concat([bilby.result.read_in_result(
    str(OUTDIR / l / f"{l}_result.json")).posterior for l in LABELS],
    ignore_index=True)
pri = build_priors(RunConfig(label="_", model="environment",
                             prior_kwargs=dict(rho_max_gcm3=2e8)), 1248331528.5)
s = {k: np.asarray(pri[k].sample(2_000_000)) for k in KEYS}
ok = np.asarray(pri.evaluate_constraints(s), dtype=bool)

print("== 1. alpha box [0.01, 0.75] -> [0.01, 0.5]")
P = np.mean(post["alpha_cloud"] <= 0.5)
f = np.mean(s["alpha_cloud"][ok] <= 0.5)
print(f"   P(alpha<=0.5 | d) = {P:.3f}   f = {f:.3f}   "
      f"shift in agnostic ln B = {np.log(P / f):+.2f}")
print(f"   alpha posterior: median {post['alpha_cloud'].median():.3f}, "
      f"90% CI [{post['alpha_cloud'].quantile(0.05):.3f}, "
      f"{post['alpha_cloud'].quantile(0.95):.3f}]")

print("== 2. scalar mass frame")
for col in ("total_mass", "total_mass_source"):
    if col in post:
        mu = scalar_mass_ev(post["alpha_cloud"], post[col])
        print(f"   {col:18s}: mu_s median {np.median(mu):.2e} eV, 90% CI "
              f"[{np.quantile(mu, 0.05):.2e}, {np.quantile(mu, 0.95):.2e}]")
    else:
        print(f"   {col} not in posterior")
if "redshift" in post:
    z = post["redshift"].median()
    print(f"   median z = {z:.3f}; (1+z)^2 = {(1 + z) ** 2:.2f} "
          "(physical density / sampled density)")

print("== 3. bilby dynesty sample methods")
from bilby.core.sampler import dynesty as dy
src = inspect.getsource(dy)
for line in src.splitlines():
    if "rwalk" in line and ("==" in line or "in (" in line or "elif" in line
                            or "sample" in line.split("rwalk")[0][-20:]):
        print("  ", line.strip()[:150])
