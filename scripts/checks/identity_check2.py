"""Which input of the truncation identity is off: P or f?

Three samplers agree on the SR evidence at tau_d = 1e6, rho_max = 2e8
(-0.03, -0.07, -0.32) and on the agnostic one (-1.29, -1.27, -1.43), so
ln Z_SR - ln Z_agn ~ +1.0..1.4, while ln(P/f) from script 08 is +2.19.
Recompute P from each agnostic posterior separately (dynesty x4,
MultiNest, nessai) and f two independent ways.
"""
import sys
import numpy as np
sys.path.insert(0, ".")
import bilby
bilby.core.utils.logger.setLevel("ERROR")
from scalar_env.inference import RunConfig, build_priors

TRIG = 1248331528.5
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
res = lambda l: bilby.result.read_in_result(f"outdir/{l}/{l}_result.json")
mk = lambda **kw: build_priors(RunConfig(label="_", model="environment", prior_kwargs=kw), TRIG)
agn = mk(rho_max_gcm3=2e8)
sr = mk(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6)

def inside(pri, post):
    return np.asarray(pri.evaluate_constraints({k: post[k].to_numpy() for k in KEYS}), dtype=bool)

print("== P = fraction of each agnostic posterior inside the tau_d = 1e6 SR region")
for l in ("gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
          "gw190728_env_rho_max2e8", "gw190728_env_pymultinest", "gw190728_env_nessai"):
    p = res(l).posterior
    print(f"  {l:28s} n={len(p):6d}  P = {inside(sr, p).mean():.3f}")

print("== f, method 1: independent draws of the four keys (script 08)")
rng = np.random.default_rng(3)
n = 2_000_000
s = {k: np.asarray(agn[k].sample(n)) for k in KEYS}
fa = np.asarray(agn.evaluate_constraints(s), bool); fs = np.asarray(sr.evaluate_constraints(s), bool)
print(f"  f_abs(agn) {fa.mean():.4f}  f_abs(SR) {fs.mean():.5f}  f = {fs.mean() / fa.mean():.4f}")

print("== f, method 2: bilby's own normalize_constraint_factor over all search keys")
keys = tuple(k for k in agn if not isinstance(agn[k], (bilby.core.prior.Constraint,
                                                         bilby.core.prior.DeltaFunction)))
ra = agn.normalize_constraint_factor(keys); rs = sr.normalize_constraint_factor(keys)
print(f"  1/ratio agn {1/ra:.4f}  1/ratio SR {1/rs:.5f}  f = {ra / rs:.4f}")

print("== inside fraction of the new SR runs (must be 100%)")
for l in ("gw190728_env_sr_eq9_taud1e6_rho2e8_nessai", "gw190728_env_sr_eq9_taud1e6_rho2e8_accwalk60"):
    print(f"  {l:44s} {inside(sr, res(l).posterior).mean():.1%}")
