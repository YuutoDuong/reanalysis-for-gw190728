"""SR posterior vs truncated agnostic posterior, every sampled parameter.

If the SR runs miss part of the posterior mass inside the SR region (a
mode in some parameter not yet compared), their evidence is low by the
missing fraction while alpha, rho_phi and Mc still look right.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
sys.path.insert(0, ".")
import bilby
bilby.core.utils.logger.setLevel("ERROR")
from scalar_env.inference import RunConfig, build_priors

KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
res = lambda l: bilby.result.read_in_result(f"outdir/{l}/{l}_result.json").posterior
sr_pri = build_priors(RunConfig(label="_", model="environment", prior_kwargs=dict(
    rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6)), 1248331528.5)
agn = pd.concat([res(l) for l in ("gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
                                  "gw190728_env_rho_max2e8", "gw190728_env_nessai",
                                  "gw190728_env_pymultinest")], ignore_index=True)
ins = np.asarray(sr_pri.evaluate_constraints({k: agn[k].to_numpy() for k in KEYS}), bool)
ref = agn[ins]
sr = pd.concat([res(l) for l in ("gw190728_env_sr_eq9_taud1e6_rho2e8",
                                 "gw190728_env_sr_eq9_taud1e6_rho2e8_nessai",
                                 "gw190728_env_sr_eq9_taud1e6_rho2e8_accwalk60")], ignore_index=True)
cols = ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
        "theta_jn", "psi", "ra", "dec", "geocent_time", "luminosity_distance",
        "alpha_cloud", "rho_phi", "chi_eff", "log_likelihood"]
print(f"SR samples {len(sr)}, agnostic-inside {len(ref)} (P = {ins.mean():.3f})")
print(f"{'param':20s} {'SR 5/50/95':>30s} {'ref 5/50/95':>30s} {'KS D':>6}")
for c in cols:
    a, b = sr[c].to_numpy(), ref[c].to_numpy()
    qa, qb = np.quantile(a, [.05, .5, .95]), np.quantile(b, [.05, .5, .95])
    d = ks_2samp(a, b).statistic
    f = lambda q: " ".join(f"{v:9.4g}" for v in q)
    print(f"{c:20s} {f(qa):>30s} {f(qb):>30s} {d:6.3f}{'  <--' if d > 0.08 else ''}")
