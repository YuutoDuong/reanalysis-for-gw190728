"""Normalisation checks: initial live points, mass cut, MultiNest volume."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import bilby
from scalar_env.inference import RunConfig, build_priors, OUTDIR
from scalar_env.priors import vacuum_priors

bilby.core.utils.logger.setLevel("ERROR")
TRIG = 1248331528.5
pri = lambda kw: build_priors(RunConfig(label="_", model="environment", prior_kwargs=kw), TRIG)
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
load = lambda l: bilby.result.read_in_result(str(OUTDIR / l / f"{l}_result.json"))

# 1. do the dead points include constraint violators or -1.8e308 likelihoods?
for label, kw in (("gw190728_env_fixfid", dict(rho_max_gcm3=2e8)),
                  ("gw190728_env_fixfid_sr_eq9_taud1e6_rho2e8",
                   dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6))):
    r = load(label)
    ns = r.nested_samples
    ok = np.asarray(pri(kw).evaluate_constraints({k: ns[k].to_numpy() for k in KEYS}), bool)
    ll = ns["log_likelihood"].to_numpy()
    print(f"{label}: {len(ns)} dead points, violating constraints {np.sum(~ok)}, "
          f"lnL < -1e300: {np.sum(ll < -1e300)}, lowest lnL {np.sort(ll)[:3]}")
    print("   meta sampler kwargs:", {k: v for k, v in r.meta_data.get("sampler_kwargs", {}).items()
                                     if k in ("nlive", "sample", "naccept", "nact", "bound", "dlogz", "maxmcmc")}
          if isinstance(r.meta_data, dict) else None)
    print("   num_likelihood_evaluations:", getattr(r, "num_likelihood_evaluations", None),
          " sampling_time:", getattr(r, "sampling_time", None))

# 2. masses in posteriors vs the vacuum cut m_{1,2} >= 5
for label in ("gw190728_vacuum", "gw190728_env", "gw190728_env_fixfid"):
    p = load(label).posterior
    print(f"{label}: min mass_2 {p['mass_2'].min():.2f}, max mass_1 {p['mass_1'].max():.2f}")

# 3. prior volume fractions
N = 2_000_000
vp = vacuum_priors(TRIG)
for kw in (dict(rho_max_gcm3=2e8), dict(rho_max_gcm3=1e7),
           dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6)):
    ep = pri(kw)
    d = {k: np.asarray(ep[k].sample(N)) for k in KEYS}
    cloud = np.asarray(ep.evaluate_constraints(d), bool)
    mc = np.asarray(vp.evaluate_constraints({k: d[k] for k in ("chirp_mass", "mass_ratio")}), bool)
    print(f"{kw}: F_constraints {cloud.mean():.4f}  F_masscut|constraints {mc[cloud].mean():.4f}"
          f"  (vacuum F_masscut {mc.mean():.4f})")
