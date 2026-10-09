"""MultiNest normalisation: do its dead points include constraint violators?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import bilby
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses as comp
from scalar_env.inference import RunConfig, build_priors, OUTDIR
from scalar_env.priors import vacuum_priors

bilby.core.utils.logger.setLevel("ERROR")
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
env_pri = build_priors(RunConfig(label="_", model="environment",
                                 prior_kwargs=dict(rho_max_gcm3=2e8)), 1248331528.5)
for label in ("gw190728_vacuum_pymultinest", "gw190728_env_pymultinest",
              "gw190728_vacuum_nessai", "gw190728_env_nessai"):
    r = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json"))
    ns = r.nested_samples
    if ns is None:
        print(label, "no nested samples stored"); continue
    m1, m2 = comp(ns["chirp_mass"].to_numpy(), ns["mass_ratio"].to_numpy())
    msg = f"{label:30s} {len(ns)} pts  m2<5: {np.mean(m2 < 5):.4f}"
    if "alpha_cloud" in ns:
        ok = np.asarray(env_pri.evaluate_constraints({k: ns[k].to_numpy() for k in KEYS}), bool)
        msg += f"  cloud-constraint violators: {np.mean(~ok):.4f}"
    ll = ns["log_likelihood"].to_numpy()
    msg += f"  lnL min {ll.min():.3g}"
    print(msg)

vp = vacuum_priors(0.0)
fr = []
for _ in range(5):
    d = vp.sample_subset(keys=["chirp_mass", "mass_ratio"], size=2_000_000)
    fr.append(np.mean(np.asarray(vp.evaluate_constraints(d), bool)))
F = np.mean(fr)
print(f"F_vac = {F:.5f} +- {np.std(fr) / np.sqrt(len(fr)):.5f};  -ln F = {-np.log(F):.4f}")
