"""Posterior numbers quoted in Secs. V.A-C and VI.B, from the stored results
(no sampling).  Posteriors are those of the exact likelihood (script 17);
the Bayes factors themselves are printed by script 17, the background by
script 30."""
import glob
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import bilby
bilby.core.utils.logger.setLevel("ERROR")
from scalar_env.inference import RunConfig, build_priors, exact_posterior, savage_dickey_lnb
from scalar_env.waveform import scalar_mass_ev

TRIG = 1248331528.5
AGN = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2", "gw190728_env_rho_max2e8"]
q = lambda x, p=(0.05, 0.5, 0.95): np.quantile(np.asarray(x), p)
fmt = lambda x: "{:.3g} [{:.3g}, {:.3g}]".format(x[1], x[0], x[2])
mu_s = lambda p: scalar_mass_ev(p.alpha_cloud, p.total_mass_source)

print("=== A. vacuum")
vac = bilby.result.read_in_result("outdir/gw190728_vacuum/gw190728_vacuum_result.json")
print(f"lnZ_vac {vac.log_evidence:.2f} +/- {vac.log_evidence_err:.2f}; "
      f"lnZ_noise {vac.log_noise_evidence:.2f}; lnBF_sig {vac.log_bayes_factor:.2f} (as sampled)")
for f in sorted(glob.glob("outdir/gw190728_vacuum_seed*/*_result.json")):
    print("  seed run", Path(f).parent.name, f"{bilby.result.read_in_result(f).log_evidence:.2f}")
vp = exact_posterior("gw190728_vacuum")
for c in ("chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance"):
    print(f"  vac {c:20s} {fmt(q(vp[c]))}")

print("\n=== B. agnostic environmental, rho_max = 2e8, four runs pooled")
post = pd.concat([exact_posterior(l) for l in AGN], ignore_index=True)
for c in ("alpha_cloud", "rho_phi", "chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance", "redshift"):
    print(f"  env {c:20s} {fmt(q(post[c]))}")
print(f"  mu_s source {fmt(q(mu_s(post)))} eV")
print(f"  alpha/(1+q) {fmt(q(post.alpha_cloud / (1 + post.mass_ratio)))}")
pri = build_priors(RunConfig(label="_", model="environment", prior_kwargs=dict(rho_max_gcm3=2e8)), TRIG)
draws = pri.sample(200_000)          # constrained prior draws
for thr in (1e5, 1e6):
    print(f"  P(rho < {thr:.0e}) = {np.mean(post.rho_phi < thr):.4f}; "
          f"prior {np.mean(np.asarray(draws['rho_phi']) < thr):.4f}")
for name, stat in (("median", np.median), ("maximum", np.max)):
    print(f"  {name} lnL: env {stat(post.log_likelihood):.2f}, vac {stat(vp.log_likelihood):.2f}, "
          f"gain {stat(post.log_likelihood) - stat(vp.log_likelihood):+.2f}")
print(f"  Savage-Dickey (4 runs pooled) {savage_dickey_lnb(SimpleNamespace(posterior=post), pri):+.2f}")

print("\n=== C. SR posteriors, rho_max = 2e8")
for e in (5, 6, 7, 8):
    p = exact_posterior(f"gw190728_env_sr_eq9_taud1e{e}_rho2e8")
    print(f"  tau_d 1e{e}: rho {fmt(q(p.rho_phi))}  alpha {fmt(q(p.alpha_cloud))}  "
          f"Mc {fmt(q(p.chirp_mass))}  chi_eff {fmt(q(p.chi_eff))}  mu_s {fmt(q(mu_s(p)))}")
