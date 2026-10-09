"""Does nested sampling get the superradiance evidence right in this geometry?

Four nested-sampling estimates of the tau_d = 1e6 yr superradiance evidence
fall short of the truncation identity by ~0.85.  Test the sampler where the
answer is known exactly: a synthetic likelihood whose posterior under the
agnostic prior is a kernel density estimate K of our real agnostic
posterior in (M_c, q, alpha, rho_phi), times 11 narrow Gaussian dimensions
that bring the problem to our 15, sampled under our real agnostic and
superradiance priors (same constraint machinery).  With L = K / pi_box,

    ln Z_agn = ln K(C) - ln pi(C),    ln Z_SR = ln K(C & W) - ln pi(C & W),

where K(.) and pi(.) are the K- and prior-mass of the agnostic constraints
C and the window W, computed by Monte Carlo.  If dynesty, run as in
production, recovers both, the real residual does not come from the
sampler meeting this constraint geometry.  (Two other routes failed:
importance sampling with a kernel-density proposal -- effective sample size
< 10 in 15 dimensions -- and nessai's importance nested sampler, which
bilby 2.8.1 does not support and which mis-normalised a constrained toy.)
~1 h.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scipy.stats import gaussian_kde
from scalar_env.inference import RunConfig, build_priors, OUTDIR

bilby.core.utils.logger.setLevel("WARNING")
NPOOL = max(1, (os.cpu_count() or 2) - 2)
KEYS = ["chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi"]
DUMMY = [f"u{i}" for i in range(11)]
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
        "gw190728_env_rho_max2e8"]
PRIORS = {"agnostic": dict(rho_max_gcm3=2e8),
          "superradiance": dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6)}

rng = np.random.default_rng(18)
post = pd.concat([bilby.result.read_in_result(str(OUTDIR / l / f"{l}_result.json")).posterior
                  for l in BASE], ignore_index=True)
KDE = gaussian_kde(post[KEYS].to_numpy()[rng.choice(len(post), 3000, replace=False)].T)


def make_priors(kw):
    env = build_priors(RunConfig(label="_", model="environment", prior_kwargs=kw), 0.0)
    pri = bilby.core.prior.PriorDict(conversion_function=env.conversion_function)
    for k, p in env.items():
        if k in KEYS or isinstance(p, bilby.core.prior.Constraint):
            pri[k] = p
    for k in DUMMY:
        pri[k] = bilby.core.prior.Uniform(0, 1, k)
    return pri


BOX = make_priors(PRIORS["agnostic"])     # the four marginal densities pi_box


class Synthetic(bilby.Likelihood):
    def __init__(self):
        super().__init__(parameters={k: None for k in KEYS + DUMMY})

    def log_likelihood(self):
        x = np.array([self.parameters[k] for k in KEYS])
        lnl = KDE.logpdf(x)[0] - sum(BOX[k].ln_prob(self.parameters[k]) for k in KEYS)
        u = np.array([self.parameters[k] for k in DUMMY])
        return float(lnl - 0.5 * np.sum(((u - 0.5) / 0.05) ** 2)
                     - len(DUMMY) * np.log(np.sqrt(2 * np.pi) * 0.05))


def exact_ln_z(pri, n=2_000_000):
    """ln K(constraints) - ln pi(constraints): both masses by Monte Carlo."""
    inside = lambda s: np.asarray(pri.evaluate_constraints(s), bool)
    k = KDE.resample(n, seed=rng)
    ks = {key: k[i] for i, key in enumerate(KEYS)}
    in_box = np.all([(k[i] >= BOX[key].minimum) & (k[i] <= BOX[key].maximum)
                     for i, key in enumerate(KEYS)], axis=0)
    k_mass = np.mean(in_box & inside(ks))
    p_mass = np.mean(inside({key: np.asarray(BOX[key].sample(n)) for key in KEYS}))
    return np.log(k_mass) - np.log(p_mass)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    out = {}
    for name, kw in PRIORS.items():
        pri = make_priors(kw)
        exact = exact_ln_z(pri)
        res = bilby.run_sampler(likelihood=Synthetic(), priors=pri, sampler="dynesty",
                                nlive=1000, npool=NPOOL, sample="rwalk", dlogz=0.1,
                                outdir=f"outdir/synthetic_{name}", label=f"synthetic_{name}",
                                plot=False, resume=True)
        out[name] = (res.log_evidence, res.log_evidence_err, exact)
        print(f"{name:14s} dynesty ln Z {res.log_evidence:+.3f} +- {res.log_evidence_err:.3f}"
              f"   exact {exact:+.3f}   error {res.log_evidence - exact:+.2f}", flush=True)
    (za, ea, xa), (zs, es, xs) = out["agnostic"], out["superradiance"]
    print(f"rise: dynesty {zs - za:+.2f}, exact {xs - xa:+.2f}; residual "
          f"{(zs - za) - (xs - xa):+.2f} +- {np.hypot(ea, es):.2f}")
