"""How good is analytic phase marginalisation for IMRPhenomXPHM?

bilby's phase-marginalised likelihood treats the reference phase as an
overall rotation of the strain, which is exact for the (2,2) mode alone.
Compare it, at posterior samples of the phase-marginalised runs, with the
marginalisation done numerically: the likelihood with the phase as a
parameter, averaged over K equally spaced phases (the trapezoidal rule,
which converges exponentially for this periodic integrand once the
spacing is below the ~0.04 rad width of its peaks).  Since

    Z_numerical / Z_analytic = E_post[ L_numerical / L_analytic ],

the difference between the two models' log-ratios is the error that the
analytic marginalisation makes in ln B.  Unlike a comparison of separate
phase-sampled runs (script 19), it carries no sampling noise from the
evidences; the printed effective sample size says how far to trust each
ratio.  Baseline and catalogue noise models.  ~1 h on 10 cores.
"""

import multiprocessing as mp
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env.inference import RunConfig, exact_likelihood, OUTDIR

bilby.core.utils.logger.setLevel("WARNING")
N_SAMPLES = 200
K = 128
PHASES = 2 * np.pi * np.arange(K) / K
NPOOL = int(os.environ.get("NPOOL", max(1, (os.cpu_count() or 2) - 2)))
HLV = ("H1", "L1", "V1")
NETWORKS = {  # name: (RunConfig changes, vacuum, agnostic and superradiance runs)
    "H1+L1, Welch PSDs": (
        {}, ["gw190728_vacuum", "gw190728_env_seed1", "gw190728_env_sr_eq9_taud1e6_rho2e8"]),
    "H1+L1+V1, GWTC-2.1 PSDs": (
        dict(detectors=HLV, psd_source="gwtc21"),
        ["gw190728_vacuum_hlv_gwtc21psd", "gw190728_env_hlv_gwtc21psd",
         "gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8"]),
}


def likelihoods(model, changes):
    """(analytic, phase-as-parameter) exact likelihoods; private distance
    tables, as the injection campaign reads bilby's default one."""
    config = RunConfig(label="_", model=model, **changes,
                       prior_kwargs=dict(rho_max_gcm3=2e8) if model == "environment" else {})
    return tuple(exact_likelihood(replace(config, phase_marginalization=m),
                                  f"data_cache/distance_lookup_{'phase' if m else 'nophase'}.npz")
                 for m in (True, False))


LIKELIHOODS = {(network, model): likelihoods(model, changes)
               for network, (changes, _) in NETWORKS.items()
               for model in ("vacuum", "environment")}


def log_ratio(task):
    """ln(L_numerical / L_analytic) at one posterior sample."""
    key, params = task
    analytic, with_phase = LIKELIHOODS[key]
    numerical = logsumexp([with_phase.log_likelihood_ratio({**params, "phase": p})
                           for p in PHASES]) - np.log(K)
    return numerical - analytic.log_likelihood_ratio({**params, "phase": 0.0})


def ln_z_ratio(network, label, pool, rng):
    """(ln Z_numerical / Z_analytic, bootstrap error, effective sample size)."""
    post = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json")).posterior
    rows = post.iloc[rng.choice(len(post), N_SAMPLES, replace=False)]
    model = "environment" if "alpha_cloud" in post else "vacuum"
    r = np.array(pool.map(log_ratio, [((network, model), row.to_dict())
                                      for _, row in rows.iterrows()]))
    boot = [logsumexp(rng.choice(r, r.size)) - np.log(r.size) for _ in range(500)]
    w = np.exp(r - r.max())
    return logsumexp(r) - np.log(r.size), np.std(boot), w.sum() ** 2 / (w ** 2).sum()


if __name__ == "__main__":
    rng = np.random.default_rng(26)
    with mp.get_context("fork").Pool(NPOOL) as pool:
        for network, (_, labels) in NETWORKS.items():
            (v, dv, ev), *envs = [ln_z_ratio(network, l, pool, rng) for l in labels]
            print(f"\n{network}: ln Z_numerical / Z_analytic, vacuum {v:+.2f} +- {dv:.2f} (ESS {ev:.0f})")
            for name, (e, de, ee) in zip(("agnostic", "superradiance"), envs):
                print(f"  {name:14s} {e:+.2f} +- {de:.2f} (ESS {ee:.0f})  ->  ln B error of "
                      f"the analytic marginalisation {v - e:+.2f} +- {np.hypot(dv, de):.2f}",
                      flush=True)
