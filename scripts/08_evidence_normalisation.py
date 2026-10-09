"""Truncation-identity check on the superradiance evidences.

The superradiance (SR) prior is the agnostic prior restricted by extra
Constraint priors, so the two evidences are related exactly by

    ln B_SR - ln B_agn = ln (P / f),

with P the agnostic-posterior mass inside the SR region and f its
agnostic-prior mass -- PROVIDED the evidence is normalised to the
constrained prior.  It is: bilby seeds dynesty by rejection sampling
from the constrained prior (core/sampler/base_sampler.py,
get_initial_points_from_prior), so the prior volume starts at the
allowed region and no 1/f correction is needed.  (An earlier version of
this script applied one, on the mistaken reading that the reported
evidence integrates over the unconstrained prior.)

P and f are measured from samples, independently of the evidence
integrals they must reproduce, so this is a genuine consistency test of
both the sampler and the constraint machinery.  It is what exposed the
tau_d bug: before the fix the measured differences tracked neither
ln P nor ln(P/f), because the constraints were enforced only when the
live points were drawn.

Runs in ~1 min on one core; no sampling.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env.inference import RunConfig, build_priors, mass_cut_offset, OUTDIR

TRIGGER = 1248331528.5            # only sets the geocent_time prior
N_PRIOR = 2_000_000
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")

bilby.core.utils.logger.setLevel("WARNING")
np.random.seed(1)


def prior_mass(prior_kwargs):
    """Fraction of the product prior satisfying this prior's constraints.

    No component-mass cut: the environmental conversion function does not
    emit mass_1/mass_2, so bilby does not apply the default BBH mass
    constraints to this model, and f must be computed the same way.
    """
    pri = build_priors(RunConfig(label="_", model="environment",
                                 prior_kwargs=prior_kwargs), TRIGGER)
    s = {k: np.asarray(pri[k].sample(N_PRIOR)) for k in KEYS}
    return float(np.mean(np.asarray(pri.evaluate_constraints(s), dtype=bool)))


def posterior_mass(posterior, prior_kwargs):
    pri = build_priors(RunConfig(label="_", model="environment",
                                 prior_kwargs=prior_kwargs), TRIGGER)
    s = {k: posterior[k].to_numpy() for k in KEYS}
    return float(np.mean(np.asarray(pri.evaluate_constraints(s), dtype=bool)))


def posterior_of(*labels):
    import pandas as pd
    return pd.concat([bilby.result.read_in_result(
        str(OUTDIR / l / f"{l}_result.json")).posterior for l in labels],
        ignore_index=True)


# Exact-likelihood ln B, mass-cut offset included (script 17).
LNB = {lab: v["exact"] for lab, v in
       json.loads(Path("reweighted_lnb.json").read_text())["lnb"].items()}
AGN = {"2e8": (["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
                "gw190728_env_rho_max2e8"], dict(rho_max_gcm3=2e8)),
       "1e7": (["gw190728_env_rho_max1e7"], dict(rho_max_gcm3=1e7))}

print("Truncation identity: superradiance vs agnostic prior, same box\n")
print(f"{'box':>4} {'tau_d':>6} {'P':>7} {'f':>8} {'ln(P/f)':>9} "
      f"{'measured':>9} {'resid':>7}  inside  {'ln B(identity)':>14}")
for box, (labels, agn_kwargs) in AGN.items():
    post = posterior_of(*labels)
    ref = float(np.mean([LNB[l] for l in labels]))
    n_agn = prior_mass(agn_kwargs)
    for exp in (5, 6, 7, 8):
        lab = f"gw190728_env_sr_eq9_taud1e{exp}" + ("_rho2e8" if box == "2e8" else "")
        if lab not in LNB:
            print(f"{box:>4} {'1e'+str(exp):>6}   (rerun pending)")
            continue
        kw = dict(agn_kwargs, superradiance_tau_d_yr=10.0**exp)
        p = posterior_mass(post, kw)
        f = prior_mass(kw) / n_agn
        pred, meas = np.log(max(p, 1e-12) / f), LNB[lab] - ref
        # A correct run has every posterior sample inside the constraints.
        sr_post = posterior_of(lab)
        inside = posterior_mass(sr_post, kw)
        print(f"{box:>4} {'1e'+str(exp):>6} {p:7.3f} {f:8.4f} {pred:+9.2f} "
              f"{meas:+9.2f} {meas - pred:+7.2f}  {inside:6.1%}  "
              f"{ref + pred:+14.2f}")

print("\nP, f from samples; measured from the exact-likelihood evidences"
      "\n(script 17).  'inside' is the fraction of the SR run's own posterior"
      "\nsatisfying its constraints -- anything below 100% means the"
      "\nconstraints were not enforced.  ln B(identity) = agnostic + ln(P/f);"
      f"\nboth include the mass-cut offset ({mass_cut_offset():+.3f}).")
