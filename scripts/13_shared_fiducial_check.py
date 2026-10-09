"""Does the relative-binning fiducial break the truncation identity?

Every production run re-optimises its relative-binning fiducial under its
own prior, so agnostic and superradiance runs evaluate slightly different
likelihood approximations.  At rho_max = 2e8 the superradiance evidence
is sampler-independent (script 10) and P, f are method-independent, yet
ln Z_SR - ln Z_agn misses ln(P/f) by ~1.  Two tests:

  A. Likelihood functions, no sampling (~20 min).  At agnostic posterior
     samples inside and outside the tau_d = 1e6 window, compare the exact
     full-grid likelihood with relative binning built on (i) one fixed
     fiducial, (ii) a fiducial optimised under the agnostic prior and
     (iii) one optimised under the superradiance prior, as in production.
  B. The identity itself (~3.5 h).  Agnostic and superradiance runs that
     share the fixed fiducial, so their likelihoods are the same function.
     If ln Z_SR - ln Z_agn now equals ln(P/f), the fiducial was the cause.

The fixed fiducial is the maximum-likelihood agnostic sample inside the
window, so it lies within both priors and near both posteriors.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env import data
from scalar_env.inference import (RunConfig, build_priors, build_likelihood,
                                  run_pe, ln_bayes_factor, OUTDIR)
from scalar_env.waveform import scalar_cloud_bbh

bilby.core.utils.logger.setLevel("WARNING")
NPOOL = max(1, (os.cpu_count() or 2) - 2)
AGN = dict(rho_max_gcm3=2e8)
SR = dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6)
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
FID_KEYS = ("chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2",
            "phi_12", "phi_jl", "luminosity_distance", "theta_jn", "psi",
            "phase", "ra", "dec", "geocent_time", "alpha_cloud", "rho_phi")

ifos, trigger = data.fetch_event("GW190728_064510")
priors = lambda kw: build_priors(RunConfig(label="_", model="environment",
                                           prior_kwargs=kw), trigger)
inside = lambda frame: np.asarray(priors(SR).evaluate_constraints(
    {k: frame[k].to_numpy() for k in KEYS}), dtype=bool)

agn_post = pd.concat([bilby.result.read_in_result(
    str(OUTDIR / l / f"{l}_result.json")).posterior for l in
    ("gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
     "gw190728_env_rho_max2e8")], ignore_index=True)
ins = inside(agn_post)
best = agn_post[ins].loc[agn_post[ins].log_likelihood.idxmax()]
FIDUCIAL = {k: float(best[k]) for k in FID_KEYS}
print("fixed fiducial:", {k: round(FIDUCIAL[k], 4) for k in KEYS}, flush=True)

# ------------------------------------------------------------ Test A
exact = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=bilby.gw.WaveformGenerator(
        duration=ifos[0].duration, sampling_frequency=data.SAMPLE_RATE,
        frequency_domain_source_model=scalar_cloud_bbh,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                                reference_frequency=data.F_LOW,
                                minimum_frequency=data.F_LOW)),
    priors=priors(AGN), distance_marginalization=True, phase_marginalization=True)

cfg = lambda kw: RunConfig(label="_", model="environment", prior_kwargs=kw)
relbin = {
    "fixed fiducial": build_likelihood(cfg(AGN), ifos, trigger, priors(AGN),
                                       fiducial=FIDUCIAL, update_fiducial=False),
    "optimised, agnostic prior": build_likelihood(cfg(AGN), ifos, trigger, priors(AGN)),
    "optimised, SR prior": build_likelihood(cfg(SR), ifos, trigger, priors(SR)),
}
rng = np.random.default_rng(11)
points = {"inside window": agn_post[ins].iloc[rng.choice(ins.sum(), 50, replace=False)],
          "outside window": agn_post[~ins].iloc[rng.choice((~ins).sum(), 30, replace=False)]}
exact_lnl = {w: np.array([float(exact.log_likelihood_ratio(r.to_dict()))
                          for _, r in p.iterrows()]) for w, p in points.items()}
test_a = {}
print(f"\nTest A: relative binning minus exact (marginalised) ln L")
for name, like in relbin.items():
    fid = {k: round(float(like.fiducial_parameters[k]), 4) for k in ("alpha_cloud", "rho_phi")}
    for where, p in points.items():
        d = np.array([float(like.log_likelihood_ratio(r.to_dict()))
                      for _, r in p.iterrows()]) - exact_lnl[where]
        test_a[f"{name} / {where}"] = (float(d.mean()), float(d.std()))
        print(f"  {name:26s} {where:15s} mean {d.mean():+6.3f}  sd {d.std():5.3f}"
              f"  max|d| {abs(d).max():5.2f}   fiducial {fid}", flush=True)

# ------------------------------------------------------------ Test B
env = RunConfig(label="gw190728_env_fixfid", model="environment", npool=NPOOL,
                prior_kwargs=AGN, fiducial=FIDUCIAL)
vac = run_pe(RunConfig(label="gw190728_vacuum", model="vacuum"))
r_agn = run_pe(env)
r_sr = run_pe(env.variant("sr_eq9_taud1e6_rho2e8", prior_kwargs=SR))

lnb_agn = ln_bayes_factor(r_agn, vac)[0]
lnb_sr = ln_bayes_factor(r_sr, vac)[0]
P = float(inside(r_agn.posterior).mean())
draws = {k: np.asarray(priors(AGN)[k].sample(1_000_000)) for k in KEYS}
f = float(np.mean(np.asarray(priors(SR).evaluate_constraints(draws), bool))
          / np.mean(np.asarray(priors(AGN).evaluate_constraints(draws), bool)))
predicted = np.log(P / f)
print(f"\nTest B (shared fiducial): agnostic {lnb_agn:+.2f}, SR {lnb_sr:+.2f}; "
      f"measured rise {lnb_sr - lnb_agn:+.2f}, ln(P/f) = {predicted:+.2f} "
      f"(P = {P:.3f}, f = {f:.4f}); residual {lnb_sr - lnb_agn - predicted:+.2f}")
print("   production runs: agnostic -1.29 (4-run mean), SR -0.03; "
      "residual -0.93")
Path("shared_fiducial.json").write_text(json.dumps(dict(
    fiducial=FIDUCIAL, test_a=test_a, lnb_agnostic=lnb_agn, lnb_sr=lnb_sr,
    P=P, f=f, predicted_rise=float(predicted)), indent=2))
