"""Is the relative-binning likelihood the same function in every run?

The truncation identity (script 08) fails by ~1 at rho_max = 2e8, although
three samplers agree on both the superradiance and the agnostic evidences
and P and f are reproduced by independent methods.  The one ingredient
that differs between runs is the relative-binning fiducial, re-optimised
under each run's prior.  Evaluate the exact full-grid likelihood at
posterior samples of each run and compare with the stored relative-binning
values: a run whose samples all sit ~1 below the exact likelihood has a
biased likelihood surface, and its evidence inherits the bias.

~10 min on one core; no sampling.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env import data
from scalar_env.inference import RunConfig, build_priors
from scalar_env.waveform import scalar_cloud_bbh

bilby.core.utils.logger.setLevel("WARNING")
N = 150
RUNS = ["gw190728_env", "gw190728_env_sr_eq9_taud1e6_rho2e8",
        "gw190728_env_sr_eq9_taud1e6_rho2e8_nessai"]

ifos, trigger = data.fetch_event("GW190728_064510")


def full_likelihood(marginalise):
    """Exact full-grid likelihood.  The stored posterior log_likelihood
    may be either the marginalised value or the value at the
    reconstructed distance and phase, so both are compared."""
    return bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=bilby.gw.WaveformGenerator(
            duration=ifos[0].duration, sampling_frequency=data.SAMPLE_RATE,
            frequency_domain_source_model=scalar_cloud_bbh,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                                    reference_frequency=data.F_LOW,
                                    minimum_frequency=data.F_LOW)),
        priors=build_priors(RunConfig(label="_", model="environment",
                                      prior_kwargs=dict(rho_max_gcm3=2e8)), trigger),
        distance_marginalization=marginalise, phase_marginalization=marginalise)


exact = {"marginalised": full_likelihood(True), "at sample": full_likelihood(False)}
rng = np.random.default_rng(7)
print(f"{'run':44s} {'exact':>12} {'mean':>6} {'sd':>5} {'5%':>6} {'95%':>6}   (exact - stored lnL)")
for label in RUNS:
    post = bilby.result.read_in_result(
        f"outdir/{label}/{label}_result.json").posterior
    rows = post.iloc[rng.choice(len(post), N, replace=False)]
    for name, like in exact.items():
        delta = np.array([float(like.log_likelihood_ratio(row.to_dict()))
                          - row["log_likelihood"] for _, row in rows.iterrows()])
        print(f"{label:44s} {name:>12} {delta.mean():+6.2f} {delta.std():5.2f} "
              f"{np.quantile(delta, 0.05):+6.2f} {np.quantile(delta, 0.95):+6.2f}",
              flush=True)
