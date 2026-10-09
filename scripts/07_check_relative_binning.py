"""Relative-binning accuracy check at large environmental dephasing.

The matched-box runs sample rho_phi up to 2e8 g/cm^3, where the
environmental dephasing reaches hundreds of radians and the fiducial
optimiser locates the maximum-likelihood point (rho_phi ~ 5e7).  At the
pipeline default the relative-binning error there reaches
|delta lnL| ~ 1.5 -- comparable to the ln B being measured -- and it does
NOT improve as epsilon is reduced, because bilby places bin edges using a
fixed vacuum PN basis that cannot track an f^(-37/6) phase term.

This script tests the fix: adding that exponent to the bin-placement
basis (RunConfig.extended_binning).  The fiducial is optimised ONCE and
reused for every setting, so the comparison is like-for-like and only one
optimisation is paid for (~20 min).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env import data
from scalar_env.inference import RunConfig, build_priors, build_likelihood
from scalar_env.waveform import scalar_cloud_bbh

TOLERANCE = 0.3
GRID_ALPHA = (0.08, 0.10, 0.17)
GRID_RHO = (0.0, 1e6, 1e7, 3e7, 5e7, 7e7, 1.5e8)

# (extended_binning, epsilon) settings to compare.
SETTINGS = [(False, 0.025), (True, 0.025), (True, 0.01)]

ifos, trigger = data.fetch_event("GW190728_064510")
base_config = RunConfig(label="_rbcheck", model="environment",
                        prior_kwargs=dict(rho_max_gcm3=2e8))

# Full-grid reference likelihood.  Constructing a marginalised likelihood
# FIXES the marginalised priors in the dict it is handed, so each
# likelihood needs its own freshly built copy.
full = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=bilby.gw.WaveformGenerator(
        duration=ifos[0].duration,
        sampling_frequency=data.SAMPLE_RATE,
        frequency_domain_source_model=scalar_cloud_bbh,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                                reference_frequency=data.F_LOW,
                                minimum_frequency=data.F_LOW)),
    priors=build_priors(base_config, trigger),
    distance_marginalization=True,
    phase_marginalization=True,
)

# One optimisation, shared by every setting.
print("optimising the fiducial point once (~20 min)...")
seed_cfg = base_config.variant("seed", extended_binning=False, epsilon=0.025)
seed = build_likelihood(seed_cfg, ifos, trigger, build_priors(seed_cfg, trigger))
fiducial = dict(seed.fiducial_parameters)
print("optimised fiducial:",
      {k: round(float(fiducial[k]), 4) for k in ("chirp_mass", "mass_ratio",
                                                 "alpha_cloud", "rho_phi")})

summary = {}
for extended, eps in SETTINGS:
    cfg = base_config.variant(f"ext{extended}_eps{eps}",
                              extended_binning=extended, epsilon=eps)
    relbin = build_likelihood(cfg, ifos, trigger, build_priors(cfg, trigger),
                              fiducial=fiducial, update_fiducial=False)

    print(f"\n===== extended_binning = {extended}, epsilon = {eps} "
          f"({relbin.number_of_bins} bins) =====")
    print(f"{'alpha':>6} {'rho_phi':>9} {'lnL relbin':>11} {'lnL full':>10} {'delta':>8}")

    worst = 0.0
    for alpha in GRID_ALPHA:
        for rho in GRID_RHO:
            point = dict(fiducial, alpha_cloud=alpha, rho_phi=rho)
            a = float(relbin.log_likelihood_ratio(point))
            b = float(full.log_likelihood_ratio(point))
            worst = max(worst, abs(a - b))
            flag = "  <--" if abs(a - b) > TOLERANCE else ""
            print(f"{alpha:6.2f} {rho:9.1e} {a:11.2f} {b:10.2f} {a - b:8.2f}{flag}")
    summary[(extended, eps)] = (worst, relbin.number_of_bins)
    print(f"largest |delta lnL|: {worst:.2f}")

print("\n===== summary =====")
for (extended, eps), (worst, nbins) in summary.items():
    verdict = "OK" if worst <= TOLERANCE else "too large"
    print(f"  extended={str(extended):5s} epsilon={eps:<6} bins={nbins:>5}  "
          f"max |delta lnL| = {worst:.2f}   {verdict}")

good = [k for k, v in summary.items() if v[0] <= TOLERANCE]
if good:
    ext, eps = good[0]
    print(f"\nUse extended_binning={ext}, epsilon={eps} for the matched-box runs.")
else:
    print("\nNo setting meets the target; the matched-box runs need the full "
          "likelihood (~10-15x slower) rather than relative binning.")
