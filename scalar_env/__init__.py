"""
scalar_env -- Independent Bayesian reanalysis of GW190728 with a
scalar-field dark-matter environmental waveform model.

Module map (sections of the paper):

    waveform.py    III        delta_psi_env(f) + custom Bilby source model
    data.py        IV.A       GWOSC strain download, PSD, conditioning
    priors.py      IV.D       vacuum + environmental priors, superradiance
    inference.py   IV.B-E     likelihood (relative binning), sampling, ln B
    analysis.py    V          corner plots, posterior comparisons, tables
    robustness.py  V.D        prior / sampler / waveform / PSD sweeps
    injections.py  IV.F, V.F  vacuum-injection false-alarm study
"""

from .waveform import (delta_psi_env, environment_transfer,
                       scalar_cloud_bbh, ionization_coupling)
from .inference import RunConfig, run_pe, ln_bayes_factor

__version__ = "0.1.0"
