"""
Section IV.B-E -- Bayesian inference pipeline.

One entry point, run_pe(config), covers every analysis in the paper: the
RunConfig dataclass captures all choices (event, model, priors, sampler,
approximant, PSD settings), so vacuum runs, environmental runs and every
robustness variation differ only in their config.

Likelihood: relative-binning GravitationalWaveTransient (Zackay et al.
2018) with distance + phase marginalisation -- the speedup that makes
laptop-scale nested sampling possible.  Evidence: ln Z from dynesty
(default), pymultinest or nessai for the sampler-comparison test.
"""

import functools
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import numpy as np

from . import data, priors as prior_module
from .waveform import scalar_cloud_bbh

OUTDIR = Path("outdir")

# GWTC-3 maximum-likelihood-ish values for GW190728, used as the relative
# binning fiducial point.  Refined automatically by bilby if enabled.
FIDUCIAL_GW190728 = dict(
    chirp_mass=8.6, mass_ratio=0.66, a_1=0.3, a_2=0.1, tilt_1=0.5,
    tilt_2=0.8, phi_12=1.0, phi_jl=1.0, luminosity_distance=870.0,
    theta_jn=1.0, psi=1.5, phase=1.0, dec=0.3, ra=5.0)


@dataclass
class RunConfig:
    """Everything that defines a single PE run."""
    label: str
    event: str = "GW190728_064510"
    model: str = "vacuum"                    # "vacuum" | "environment"
    sampler: str = "dynesty"
    nlive: int = 1000
    npool: int = 1                           # parallel likelihood-eval workers
    epsilon: float = 0.025                   # relative-binning bin resolution
    extended_binning: bool = False           # add f^(-37/6) to the bin basis
    approximant: str = "IMRPhenomXPHM"
    psd_fftlength: int = 4                   # Welch segment length [s]
    psd_source: str = "welch"                # "welch" | "gwtc21" (catalogue)
    roll_off: float = data.ROLL_OFF          # Tukey rise time before the FFT [s]
    phase_marginalization: bool = True       # False: sample the phase
    detectors: tuple = ("H1", "L1")
    seed: int = 42
    prior_kwargs: dict = field(default_factory=dict)   # -> environmental_priors
    sampler_kwargs: dict = field(default_factory=dict)  # override sample/dlogz etc.
    fiducial: dict = field(default_factory=dict)  # fixed relative-binning reference

    def variant(self, label_suffix, **changes):
        return replace(self, label=f"{self.label}_{label_suffix}", **changes)


def build_waveform_generator(config, duration):
    """RelativeBinningGravitationalWaveTransient requires a source model that
    honours the 'fiducial' / 'frequency_bin_edges' waveform_arguments it
    injects (see bilby.gw.source.lal_binary_black_hole_relative_binning) --
    the plain lal_binary_black_hole silently ignores them and returns a
    full-resolution waveform where a coarse one was requested.
    """
    import bilby

    if config.model == "environment":
        source_model = scalar_cloud_bbh
    else:
        source_model = bilby.gw.source.lal_binary_black_hole_relative_binning

    return bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=data.SAMPLE_RATE,
        frequency_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=config.approximant,
            reference_frequency=data.F_LOW,
            minimum_frequency=data.F_LOW),
    )


def build_priors(config, trigger_time):
    if config.model == "environment":
        return prior_module.environmental_priors(trigger_time,
                                                 **config.prior_kwargs)
    return prior_module.vacuum_priors(trigger_time, **config.prior_kwargs)


def _environmental_relative_binning_class():
    """Relative binning whose bin placement also resolves delta_psi_env.

    Kept but OFF by default (script 07 measures no benefit): retained
    because it is the natural first response if a future configuration
    does strain the binning, and because the negative result is worth
    recording.  bilby places bin edges so that a fixed vacuum PN basis
    gamma = [-5/3, -2/3, 1, 5/3, 7/3] changes by `epsilon` per bin
    (relative.py, setup_bins), and normalises each exponent to the same
    total variation -- so adding f^(-37/6) buys only ~6 bins, because the
    ones it asks for near f_low would be narrower than the data's
    frequency spacing and are dropped (relative.py, "bins are at least as
    wide as this spacing").  The large errors seen before (max |dlnL| =
    1.8 at the matched box) were never a binning failure: they came from
    linearising Ldot_phi/Ldot_GW in the waveform.  With the resummed
    waveform the stock binning gives max |dlnL| = 0.13.
    """
    import bilby

    class EnvironmentalRelativeBinning(
            bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient):

        def setup_bins(self):
            self.gamma = np.array(sorted(set(
                self.gamma.tolist() + [-37.0 / 6.0])))
            super().setup_bins()

    return EnvironmentalRelativeBinning


def build_likelihood(config, ifos, trigger_time, priors,
                     fiducial=FIDUCIAL_GW190728, update_fiducial=True):
    """Relative-binning likelihood (Section IV.B)."""
    import bilby

    wfg = build_waveform_generator(config, ifos[0].duration)
    # A supplied fiducial keeps its own merger time; only the default,
    # which has none, is placed at the trigger.
    fiducial = dict(fiducial)
    fiducial.setdefault("geocent_time", trigger_time)
    if config.model == "environment":
        # Default fiducial starts on the vacuum submanifold (rho_phi = 0 is
        # exact vacuum); setdefault so a caller-supplied point survives.
        # update_fiducial then optimises every sampled parameter within the
        # prior bounds, ignoring Constraint priors: in practice it lands
        # near alpha ~ 0.06, rho_phi ~ 1e8, outside any superradiance window.
        fiducial.setdefault("alpha_cloud", 0.1)
        fiducial.setdefault("rho_phi", 0.0)

    if config.model == "environment" and config.extended_binning:
        likelihood_class = _environmental_relative_binning_class()
    else:
        likelihood_class = \
            bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient

    return likelihood_class(
        interferometers=ifos,
        waveform_generator=wfg,
        priors=priors,
        fiducial_parameters=fiducial,
        update_fiducial_parameters=update_fiducial,
        distance_marginalization=True,
        phase_marginalization=config.phase_marginalization,
        epsilon=config.epsilon,
    )


def exact_likelihood(config, distance_lookup=None, ifos=None, trigger_time=None):
    """The full-grid likelihood that relative binning approximates, with
    the run's own data, approximant and marginalisations (or the given
    ifos, e.g. with a modified PSD).

    distance_lookup: file for bilby's distance-marginalisation table; give
    a private one when other processes may be reading the default file.
    """
    import bilby

    if ifos is None:
        ifos, trigger_time = data.fetch_event(
            config.event, detectors=config.detectors,
            psd_fftlength=config.psd_fftlength, psd_source=config.psd_source,
            roll_off=config.roll_off)
    source = (scalar_cloud_bbh if config.model == "environment"
              else bilby.gw.source.lal_binary_black_hole)
    return bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=bilby.gw.WaveformGenerator(
            duration=ifos[0].duration, sampling_frequency=data.SAMPLE_RATE,
            frequency_domain_source_model=source,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(waveform_approximant=config.approximant,
                                    reference_frequency=data.F_LOW,
                                    minimum_frequency=data.F_LOW)),
        priors=build_priors(config, trigger_time),
        distance_marginalization=True,
        phase_marginalization=config.phase_marginalization,
        distance_marginalization_lookup_table=distance_lookup)


def _ensure_fork_start_method():
    """Python >= 3.14 defaults to 'forkserver' on Linux, which (a)
    re-imports the calling script in every pool worker, breaking
    module-level run scripts, and (b) requires pickling pool state,
    which bilby's nessai model (a class defined inside a function)
    cannot survive.  Restore 'fork', which bilby/dynesty/nessai pools
    are built around; safe alongside npool because BLAS threading is
    capped to one thread per worker (see README).  Called for every
    run: samplers like nessai create internal pools even at npool = 1.
    """
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != "fork":
        try:
            mp.set_start_method("fork", force=True)
        except (RuntimeError, ValueError):   # fork unavailable (Windows)
            pass


def run_pe(config, ifos=None, trigger_time=None, injection=None):
    """Run one PE analysis end-to-end; returns the bilby Result.

    Completed runs are resumed/reloaded from OUTDIR, so sweeps are
    restartable.  Pass ifos/trigger_time explicitly for injections.
    """
    import bilby

    _ensure_fork_start_method()

    result_file = OUTDIR / config.label / f"{config.label}_result.json"
    if result_file.exists():
        return bilby.result.read_in_result(str(result_file))

    if ifos is None:
        ifos, trigger_time = data.fetch_event(
            config.event, detectors=config.detectors,
            psd_fftlength=config.psd_fftlength, psd_source=config.psd_source,
            roll_off=config.roll_off)

    priors = build_priors(config, trigger_time)
    # By default each run re-optimises its relative-binning reference
    # under its own prior; a fixed one makes runs share one likelihood.
    fixed = (dict(fiducial=config.fiducial, update_fiducial=False)
             if config.fiducial else {})
    likelihood = build_likelihood(config, ifos, trigger_time, priors, **fixed)

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=config.sampler,
        nlive=config.nlive,
        npool=config.npool,
        **{"sample": "rwalk", "dlogz": 0.1, **config.sampler_kwargs},
        seed=config.seed,
        outdir=str(OUTDIR / config.label),
        label=config.label,
        injection_parameters=injection,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        resume=True,
    )
    return result


# ----------------------------------------------------------------- V.E
def ln_bayes_factor(result_env, result_vac):
    """(ln B^env_vac, sigma) from nested-sampling evidences.

    The raw difference of the two runs, as stored in every results file;
    add mass_cut_offset() before quoting it.
    """
    lnb = result_env.log_evidence - result_vac.log_evidence
    err = np.hypot(result_env.log_evidence_err, result_vac.log_evidence_err)
    return lnb, err


@functools.lru_cache
def mass_cut_offset(n=2_000_000):
    """-ln F = +0.180, to add to every raw ln B from ln_bayes_factor.

    vacuum_priors inherits bilby's default Constraint(5, 100) on mass_1
    and mass_2, which the vacuum conversion function evaluates; the
    environmental one (priors._env_conversion) never emits the component
    masses, so that prior is uncut.  Every sampler normalises the evidence
    to its own constrained prior (dead points confirm it: none below
    m2 = 5 in vacuum runs, 16.5% of the first thousand in environmental
    ones), so the vacuum evidence is divided by F = P(m1, m2 in [5, 100])
    = 0.835 and the environmental one is not.  No posterior has support
    below m2 = 5 Msun, so multiplying it back compares the two models
    under the same, uncut mass prior.  The Savage-Dickey estimate, which
    involves no vacuum run, agrees with the corrected value.
    """
    priors = prior_module.vacuum_priors(trigger_time=0.0)
    unit = np.random.default_rng(0).uniform(size=(2, n))   # fixed draws: the same offset every run
    draws = {key: priors[key].rescale(u) for key, u in zip(("chirp_mass", "mass_ratio"), unit)}
    kept = np.asarray(priors.evaluate_constraints(draws), dtype=bool)
    return -float(np.log(kept.mean()))


def savage_dickey_lnb(result_env, priors, n_prior=400_000):
    """Savage-Dickey cross-check of ln B^env_vac (Section IV.E).

    The Roy et al. parametrization nests vacuum exactly at rho_phi = 0,
    on the uniform prior's boundary, so B is the ratio of prior to
    posterior density of rho_phi there.  Both must refer to the
    *constrained* prior, which the sampler normalises: its density at
    rho_phi -> 0 is 1/rho_max times (fraction of draws satisfying the
    constraints there) / (fraction satisfying them overall).  Taking
    1/rho_max alone, as an earlier version did, ignores the cloud-mass
    cut.  The posterior density uses a KDE reflected about rho_phi = 0.
    A consistency check on the nested-sampling ln B, not a replacement.
    """
    from scipy.stats import gaussian_kde

    keys = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
    draws = {k: np.asarray(priors[k].sample(n_prior)) for k in keys}
    f_all = np.mean(np.asarray(priors.evaluate_constraints(draws), dtype=bool))
    # Constraint priors are open intervals, so probe just above zero.
    draws["rho_phi"] = np.full(n_prior, 1e-9 * priors["rho_phi"].maximum)
    f_zero = np.mean(np.asarray(priors.evaluate_constraints(draws), dtype=bool))
    prior_at_zero = f_zero / (f_all * priors["rho_phi"].maximum)

    samples = result_env.posterior["rho_phi"].to_numpy()
    kde = gaussian_kde(np.concatenate([samples, -samples]))
    posterior_at_zero = 2.0 * kde(0.0)[0]
    return float(np.log(prior_at_zero / posterior_at_zero))


def evidence_scatter(configs, ifos=None, trigger_time=None):
    """Repeat a run with different seeds; report ln Z mean and std.

    Convergence requirement (Section IV.C): std(ln Z) < 0.5 across seeds.
    """
    lnzs = [run_pe(c, ifos, trigger_time).log_evidence for c in configs]
    return float(np.mean(lnzs)), float(np.std(lnzs))


def seed_variants(config, n_seeds=3):
    return [config.variant(f"seed{s}", seed=s) for s in range(n_seeds)]
