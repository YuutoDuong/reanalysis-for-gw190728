"""
Section IV.D -- Prior choices.

Vacuum priors follow LVK conventions (uniform in *component masses*, which
matters: a flat mass-ratio prior visibly distorts q against the GWTC
releases).

Environmental priors follow Roy et al. (2026): broad uniform prior on the
mean scalar density rho_phi [g/cm^3] and on alpha, subject to
  (i)  alpha <= 0.5 (1 + q)          [validity of the coupling expansion]
  (ii) rho_phi r_phi^3 < 0.1 M       [cloud lighter than 10% of the binary]
tau_d is NOT a parameter: in the superradiance variant it is a fixed
analysis setting (Roy et al. use 1e5..1e8 yr), bound into the conversion
function that evaluates the constraints (see environmental_priors).
"""

import functools

import numpy as np

from .waveform import T_SUN, YEAR, RHO_GEO_PER_GCM3


def vacuum_priors(trigger_time, chirp_mass_bounds=(6.0, 14.0)):
    """Standard 15-parameter BBH prior dict (LVK conventions)."""
    import bilby

    priors = bilby.gw.prior.BBHPriorDict()
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
        *chirp_mass_bounds, name="chirp_mass", unit="$M_\\odot$")
    priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(
        0.125, 1.0, name="mass_ratio")
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        1e2, 3e3, name="luminosity_distance", unit="Mpc")
    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger_time - 0.1, trigger_time + 0.1, name="geocent_time")
    return priors


def environmental_priors(trigger_time, chirp_mass_bounds=(6.0, 14.0),
                         alpha_prior="uniform", alpha_bounds=(0.01, 0.75),
                         rho_max_gcm3=1e7, superradiance_tau_d_yr=None):
    """Vacuum priors + Roy et al. cloud priors on (rho_phi, alpha).

    Every keyword is a robustness-study axis (Section V.D).  rho_phi = 0
    is inside the prior support, so the model nests vacuum exactly and
    the Savage-Dickey ratio is well defined.
    """
    import bilby

    # tau_d is bound into the conversion function rather than carried as a
    # DeltaFunction prior.  bilby evaluates Constraint priors on the
    # *sampled* parameters only (its dynesty likelihood wrapper builds the
    # constraint dict from the search keys), so a fixed prior is invisible
    # there: the conversion never emitted the sr_* keys, and bilby silently
    # skips a Constraint whose key is absent.  Every superradiance run made
    # with the DeltaFunction version was therefore an agnostic run -- only
    # 9-67% of their posterior samples lie inside the SR region, matching
    # the agnostic posterior's own fraction.
    conversion = _env_conversion
    if superradiance_tau_d_yr is not None:
        conversion = functools.partial(_env_conversion,
                                       tau_d_yr=superradiance_tau_d_yr)
    priors = bilby.gw.prior.BBHPriorDict(conversion_function=conversion)
    priors.update(vacuum_priors(trigger_time, chirp_mass_bounds))

    alpha_cls = {"uniform": bilby.core.prior.Uniform,
                 "log-uniform": bilby.core.prior.LogUniform}[alpha_prior]
    priors["alpha_cloud"] = alpha_cls(*alpha_bounds, name="alpha_cloud",
                                      latex_label=r"$\alpha$")
    priors["rho_phi"] = bilby.core.prior.Uniform(
        0.0, rho_max_gcm3, name="rho_phi",
        latex_label=r"$\bar\rho_\phi$ [g/cm$^3$]")

    priors["alpha_bound_ratio"] = bilby.core.prior.Constraint(
        minimum=0.0, maximum=1.0, name="alpha_bound_ratio")
    priors["cloud_mass_ratio"] = bilby.core.prior.Constraint(
        minimum=0.0, maximum=1.0, name="cloud_mass_ratio")

    if superradiance_tau_d_yr is not None:
        # Roy et al. SM Eqs. (9)-(10): tau_SR <= tau_d <= tau_ann as a
        # *spin-independent* window on alpha/(1+q) (the natal spin drives
        # the instability; the sampled merger spin is post-spin-down and
        # must NOT enter), plus the saturation-mass density cap.
        priors["sr_window_lo"] = bilby.core.prior.Constraint(
            minimum=1.0, maximum=np.inf, name="sr_window_lo")
        priors["sr_window_hi"] = bilby.core.prior.Constraint(
            minimum=0.0, maximum=1.0, name="sr_window_hi")
        priors["sr_density_ratio"] = bilby.core.prior.Constraint(
            minimum=0.0, maximum=1.0, name="sr_density_ratio")
    return priors


def superradiance_growth_time_yr(mass_1, chi_1, alpha):
    """e-folding time of the |211> superradiant instability [yr].

    Detweiler's small-alpha approximation in the form given by Baumann,
    Chia & Porto (2019), their Eqs. (2.12)-(2.13):

        Gamma M = 2 rt_+ C_211 (m Omega_H M - alpha) alpha^(4l+5),
        C_211   = (1/48) [ (1 - chi^2) + (chi - 2 rt_+ alpha)^2 ],

    with rt_+ = r_+/M = 1 + sqrt(1 - chi^2) and
    Omega_H M = chi / (2 rt_+).  Note alpha^9 for l = 1 (an earlier
    version of this function used alpha^8, which overestimated the rate
    by ~1/(rt_+ alpha) and hence made the tau_SR <= tau_d constraint too
    permissive).  Returns inf where the mode is not superradiant.

    Used only for the alternative, merger-spin-conditioned prior; the
    baseline superradiance analysis uses Roy et al.'s published window
    (Eqs. 9-10), which is spin-independent.
    """
    mass_1, chi_1, alpha = np.broadcast_arrays(mass_1, chi_1, alpha)
    chi_1 = np.clip(chi_1, 0.0, 0.999)
    rt_plus = 1.0 + np.sqrt(1.0 - chi_1**2)               # r_+ / M
    omega_h = chi_1 / (2.0 * rt_plus)                     # Omega_H M
    c_211 = ((1.0 - chi_1**2) + (chi_1 - 2.0 * rt_plus * alpha) ** 2) / 48.0
    rate = 2.0 * rt_plus * c_211 * (omega_h - alpha) * alpha**9
    tau_s = np.where(rate > 0, mass_1 * T_SUN / np.where(rate > 0, rate, 1.0),
                     np.inf)
    return tau_s / YEAR


def _env_conversion(parameters, tau_d_yr=None):
    """Bilby conversion hook adding the constrained derived quantities.

    Note: this does not emit mass_1/mass_2, so the default BBH prior's
    mass Constraint(5, 100) is not applied to the environmental model
    (bilby skips absent keys), while the vacuum model applies it.  The
    likelihood vanishes below 5 Msun for GW190728-like signals, so the
    posteriors are unaffected, but each evidence is normalised to its own
    constrained prior: every raw ln B is low by -ln F = 0.180 (see
    inference.mass_cut_offset).  It is also why scripts/08 computes prior
    fractions without a mass cut, matching what the sampler enforces.
    """
    out = parameters.copy()
    q = out["mass_ratio"]
    m_tot = out["chirp_mass"] * (1.0 + q) ** 1.2 / q**0.6   # detector frame
    mass_1 = m_tot / (1.0 + q)
    alpha = out["alpha_cloud"]

    # (i) validity of the multipole expansion: alpha <= 0.5 (1 + q)
    out["alpha_bound_ratio"] = alpha / (0.5 * (1.0 + q))

    # (ii) cloud mass rho_phi r_phi^3 < 0.1 M, with r_phi = M / alpha^2
    rho_geo = out["rho_phi"] * RHO_GEO_PER_GCM3
    m_tot_s = m_tot * T_SUN
    out["cloud_mass_ratio"] = 10.0 * rho_geo * m_tot_s**2 / alpha**6

    if tau_d_yr is not None:
        # Roy et al. SM Eq. (9): the tau_SR <= tau_d <= tau_ann window,
        #   0.02 s^(1/9) <= alpha/(1+q) <= 0.11 s^(1/15),
        # with s = (m1 / 10 Msun) (1 Myr / tau_d).  Detector-frame m1 is
        # used (source/detector distinction shifts the edges by < 2% at
        # GW190728's redshift through the 1/9 and 1/15 powers).
        x = alpha / (1.0 + q)
        s = (mass_1 / 10.0) * (1e6 / tau_d_yr)
        out["sr_window_lo"] = x / (0.02 * s ** (1.0 / 9.0))
        out["sr_window_hi"] = x / (0.11 * s ** (1.0 / 15.0))
        # Roy et al. SM Eq. (10): saturation-mass bound on the density,
        #   rho <= [1.3e8 g/cm^3 / (8 (1+q)^-3)] (x/0.07)^7 (10 Msun/m1)^2
        #        =  1.3e8 g/cm^3 * (1+q)^3 / 8 * (x/0.07)^7 * (10 Msun/m1)^2.
        # 8(1+q)^-3 is the DENOMINATOR (verified against the typeset
        # equation).  An earlier version multiplied by 8/(1+q)^3, giving a
        # cap too permissive by 64/(1+q)^6 ~ 2.6 at q ~ 0.7; harmless for
        # the rho_max = 1e7 runs, where the prior box bound 97% of samples.
        rho_cap = (1.3e8 * (1.0 + q) ** 3 / 8.0
                   * (x / 0.07) ** 7 * (10.0 / mass_1) ** 2)
        out["sr_density_ratio"] = out["rho_phi"] / rho_cap
    return out
