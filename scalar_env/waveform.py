"""
Section III -- Waveform model.

Vacuum baseline (IV.A): IMRPhenomXPHM, called through Bilby / LALSimulation.

Environmental correction (IV.B), following Roy et al. (2026)
[arXiv:2510.17967]: the binary loses angular momentum to the surrounding
scalar configuration through *ionization* (perturbative scattering of
bound "gravitational molecule" l = 0 states into unbound ones).  The
time-averaged torque is their Eq. (2),

    Ldot_phi / M = -4 pi rho_phi M^2 I(v, alpha, q) / v^4 ,

with the coupling function I given by their Eq. (6) as a sum over the
binary's tidal multipoles (l, m > 0), where beta = R / r_phi = alpha^2/v^2
and r_phi = M / alpha^2.  The free environmental parameters are the mean
scalar density rho_phi (in g/cm^3, uniform prior) and alpha = mu_s M.

The correction is an inspiral dephasing plus the SPA amplitude change
that necessarily accompanies it (their Eq. 3, A ~ 1/sqrt(Omegadot)),

    h_env(f) = h_XPHM(f) * exp(-i delta_psi) / sqrt(1 + Ldot_phi/Ldot_GW),

applied below the (Schwarzschild) ISCO.  Validation anchor: Roy et al.
Fig. 1 (60 Msun equal-mass NR run, alpha = 0.43, rho_phi = 8.6e5 g/cm^3).

Conventions: geometric units G = c = 1 internally; masses carried in
seconds (m[s] = m[Msun] * T_SUN).  Frequencies are GW frequencies in Hz.
"""

from math import gamma, pi

import numpy as np

T_SUN = 4.925490947641267e-06   # G Msun / c^3  [s]
YEAR = 3.15576e7                # Julian year   [s]
EV_PER_INV_MSUN = 1.3357e-10    # hbar / T_SUN in eV: mu_s = alpha * this / (M_tot/Msun)
RHO_GEO_PER_GCM3 = 6.674e-08    # G * (1 g/cm^3)  ->  geometric density [1/s^2]


# ----------------------------------------------------------------- IV.B, Eq. (6)
# (l, m) tidal multipoles with |Y_lm(pi/2, 0)|^2 / Gamma(l + 3/2)^2.
# l = 1 vanishes identically (mass factor 1 + (-1)^l q^(l-1) = 0) and
# modes with Y_lm(pi/2, 0) = 0 (odd l+m) are omitted.
#   |Y_lm(pi/2,0)|^2 = (2l+1)/(4 pi) * (l-m)!/(l+m)! * [P_l^m(0)]^2
# with P_2^2(0)=3, P_3^1(0)=3/2, P_3^3(0)=-15, P_4^2(0)=-15/2, P_4^4(0)=105.
_MODES = {
    (2, 2): (15.0 / (32 * pi)) / gamma(3.5) ** 2,
    (3, 1): (21.0 / (64 * pi)) / gamma(4.5) ** 2,
    (3, 3): (35.0 / (64 * pi)) / gamma(4.5) ** 2,
    (4, 2): (45.0 / (128 * pi)) / gamma(5.5) ** 2,
    (4, 4): (315.0 / (512 * pi)) / gamma(5.5) ** 2,
}


def ionization_coupling(v, alpha_cloud, q):
    """Dimensionless coupling I(v, alpha, q) of Roy et al. Eq. (6).

    beta = R / r_phi = (alpha / v)^2 for a Keplerian orbit at separation
    R = M / v^2 around a cloud of Bohr-like radius r_phi = M / alpha^2.

    Roy et al. write Eq. (6) entirely in terms of q_2 = m2/M = q/(1+q)
    (defined in their SM above the equation), NOT the mass ratio
    q = m2/m1: both the bracket [2(1 + q_2)/m]^(1/2-l) and the multipole
    mass factor [1 + (-1)^l q_2^(l-1)]^2 carry the subscript.  Verified
    against the typeset equation in the published PDF; the plain-text
    rendering of sub- and superscripts is ambiguous here.  The l = 1
    dipole still vanishes identically, since q_2^0 = 1.
    """
    beta = (alpha_cloud / v) ** 2
    q2 = q / (1.0 + q)                     # m2 / M, not m2 / m1
    total = np.zeros_like(np.asarray(v, dtype=float))
    for (ell, m), y_over_gamma in _MODES.items():
        mass_factor = (1.0 + (-1.0) ** ell * q2 ** (ell - 1)) ** 2
        total += (pi**2 * beta ** (0.25 + ell / 2.0)
                  * (2.0 * (1.0 + q2) / m) ** (0.5 - ell)
                  * mass_factor * y_over_gamma)
    return total


# ----------------------------------------------------------------- IV.B, SPA
def _env_response(frequency_array, mass_1, mass_2, alpha_cloud, rho_phi,
                  f_cut=None):
    """(delta_psi [rad], Ldot_phi / Ldot_GW) on the given frequency grid.

    Roy et al. obtain Omegadot from the *full* angular-momentum balance,
    Ldot = -(Ldot_GW + Ldot_phi) (their text above Eq. 3), with the
    Newtonian L = eta M^2 / v and quadrupole Ldot_GW = (32/5) eta^2 M v^7.
    Writing R = Ldot_phi / Ldot_GW, the inspiral is shortened by

        -delta(dt/df) = |dL/df| / Ldot_GW * R / (1 + R) ,

    and the stationary-phase correction is

        delta_psi(f) = -2 pi int_f^fcut (f'-f) [-delta(dt/df')] df' ,

    where the (f'-f) kernel absorbs the t_c and phi_c degeneracies
    (waveforms aligned at f_cut).

    The R/(1+R) factor matters: linearising it to R -- correct only while
    the environmental torque is a small correction to the GW torque --
    overstates delta_psi by up to an order of magnitude in the density
    range Roy et al.'s GW190728 posterior occupies (R ~ 3-40 at 20 Hz),
    and would drive the time to merger negative.  At their NR validation
    point (60 Msun, alpha = 0.43, rho_phi = 8.6e5) R = 0.21 and the two
    forms agree to 7%.  rho_phi is in g/cm^3.
    """
    m1, m2 = mass_1 * T_SUN, mass_2 * T_SUN
    m_tot = m1 + m2
    eta = m1 * m2 / m_tot**2
    q = m2 / m1
    if f_cut is None:
        f_cut = 1.0 / (6.0**1.5 * np.pi * m_tot)         # Schwarzschild ISCO

    f = np.asarray(frequency_array, dtype=float)
    dpsi, ratio = np.zeros_like(f), np.zeros_like(f)
    band = (f > 0) & (f < f_cut)
    if not band.any() or rho_phi <= 0:
        return dpsi, ratio

    fb = f[band]
    v = (np.pi * m_tot * fb) ** (1.0 / 3.0)
    rho_geo = rho_phi * RHO_GEO_PER_GCM3

    ldot_gw = (32.0 / 5.0) * eta**2 * m_tot * v**7
    ldot_env = (4.0 * np.pi * rho_geo * m_tot**3
                * ionization_coupling(v, alpha_cloud, q) / v**4)
    dl_df = eta * m_tot**2 / (3.0 * fb * v)              # |dL_orb/df|

    r = ldot_env / ldot_gw
    g = dl_df / ldot_gw * r / (1.0 + r)                  # -delta(dt/df)
    i0 = _cumtrapz_from_top(g, fb)                       # int_f^fmax g df'
    i1 = _cumtrapz_from_top(fb * g, fb)                  # int_f^fmax f' g df'
    dpsi[band] = -2.0 * np.pi * (i1 - fb * i0)
    ratio[band] = r
    return dpsi, ratio


def delta_psi_env(frequency_array, mass_1, mass_2, alpha_cloud, rho_phi,
                  f_cut=None):
    """Environmental SPA phase correction delta_psi(f)  [rad]."""
    return _env_response(frequency_array, mass_1, mass_2, alpha_cloud,
                         rho_phi, f_cut)[0]


def environment_transfer(frequency_array, mass_1, mass_2, alpha_cloud,
                         rho_phi, f_cut=None):
    """Complex factor h_env / h_vac from the environmental torque.

    The SPA amplitude carries a 1/sqrt(m Omegadot) factor (Roy et al.
    Eq. 3), so the extra torque suppresses the amplitude as well as
    advancing the phase: sweeping through a frequency interval faster
    leaves less signal in it.  Relative to vacuum,

        h_env / h_vac = exp(-i delta_psi) / sqrt(1 + R) .

    Both factors go to unity as R -> 0, so the vacuum limit (rho_phi = 0)
    is exact and the NR-validated configuration is essentially unchanged.
    """
    dpsi, ratio = _env_response(frequency_array, mass_1, mass_2,
                                alpha_cloud, rho_phi, f_cut)
    return np.exp(-1j * dpsi) / np.sqrt(1.0 + ratio)


def _cumtrapz_from_top(y, x):
    """Trapezoidal int_x^x[-1] y dx', returned on the same grid as x."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.zeros_like(y)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def scalar_mass_ev(alpha_cloud, total_mass):
    """Boson mass mu_s [eV] from Roy et al.'s alpha = mu_s M (G = c = hbar
    = 1) with M the *total* binary mass -- not the primary mass.  The
    primary's own coupling is alpha/(1+q) = mu_s m1.

    Pass the SOURCE-frame mass: alpha is frame-invariant, so the detector-
    frame mass (1+z)M would understate the rest mass by 1+z (~17% for
    GW190728).  For the same reason the sampled rho_phi, which the
    waveform pairs with detector-frame masses, is (1+z)^-2 times the
    physical density."""
    return alpha_cloud * EV_PER_INV_MSUN / total_mass


# ----------------------------------------------------------------- IV.B.5
def scalar_cloud_bbh(frequency_array, mass_1, mass_2, luminosity_distance,
                     a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn,
                     phase, alpha_cloud, rho_phi, **kwargs):
    """Bilby frequency-domain source model: IMRPhenomXPHM x environment_transfer.

    rho_phi in g/cm^3.  tau_d is an analysis setting that acts only
    through the prior constraints; it is dropped here in case it arrives
    via **kwargs (posteriors saved before it stopped being a DeltaFunction
    prior carry it as a column).

    Approximation: the same correction is applied to both polarizations
    and all multipoles (dominant-mode approximation).  This is ours, not
    Roy et al.'s -- they propagate it mode by mode with
    Phi_lm(f) = (m/2) Phi_22(2f/m) -- and is adequate only for
    near-equal-mass systems such as GW190728.
    """
    from bilby.gw.source import (lal_binary_black_hole,
                                 lal_binary_black_hole_relative_binning)

    kwargs.pop("tau_d", None)
    kwargs.setdefault("waveform_approximant", "IMRPhenomXPHM")

    # Under relative binning the likelihood requests the waveform on the
    # coarse bin-edge grid (kwargs["frequency_bin_edges"], with
    # kwargs["fiducial"] = 1 flagging the one full-resolution call);
    # the dephasing must be evaluated on whichever grid the strain uses.
    if "frequency_bin_edges" in kwargs or "fiducial" in kwargs:
        source, grid = lal_binary_black_hole_relative_binning, (
            frequency_array if kwargs.get("fiducial", 0)
            else kwargs["frequency_bin_edges"])
    else:
        source, grid = lal_binary_black_hole, frequency_array

    h = source(frequency_array, mass_1, mass_2, luminosity_distance,
               a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
               **kwargs)
    if h is None:
        return None

    transfer = environment_transfer(grid, mass_1, mass_2, alpha_cloud,
                                    rho_phi)
    return {pol: strain * transfer for pol, strain in h.items()}
