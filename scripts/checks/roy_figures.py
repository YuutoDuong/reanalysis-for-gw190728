"""What Roy et al.'s figures say about their priors and posteriors.

Their Figs. 3 and 6 (GW190728) are vector graphics in the HTML version of
arXiv:2510.17967v2, so the plotted curves can be read as numbers: each
curve is a path in page points, and the tick marks give the axes.  Print
the 5%, 50% and 95% quantiles of their prior and posterior curves next to
those of our exact-likelihood posteriors, and the height of their
agnostic density prior, which fixes the bound they did not quote.

Then compare the physical effect.  Their Supplemental Material gives the
number of cycles with and without the environment at their
maximum-likelihood sample, from four starting frequencies; print the
fraction our model keeps at our posterior median and at theirs.
Needs network access (three files of ~100 kB).
"""
import re
import sys
import urllib.request

import numpy as np

sys.path.insert(0, ".")
from scalar_env.inference import exact_posterior
from scalar_env.waveform import _env_response, scalar_mass_ev, T_SUN

URL = "https://arxiv.org/html/2510.17967v2/"
BLUE, ORANGE, GREEN, PINK = "#4169e1", "#ff7f0e", "#008000", "#cc79a7"
# panel: (file, page x of the tick at 0, page points per axis unit, page y of the axis)
RHO3 = ("gw190728_res_1e6.svg", 43.516, 94.33, 251.9)        # Fig. 3, density in 1e8 g/cm^3
MPHI3 = ("gw190728_res_1e6.svg", 53.174, 30.735, 49.42)      # Fig. 3, boson mass in 1e-12 eV
RHO3_HEIGHT = 44.625                                         # its page points per unit of probability density
RHO6 = {  # Fig. 6, density per delay time: (curve colour, panel)
    1e5: (BLUE, ("gw190728_density.svg", 29.75, 74.49, 231.3)),
    1e6: (ORANGE, ("gw190728_density.svg", 235.19, 74.49, 231.3)),
    1e7: (GREEN, ("gw190728_density.svg", 29.75, 74.49, 53.4)),
    1e8: (PINK, ("gw190728_density.svg", 235.19, 74.49, 53.4)),
}
CYCLES = {20: 42 / 52, 25: 30 / 35, 30: 22 / 25, 35: 17 / 18}   # their SM: environment / vacuum
FILES = {}


def curves(name):
    """[(colour, dashed, x, y)] of the data curves of one figure file."""
    if name not in FILES:
        FILES[name] = urllib.request.urlopen(URL + name).read().decode()
    out = []
    for attributes, d in re.findall(r'<path transform="matrix\(1,0,0,-1,0,[\d.]+\)"([^>]*?)d="([^"]+)"/>', FILES[name]):
        colour = re.search(r'stroke="(#[0-9a-f]+)"', attributes)
        if not colour or 'stroke-width="1.2"' not in attributes:
            continue
        out.append((colour.group(1), "dasharray" in attributes, *points(d)))
    return out


def points(d):
    """(x, y) of an SVG path made of M, L, H and V commands."""
    xy, command, pending = [(0.0, 0.0)], "M", []
    for token in re.findall(r"[MLHV]|-?[\d.]+", d):
        if token in "MLHV":
            command, pending = token, []
            continue
        pending.append(float(token))
        if command == "H":
            xy.append((pending.pop(), xy[-1][1]))
        elif command == "V":
            xy.append((xy[-1][0], pending.pop()))
        elif len(pending) == 2:
            xy.append(tuple(pending))
            pending = []
    return np.array(xy[1:]).T


def curve(panel, colour, dashed):
    """(x in axis units, density) of one curve of a panel."""
    name, x0, scale, y0 = panel
    for c, d, x, y in curves(name):
        if c == colour and d == dashed and abs(y.min() - y0) < 1 and abs(x.min() - x0) < 15:
            return (x - x0) / scale, y - y0
    raise LookupError((panel, colour, dashed))


def quantiles(x, density):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x))])
    return np.interp([0.05, 0.5, 0.95], cdf / cdf[-1], x)


def show(name, theirs, ours=None):
    text = lambda q: "{:.2f} [{:.2f}, {:.2f}]".format(q[1], q[0], q[2])
    print(f"  {name:34s} theirs {text(theirs)}" + (f"   ours {text(np.quantile(ours, [0.05, 0.5, 0.95]))}" if ours is not None else ""))


def cycles_kept(m1, m2, alpha, rho, start):
    """Cycles from `start` to the ISCO with the environment, over those in vacuum."""
    f = np.linspace(start, 1 / (6 ** 1.5 * np.pi * (m1 + m2) * T_SUN), 20000)
    ratio = _env_response(f, m1, m2, alpha, rho)[1]
    return np.sum(f ** (-8 / 3) / (1 + ratio)) / np.sum(f ** (-8 / 3))      # f dt/df of the vacuum chirp


if __name__ == "__main__":
    agnostic = exact_posterior("gw190728_env")
    sr = {tau: exact_posterior(f"gw190728_env_sr_eq9_taud1e{int(np.log10(tau))}_rho2e8") for tau in RHO6}
    m_phi = lambda p: scalar_mass_ev(p["alpha_cloud"], p["total_mass_source"]) / 1e-12

    x, y = curve(RHO3, ORANGE, True)
    height = y[(x > 0.1) & (x < 3.9)] / RHO3_HEIGHT
    print(f"their agnostic density prior: plotted to {x.max():.2f}e8 at {height.min():.3f}-{height.max():.3f} per 1e8, "
          f"{np.trapezoid(y / RHO3_HEIGHT, x):.2f} of its mass below that -> bound ~ {1 / height.mean():.1f}e8 g/cm^3")
    print("their superradiance density priors end at " + ", ".join(
        f"{curve(panel, colour, True)[0].max():.2f}e8" for colour, panel in RHO6.values()) + " (tau_d = 1e5..1e8 yr)")

    print("\ndensity [1e8 g/cm^3] and boson mass [1e-12 eV], median [5%, 95%]; ours: H1+L1, Welch, rho_max = 2e8")
    show("density, agnostic posterior", quantiles(*curve(RHO3, ORANGE, False)), agnostic["rho_phi"] / 1e8)
    show("boson mass, agnostic posterior", quantiles(*curve(MPHI3, ORANGE, False)), m_phi(agnostic))
    show("boson mass, agnostic prior", quantiles(*curve(MPHI3, ORANGE, True)))
    for tau, (colour, panel) in RHO6.items():
        show(f"density, SR posterior, tau {tau:.0e}", quantiles(*curve(panel, colour, False)), sr[tau]["rho_phi"] / 1e8)
        show(f"density, SR prior, tau {tau:.0e}", quantiles(*curve(panel, colour, True)))
    show("boson mass, SR posterior, tau 1e6", quantiles(*curve(MPHI3, BLUE, False)), m_phi(sr[1e6]))
    show("boson mass, SR prior, tau 1e6", quantiles(*curve(MPHI3, BLUE, True)))

    print("\nfraction of cycles kept, from 20, 25, 30 and 35 Hz (tau_d = 1e6 yr)")
    print("  their maximum-likelihood sample:       " + ", ".join(f"{v:.2f}" for v in CYCLES.values()))
    median = sr[1e6][["mass_1", "mass_2", "alpha_cloud", "rho_phi"]].median()
    for name, rho in (("our model at our posterior median", median["rho_phi"]),
                      ("our model at their median density", 1e8 * quantiles(*curve(RHO6[1e6][1], ORANGE, False))[1])):
        print(f"  {name} ({rho:.1e}): " + ", ".join(
            f"{cycles_kept(median['mass_1'], median['mass_2'], median['alpha_cloud'], rho, f):.2f}" for f in CYCLES))
