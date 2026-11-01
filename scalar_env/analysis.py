"""
Section V -- Results: figures and tables.

VI.A  compare_with_catalog     vacuum posteriors vs published GWTC samples
VI.B  environment_corner       corner plot of (alpha, epsilon, tau_d, mu_s)
      vacuum_vs_env_overlay    chirp mass / chi_eff shift (Roy et al. Fig 2)
      phase_difference_plot    best-fit dephasing delta_psi(f) (Fig 4)
      mu_s_credible_interval   90% CI on the boson mass
VI.C  results_table            per-event summary (markdown)
"""

from pathlib import Path

import numpy as np

from .waveform import delta_psi_env, scalar_mass_ev

FIGDIR = Path("figures")

CORNER_KWARGS = dict(bins=40, smooth=1.0, show_titles=True,
                     quantiles=[0.05, 0.5, 0.95],
                     levels=(0.39, 0.86),          # 1 & 2 sigma in 2D
                     plot_datapoints=False, fill_contours=True)
# Overlaid posteriors have their own bin ranges (a shared one would
# over-smooth the narrower), so each 1-D histogram is drawn as a density.
# A new dict per call: corner writes its colour into the one it is given.
DENSITY = lambda: dict(density=True)
LABELS = {"chirp_mass": r"$\mathcal{M}\,[M_\odot]$", "mass_ratio": r"$q$",
          "chi_eff": r"$\chi_{\rm eff}$", "luminosity_distance": r"$d_L$ [Gpc]"}
SCALE = {"luminosity_distance": 1e-3}   # Mpc -> Gpc


def _save(fig, name):
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")


# ----------------------------------------------------------------- VI.A
def compare_with_catalog(result, catalog_samples, name="vacuum_vs_gwtc",
                         catalog_label="GWTC-2.1",
                         params=("chirp_mass", "mass_ratio", "chi_eff",
                                 "luminosity_distance")):
    """Overlay our vacuum posterior on the published GWTC posterior.

    catalog_samples: dict-like of arrays keyed by the same parameter names
    (e.g. loaded from the PESummary file on GWOSC/Zenodo).
    """
    import corner
    import matplotlib.pyplot as plt

    column = lambda samples, p: np.asarray(samples[p]) * SCALE.get(p, 1.0)
    ours = np.column_stack([column(result.posterior, p) for p in params])
    theirs = np.column_stack([column(catalog_samples, p) for p in params])

    labels = [LABELS.get(p, p) for p in params]
    fig = corner.corner(theirs, labels=labels, color="C1", hist_kwargs=DENSITY(),
                        **CORNER_KWARGS)
    corner.corner(ours, fig=fig, color="C0", hist_kwargs=DENSITY(), **CORNER_KWARGS)
    fig.legend(handles=[plt.Line2D([], [], color="C1", label=catalog_label),
                        plt.Line2D([], [], color="C0", label="this work")],
               loc="upper right")
    _save(fig, name)
    return fig


# ----------------------------------------------------------------- VI.B
def environment_corner(result_env, name="env_corner"):
    """Corner plot of the cloud parameters, including derived mu_s."""
    import corner

    post = result_env.posterior
    mu_s = scalar_mass_ev(post["alpha_cloud"], post["total_mass_source"])
    samples = np.column_stack([post["alpha_cloud"],
                               post["rho_phi"] / 1e6,
                               np.log10(mu_s)])
    labels = [r"$\alpha$", r"$\bar\rho_\phi$ [$10^6$ g/cm$^3$]",
              r"$\log_{10}(\mu_s/{\rm eV})$"]
    fig = corner.corner(samples, labels=labels, color="C2",
                        title_kwargs=dict(fontsize=9), **CORNER_KWARGS)
    _save(fig, name)
    return fig


def vacuum_vs_env_overlay(result_vac, result_env, name="mc_chieff_shift",
                          params=("chirp_mass", "chi_eff")):
    """The headline posterior shift: Roy et al. saw Mc 10.14 -> 9.69 Msun
    and chi_eff 0.14 -> ~0 for GW190728."""
    import corner
    import matplotlib.pyplot as plt

    vac = np.column_stack([result_vac.posterior[p] for p in params])
    env = np.column_stack([result_env.posterior[p] for p in params])
    labels = [r"$\mathcal{M}_c\,[M_\odot]$", r"$\chi_{\rm eff}$"]

    fig = corner.corner(vac, labels=labels, color="C0", hist_kwargs=DENSITY(),
                        **CORNER_KWARGS)
    corner.corner(env, fig=fig, color="C3", hist_kwargs=DENSITY(), **CORNER_KWARGS)
    fig.legend(handles=[plt.Line2D([], [], color="C0", label="vacuum"),
                        plt.Line2D([], [], color="C3", label="environmental")],
               loc="upper right")
    _save(fig, name)
    return fig


def phase_difference_plot(result_env, name="dephasing", f_low=20.0,
                          f_high=512.0):
    """delta_psi_env(f) for the maximum-posterior cloud parameters."""
    import matplotlib.pyplot as plt

    best = result_env.posterior.iloc[
        result_env.posterior["log_likelihood"].idxmax()]
    f = np.geomspace(f_low, f_high, 500)
    dpsi = delta_psi_env(f, best["mass_1"], best["mass_2"],
                         best["alpha_cloud"], best["rho_phi"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(f, np.abs(dpsi))
    ax.set_xlabel("GW frequency [Hz]")
    ax.set_ylabel(r"$|\delta\Psi_{\rm env}(f)|$ [rad]")
    ax.set_title("Best-fit environmental dephasing")
    _save(fig, name)
    return fig


def mu_s_credible_interval(result_env, ci=0.90):
    """(median, lower, upper) of the boson rest mass mu_s in eV."""
    post = result_env.posterior
    mu_s = scalar_mass_ev(post["alpha_cloud"], post["total_mass_source"])
    lo, mid, hi = np.quantile(mu_s, [(1 - ci) / 2, 0.5, (1 + ci) / 2])
    return mid, lo, hi


# ----------------------------------------------------------------- VI.C
def results_table(rows, path="results_table.md"):
    """rows: iterables of (event, snr, lnb, lnb_err, mu_s_ci) -> markdown."""
    lines = ["| Event | SNR | ln B^env_vac | mu_s 90% CI [eV] |",
             "|-------|-----|--------------|------------------|"]
    for event, snr, lnb, err, (mid, lo, hi) in rows:
        lines.append(f"| {event} | {snr:.1f} | {lnb:+.2f} +/- {err:.2f} "
                     f"| {mid:.2e} ({lo:.2e} - {hi:.2e}) |")
    table = "\n".join(lines)
    Path(path).write_text(table, encoding="utf-8")
    return table
