"""Paper versions of the robustness and background figures (no sampling).

Scripts 04-06 draw quick-look figures with raw run labels, a fixed
x-range and raw ln B; these are the versions the paper uses: grouped,
labelled rows, a histogram whose range always covers every clean
injection, and every ln B for the full-grid likelihood with the mass-cut
offset included.  Reads reweighted_lnb.json (script 17) and, for the
injections, injections.background (full grid once script 30 has run);
writes figures/.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env.analysis import environment_corner, vacuum_vs_env_overlay
from scalar_env.inference import mass_cut_offset, OUTDIR
from scalar_env.injections import background, plot_sr_comparison, select_sr_subset

OFFSET = mass_cut_offset()
LNB = {label: (v["exact"], v["err"]) for label, v in   # full grid, offset included
       json.loads(Path("reweighted_lnb.json").read_text())["lnb"].items()}
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
        "gw190728_env_rho_max2e8"]
ROY = {5: 3.4, 6: 3.5, 7: 3.5, 8: 2.8}          # Roy et al., SR priors

# (group, [(row label, run label), ...]); agnostic rows use rho_max = 2e8
GROUPS = [
    ("Density bound", [(r"$\rho_{\max}=10^{6}$", "gw190728_env_rho_max1e6"),
                       (r"$\rho_{\max}=10^{7}$", "gw190728_env_rho_max1e7"),
                       (r"$\rho_{\max}=10^{8}$", "gw190728_env_rho_max1e8")]),
    ("Prior on $\\alpha$", [("log-uniform", "gw190728_env_log_alpha")]),
    ("Sampler", [("MultiNest", "gw190728_env_pymultinest"),
                 ("nessai", "gw190728_env_nessai")]),
    ("Waveform", [("IMRPhenomPv2", "gw190728_env_imrphenompv2"),
                  ("IMRPhenomXP", "gw190728_env_imrphenomxp")]),
    ("Noise / network", [("Welch 8 s", "gw190728_env_psd8s"),
                         ("Welch 16 s", "gw190728_env_psd16s"),
                         ("H1 only", "gw190728_env_only_H1"),
                         ("L1 only", "gw190728_env_only_L1"),
                         ("H1+L1+V1", "gw190728_env_hlv"),
                         ("H1+L1+V1, BayesWave", "gw190728_env_hlv_gwtc21psd")]),
    ("Phase", [("sampled", "gw190728_env_phase_sampled"),
               ("sampled, H1+L1+V1, BayesWave", "gw190728_env_hlv_gwtc21psd_phase_sampled")]),
    ("Superradiance, $2\\times10^{8}$",
     [(rf"$\tau_d=10^{e}$ yr", f"gw190728_env_sr_eq9_taud1e{e}_rho2e8") for e in (5, 6, 7, 8)]),
    ("Superradiance, $10^{7}$",
     [(rf"$\tau_d=10^{e}$ yr", f"gw190728_env_sr_eq9_taud1e{e}") for e in (5, 6, 7, 8)]),
    ("Superradiance, H1+L1+V1, BayesWave",
     [(r"$\tau_d=10^{6}$ yr", "gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8"),
      (r"$\tau_d=10^{6}$ yr, phase sampled",
       "gw190728_env_hlv_gwtc21psd_phase_sampled_sr_eq9_taud1e6_rho2e8")]),
]


def robustness_figure(name="lnb_robustness_paper"):
    base = np.array([LNB[l][0] for l in BASE])
    rows = [("Baseline (4 runs)", base.mean(), base.std(ddof=1), None, None)]
    for group, members in GROUPS:
        rows.append((group, None, None, None, None))
        for text, label in (m for m in members if m[1] in LNB):   # skip failed runs
            roy = ROY[int(label.split("taud1e")[1][0])] if "sr_eq9" in label else None
            rows.append(("    " + text, *LNB[label], roy, label))

    fig, ax = plt.subplots(figsize=(5.2, 0.27 * len(rows) + 0.8))
    ax.axvspan(-1, 1, color="0.9", zorder=0, label=r"$|\ln B|<1$")
    ax.axvline(base.mean(), color="C0", lw=0.8, ls="--", zorder=1)
    for y, (text, lnb, err, roy, _) in enumerate(rows):
        if lnb is None:
            continue
        ax.errorbar(lnb, y, xerr=err, fmt="o", color="C0", ms=4, capsize=2, zorder=3)
        if roy is not None:
            ax.plot(roy, y, marker="D", color="C3", ms=4, zorder=3)
    ax.plot([], [], "D", color="C3", ms=4, label="Roy et al.")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row[1] is None:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\ln\mathcal{B}^{\rm env}_{\rm vac}$")
    ax.legend(fontsize=7, loc="upper right")
    _save(fig, name)


def background_figure(name="injection_background_paper", roy=3.5):
    observed = np.mean([LNB[l][0] for l in BASE])
    lnb = np.array([agnostic for agnostic, _ in background().values()])
    hi = 6.0
    on = lnb[lnb <= hi]
    lo = np.floor(on.min()) - 0.5
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(on, bins=np.arange(lo, hi + 0.5, 0.5), color="C0", alpha=0.75,
            label=f"vacuum injections ($N={lnb.size}$)")
    ax.axvline(observed, color="C3", ls="--", label=f"GW190728 ({observed:+.2f})")
    ax.axvline(roy, color="C1", ls=":", label=f"Roy et al. ({roy})")
    n_off = int((lnb > hi).sum())
    if n_off:
        ax.annotate(f"+{n_off} off scale ($\\ln\\mathcal{{B}}={lnb.max():.0f}$,\n"
                    "glitch segment)", xy=(0.98, 0.45), xycoords="axes fraction",
                    ha="right", fontsize=7, color="C0")
    ax.set_xlabel(r"$\ln\mathcal{B}^{\rm env}_{\rm vac}$")
    ax.set_ylabel("injections")
    ax.legend(fontsize=7, loc="upper right")
    _save(fig, name)


def posterior_figures():
    """Figs. 3-4 from the four baseline runs combined, as the text quotes
    them (script 02 draws them from the first run alone)."""
    load = lambda l: bilby.result.read_in_result(str(OUTDIR / l / f"{l}_result.json"))
    env = SimpleNamespace(posterior=pd.concat([load(l).posterior for l in BASE],
                                              ignore_index=True))
    environment_corner(env, name="env_corner_paper")
    vacuum_vs_env_overlay(load("gw190728_vacuum"), env, name="mc_chieff_shift_paper")


def sr_comparison_figure(name="injection_sr_comparison_paper"):
    errors = json.loads(Path("injection_sr_lnb.json").read_text())
    values = background()
    subset = {key: dict(lnb_agnostic=values[key][0], lnb_sr=values[key][1],
                        err=errors[key]["err"])
              for key, _ in select_sr_subset("injection_lnb.json")}
    observed = LNB["gw190728_env_sr_eq9_taud1e6_rho2e8"][0]
    plot_sr_comparison(subset, observed_sr=observed, name=name)


def _save(fig, name):
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    print(f"full-grid ln B, mass-cut offset {OFFSET:+.4f} included")
    robustness_figure()
    background_figure()
    sr_comparison_figure()
    posterior_figures()
    print("figures/*_paper.* written")
