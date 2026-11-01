"""Which of our analysis configurations resembles that of Roy et al.?

They state neither their detector network, nor their noise spectra, nor
how they treat the coalescence phase.  Their Supplemental Material does
give numbers that depend on those choices, all for GW190728 under the
superradiance prior at tau_d = 1e6 yr: the gain in median log-likelihood
over vacuum, the chirp mass and effective spin under both models, and (in
their Fig. 6) the density.
Print the same quantities for every configuration we sampled, from the
posteriors of the exact likelihood (script 17), next to theirs.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from scalar_env.inference import exact_posterior

SR = "_sr_eq9_taud1e6_rho2e8"
CONFIGURATIONS = {  # name: label fragment shared by its vacuum and environmental runs
    "H1+L1, Welch": "",
    "H1+L1, Welch, phase sampled": "_phase_sampled",
    "H1+L1, BayesWave": "_h1l1_gwtc21psd",
    "H1+L1+V1, Welch": "_hlv",
    "H1+L1+V1, BayesWave": "_hlv_gwtc21psd",
    "H1+L1+V1, BayesWave, phase sampled": "_hlv_gwtc21psd_phase_sampled",
    "H1 only": "_only_H1",
    "L1 only": "_only_L1",
}
ROY = ("Roy et al.", "+0.4", "+3.5", "", "+3.7", "10.14 +0.10 -0.09", "9.69 +0.27 -0.28",
       "+0.14 +0.18 -0.11", "+0.00 +0.15 -0.13", "7.46 +3.02 -4.01")   # density: checks/roy_figures.py
LNB = json.loads(Path("reweighted_lnb.json").read_text())["lnb"]


def interval(x, sign=""):
    lo, mid, hi = np.quantile(x, [0.05, 0.5, 0.95])
    return f"{mid:{sign}.2f} +{hi - mid:.2f} -{mid - lo:.2f}"


def row(name, fragment):
    """One configuration: its agnostic and superradiance runs against its vacuum run."""
    labels = {"agn": f"gw190728_env{fragment}", "sr": f"gw190728_env{fragment}{SR}"}
    vacuum = exact_posterior(f"gw190728_vacuum{fragment}")
    env = {prior: exact_posterior(label) for prior, label in labels.items() if label in LNB}
    cell = lambda prior, text: text(env[prior]) if prior in env else ""
    lnb = lambda prior: cell(prior, lambda _: f"{LNB[labels[prior]]['exact']:+.2f}")
    gain = lambda prior: cell(prior, lambda p: f"{p['log_likelihood'].median() - vacuum['log_likelihood'].median():+.2f}")
    return (name, lnb("agn"), lnb("sr"), gain("agn"), gain("sr"),
            interval(vacuum["chirp_mass"]), cell("sr", lambda p: interval(p["chirp_mass"])),
            interval(vacuum["chi_eff"], "+"), cell("sr", lambda p: interval(p["chi_eff"], "+")),
            cell("sr", lambda p: interval(p["rho_phi"] / 1e7)))


HEADER = ("configuration", "lnB agn", "lnB SR", "gain agn", "gain SR", "Mc vacuum", "Mc SR",
          "chi_eff vacuum", "chi_eff SR", "rho SR [1e7]")
rows = [HEADER, ROY] + [row(name, fragment) for name, fragment in CONFIGURATIONS.items()
                        if f"gw190728_env{fragment}" in LNB]
widths = [max(len(r[i]) for r in rows) for i in range(len(HEADER))]
for r in rows:
    print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
print("\nln B at rho_max = 2e8 (mass-cut offset included); gain = median log-likelihood of the\n"
      "posterior, environment minus vacuum; chirp mass in the detector frame; 90% intervals.")
