"""Separate the two changes in the catalogue noise model.

Relative to the baseline, script 20 changed two things at once: it added
Virgo and replaced the off-source Welch PSDs with the BayesWave PSDs of
GWTC-2.1.  Fill in the missing cells of that 2 x 2 table -- H1+L1 with
the catalogue PSDs, and H1+L1+V1 with Welch PSDs under the superradiance
prior -- so that the effect of each change can be read off separately.
The other cells are loaded from cache.

Runs beside the injection campaign on 4 cores at low priority: ~1-2 days.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig, run_pe, ln_bayes_factor, mass_cut_offset

NPOOL = 4
SR = dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6)
CELLS = {  # name: (detectors, PSD source, label tag)
    "H1+L1,    Welch":    (("H1", "L1"), "welch", ""),
    "H1+L1+V1, Welch":    (("H1", "L1", "V1"), "welch", "_hlv"),
    "H1+L1,    GWTC-2.1": (("H1", "L1"), "gwtc21", "_h1l1_gwtc21psd"),
    "H1+L1+V1, GWTC-2.1": (("H1", "L1", "V1"), "gwtc21", "_hlv_gwtc21psd"),
}


def configs(detectors, psd_source, tag):
    """Vacuum, agnostic and superradiance runs on one noise model."""
    noise = dict(detectors=detectors, psd_source=psd_source, npool=NPOOL)
    agnostic = RunConfig(label=f"gw190728_env{tag}", model="environment",
                         prior_kwargs=dict(rho_max_gcm3=2e8), **noise)
    return (RunConfig(label=f"gw190728_vacuum{tag}", model="vacuum", **noise),
            agnostic, agnostic.variant("sr_eq9_taud1e6_rho2e8", prior_kwargs=SR))


if __name__ == "__main__":
    offset = mass_cut_offset()
    table = {}
    for name, cell in CELLS.items():
        vac, *envs = [run_pe(c) for c in configs(*cell)]
        table[name] = [ln_bayes_factor(e, vac) for e in envs]

    print(f"\nln B at rho_max = 2e8 (mass-cut offset included)\n"
          f"{'':20s}{'agnostic':>19s}{'SR, tau_d = 1e6':>19s}")
    for name, cols in table.items():
        print(f"{name:20s}" + "".join(f"{x + offset:+11.2f} +- {e:.2f}" for x, e in cols))
