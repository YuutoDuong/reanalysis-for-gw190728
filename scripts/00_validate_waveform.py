"""Section III.B validation -- no data or LAL required (numpy + matplotlib).

Left: dephasing at GW190728-like masses across the density prior range.
Right: dependence on alpha at fixed density.

Roy et al.'s numerical-relativity waveform (60 Msun, equal masses,
alpha = 0.43) is not drawn: its density near the binary is quoted only as
a range, 10^7.3-10^8.3 g/cm^3 (the 8.6e5 of their Fig. 1 is the value at
the edge of the simulation box), and beta ~ 2 there, outside the regime
of the coupling function.  scripts/checks/roy_figures.py compares the
model with their results instead.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.waveform import delta_psi_env, scalar_mass_ev

M1, M2 = 12.5, 8.0                      # GW190728-like [Msun]
f = np.geomspace(20.0, 400.0, 600)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

for rho in (1e3, 1e4, 1e5, 1e6):
    dpsi = delta_psi_env(f, M1, M2, alpha_cloud=0.2, rho_phi=rho)
    ax1.loglog(f, np.abs(dpsi),
               label=rf"$\bar\rho_\phi=10^{{{int(np.log10(rho))}}}$ g/cm$^3$")
ax1.set_title(rf"GW190728-like, $\alpha=0.2$, varying $\bar\rho_\phi$")

for alpha in (0.05, 0.1, 0.2, 0.4):
    dpsi = delta_psi_env(f, M1, M2, alpha_cloud=alpha, rho_phi=1e5)
    ax2.loglog(f, np.abs(dpsi),
               label=rf"$\alpha={alpha}$ ($\mu_s$={scalar_mass_ev(alpha, M1 + M2):.1e} eV)")

ax2.set_title(r"$\bar\rho_\phi=10^5$ g/cm$^3$, varying $\alpha$")

for ax in (ax1, ax2):
    ax.set_xlabel("GW frequency [Hz]")
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)   # 1-radian detectability
    ax.legend(fontsize=8)
    # Explicit decade-ish ticks: matplotlib's default log minor labels
    # collide on a narrow axis.
    ax.set_xticks([20, 50, 100, 200, 400], minor=False)
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(20, 400)
ax1.set_ylabel(r"$|\delta\Psi_{\rm env}(f)|$ [rad]")

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/validation_dephasing.png", dpi=200, bbox_inches="tight")
print("figures/validation_dephasing.png written.")
