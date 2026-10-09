"""Section III.B validation -- no data or LAL required (numpy + matplotlib).

Left: dephasing at GW190728-like masses across the density prior range.
Right: dependence on alpha at fixed density, plus the Roy et al. Fig. 1
NR anchor configuration (60 Msun equal-mass, alpha = 0.43,
rho_phi = 8.6e5 g/cm^3), which produced a visible frequency-chirp
acceleration over the last NR orbits and ln B = 3.8 on reinjection.
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

f_nr = np.geomspace(20.0, 220.0, 400)
dpsi_nr = delta_psi_env(f_nr, 30.0, 30.0, alpha_cloud=0.43, rho_phi=8.6e5)
ax2.loglog(f_nr, np.abs(dpsi_nr), "k--",
           label=r"Roy Fig. 1: $60\,M_\odot$, $\alpha=0.43$")
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
print("Anchor check: the 60 Msun / alpha=0.43 / 8.6e5 g/cm^3 curve should "
      "give O(1-10) rad of accumulated inspiral dephasing, consistent with "
      "the visible NR chirp acceleration in Roy et al. Fig. 1.")
