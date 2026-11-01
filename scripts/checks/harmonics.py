"""The harmonic-by-harmonic waveform model (Section III.B).

  1. The pieces IMRPhenomXPHM is split into, one per harmonic m, add up
     to the full waveform.
  2. Share of the power above 20 Hz in each harmonic.
  3. With zero density both source models return the vacuum waveform.
  4. Dephasing of each harmonic at 25 Hz against the dominant one,
     (m/2) delta_psi(2f/m) / delta_psi(f); (m/2)^(43/6) at low density.
  5. For an approximant with the dominant harmonic alone, the two source
     models coincide.
  6. Cost of one call of each model.
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from bilby.gw.source import lal_binary_black_hole
from scalar_env.waveform import (_HARMONICS, delta_psi_env, scalar_cloud_bbh,
                                 scalar_cloud_bbh_dominant_mode)

F = np.linspace(0, 1024, 8193)                     # the analysis grid: 8 s, 2048 Hz
BAND = F >= 20
BINARY = dict(mass_1=14.0, mass_2=9.5, luminosity_distance=900.0, a_1=0.3, tilt_1=0.6,
              phi_12=1.0, a_2=0.2, tilt_2=1.1, phi_jl=2.0, theta_jn=0.9, phase=1.3)
CLOUD = dict(alpha_cloud=0.1, rho_phi=5e7)
LAL = dict(reference_frequency=20.0, minimum_frequency=20.0)
XPHM = dict(waveform_approximant="IMRPhenomXPHM", **LAL)


def difference(a, b):
    """Largest |a - b| over both polarizations, relative to the largest |b|."""
    return max(np.abs(a[p] - b[p]).max() / np.abs(b[p]).max() for p in b)


full = lal_binary_black_hole(F, **BINARY, **XPHM)
pieces = {m: lal_binary_black_hole(F, **BINARY, mode_array=modes, **XPHM)
          for m, modes in _HARMONICS["IMRPhenomXPHM"].items()}
total = {p: sum(h[p] for h in pieces.values()) for p in full}
print(f"1. sum of the harmonic pieces vs the full waveform: {difference(total, full):.1e}")

power = {m: sum(np.sum(np.abs(h[p][BAND]) ** 2) for p in h) for m, h in pieces.items()}
print("2. share of |h|^2 above 20 Hz: "
      + ", ".join(f"m={m} {v / sum(power.values()):.2%}" for m, v in sorted(power.items())))

vacuum = dict(CLOUD, rho_phi=0.0)
print("3. zero density vs vacuum: harmonic by harmonic "
      f"{difference(scalar_cloud_bbh(F, **BINARY, **vacuum, **XPHM), full):.1e}, dominant mode "
      f"{difference(scalar_cloud_bbh_dominant_mode(F, **BINARY, **vacuum, **XPHM), full):.1e}")

grid = np.linspace(5.0, 400.0, 40000)
for rho in (1e5, CLOUD["rho_phi"]):                # linear regime; a posterior-like density
    dpsi = delta_psi_env(grid, BINARY["mass_1"], BINARY["mass_2"], CLOUD["alpha_cloud"], rho)
    print(f"4. rho_phi = {rho:.0e}: dephasing at 25 Hz {np.interp(25.0, grid, dpsi):+.2f} rad; "
          "over it, " + ", ".join(
              f"m={m} {0.5 * m * np.interp(50.0 / m, grid, dpsi) / np.interp(25.0, grid, dpsi):.2f}"
              for m in sorted(pieces)))

xp = dict(waveform_approximant="IMRPhenomXP", **LAL)
print("5. IMRPhenomXP, harmonic by harmonic vs dominant mode: "
      f"{difference(scalar_cloud_bbh(F, **BINARY, **CLOUD, **xp), scalar_cloud_bbh_dominant_mode(F, **BINARY, **CLOUD, **xp)):.1e}")

for name, model in (("harmonic by harmonic", scalar_cloud_bbh),
                    ("dominant mode", scalar_cloud_bbh_dominant_mode)):
    start = time.perf_counter()
    for _ in range(20):
        model(F, **BINARY, **CLOUD, **XPHM)
    print(f"6. {name}: {(time.perf_counter() - start) / 20 * 1e3:.0f} ms per call")
