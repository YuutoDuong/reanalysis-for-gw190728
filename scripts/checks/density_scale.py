"""Would a different density normalisation reconcile us with Roy et al.?

Their posteriors put the density of GW190728 several times above ours
while removing the same fraction of cycles (checks/roy_figures.py).  Test
the simplest reading: their code needs N times our density for the same
waveform.  Their priors, stated in their units, are then truncations of
ours in our units -- a box of 1e9 / N for the agnostic analysis and
2.09e8 / N for the superradiance ones (the ends of their plotted priors),
alpha <= 0.5, and the cloud-mass cut and superradiance density cap
divided by N -- so the truncation identity gives the Bayes factors they
would obtain, and the samples inside each region their posteriors:

    ln B_theirs(N) = ln B_ours + ln(P / f),

P and f being the fractions of our posterior and prior inside the region.
Print both against their published values.  No sampling; ~1 min.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import bilby
from scalar_env.inference import RunConfig, build_priors, exact_posterior
from scalar_env.waveform import T_SUN, RHO_GEO_PER_GCM3

bilby.core.utils.logger.setLevel("ERROR")
SCALES = (1, 3, 5, 7, 10)
TAUS = (1e5, 1e6, 1e7, 1e8)
AGNOSTIC_BOX, SR_BOX = 1e9, 2.09e8
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
CONFIGURATIONS = {  # name: agnostic runs at rho_max = 2e8
    "H1+L1, Welch": ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2", "gw190728_env_rho_max2e8"],
    "H1+L1+V1, BayesWave": ["gw190728_env_hlv_gwtc21psd"],
    "H1+L1+V1, BayesWave, phase sampled": ["gw190728_env_hlv_gwtc21psd_phase_sampled"],
}
THEIRS = {  # ln B, and median density in 1e8 g/cm^3 read off their Figs. 3 and 6
    "agnostic": (0.4, 0.44), 1e5: (3.4, 0.73), 1e6: (3.5, 0.75), 1e7: (3.5, 0.72), 1e8: (2.8, 0.64)}
LNB = json.loads(Path("reweighted_lnb.json").read_text())["lnb"]


def regions(samples, scale):
    """Their priors as masks on our samples, the density being `scale`
    times ours in their units: {"agnostic" or tau_d: mask}."""
    q, alpha, rho = samples["mass_ratio"], samples["alpha_cloud"], samples["rho_phi"] * scale
    m_tot = samples["chirp_mass"] * (1 + q) ** 1.2 / q ** 0.6
    m1, x = m_tot / (1 + q), alpha / (1 + q)
    common = (alpha <= 0.5) & (10 * rho * RHO_GEO_PER_GCM3 * (m_tot * T_SUN) ** 2 / alpha ** 6 < 1)
    out = {"agnostic": common & (rho < AGNOSTIC_BOX)}
    for tau in TAUS:                                   # priors._env_conversion, Roy et al. SM Eqs. (9)-(10)
        s = (m1 / 10) * (1e6 / tau)
        cap = 1.3e8 * (1 + q) ** 3 / 8 * (x / 0.07) ** 7 * (10 / m1) ** 2
        out[tau] = (common & (x >= 0.02 * s ** (1 / 9)) & (x <= 0.11 * s ** (1 / 15))
                    & (rho <= cap) & (rho < SR_BOX))
    return out


if __name__ == "__main__":
    # our prior in a box wide enough to hold theirs for any scale; ours is its part below 2e8
    prior = build_priors(RunConfig(label="_", model="environment", prior_kwargs=dict(rho_max_gcm3=AGNOSTIC_BOX)), 0.0)
    unit = np.random.default_rng(0).uniform(size=(len(KEYS), 3_000_000))
    draws = pd.DataFrame({k: prior[k].rescale(u) for k, u in zip(KEYS, unit)})
    draws = draws[np.asarray(prior.evaluate_constraints({k: draws[k].to_numpy() for k in KEYS}), dtype=bool)]
    ours = (draws["rho_phi"] < 2e8).mean()

    print("ln B they would obtain (median density in 1e8 g/cm^3, their units), by density scale N")
    print(f"{'':10s}" + "".join(f"{str(k) if k == 'agnostic' else f'SR {k:.0e}':>15s}" for k in THEIRS))
    print(f"{'theirs':10s}" + "".join(f"{lnb:+9.1f} ({rho:.2f})" for lnb, rho in THEIRS.values()))
    for name, labels in CONFIGURATIONS.items():
        posterior = pd.concat([exact_posterior(l) for l in labels], ignore_index=True)
        base = np.mean([LNB[l]["exact"] for l in labels])
        print(f"{name}: our agnostic ln B at 2e8 {base:+.2f}")
        for scale in SCALES:
            inside, volume = regions(posterior, scale), regions(draws, scale)
            cells = [(base + np.log(inside[k].mean() * ours / volume[k].mean()),
                      np.median(posterior["rho_phi"][inside[k]]) * scale / 1e8) for k in THEIRS]
            print(f"  N = {scale:<5d}" + "".join(f"{lnb:+9.2f} ({rho:.2f})" for lnb, rho in cells))
