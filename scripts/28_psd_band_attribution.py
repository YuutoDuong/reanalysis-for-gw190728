"""Which part of the noise spectrum moves the Bayes factor?

Replacing the off-source Welch PSDs of H1 and L1 by the BayesWave PSDs of
the GWTC-2.1 release raises ln B by ~1.5.  Swap the PSD only in a chosen
set of frequency bins and recompute ln B by reweighting the baseline
runs' posterior samples,

    Z_hybrid = Z_RB * E_post[ L_hybrid / L_RB ],

so every hybrid is evaluated at the same samples and the differences
between hybrids carry little noise.  The bins whose swap moves ln B are
the ones the evidence rests on.  Swaps: by detector and band, by 5-Hz band
below 50 Hz, and below 50 Hz split into the bins where the two PSDs differ
by more than 30% (lines, mostly) and the rest.  Also prints the
BayesWave / Welch PSD ratio per band.  ~5 min on 10 cores.
"""

import copy
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env import data
from scalar_env.inference import RunConfig, exact_likelihood, mass_cut_offset, OUTDIR

bilby.core.utils.logger.setLevel("WARNING")
EVENT = "GW190728_064510"
N_SAMPLES = 200
NPOOL = int(os.environ.get("NPOOL", max(1, (os.cpu_count() or 2) - 2)))
BANDS = [(20, 30), (30, 50), (50, 100), (100, 1024)]
FINE = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 50)]
RUNS = {"vacuum": "gw190728_vacuum", "agnostic": "gw190728_env_seed1",
        "superradiance": "gw190728_env_sr_eq9_taud1e6_rho2e8"}
WELCH, TRIGGER = data.fetch_event(EVENT)
CATALOGUE, _ = data.fetch_event(EVENT, psd_source="gwtc21")


def hybrid(select):
    """Welch PSDs, with the catalogue PSD in the bins select(detector,
    frequencies, catalogue / Welch) marks."""
    ifos = copy.deepcopy(WELCH)
    for ifo, cat in zip(ifos, CATALOGUE):
        f, welch, bw = (ifo.frequency_array, ifo.power_spectral_density_array,
                        cat.power_spectral_density_array)
        psd = np.where(select(ifo.name, f, bw / welch), bw, welch)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=f, psd_array=psd)
    return ifos


def band(detector, lo, hi):
    return lambda d, f, r: (d == detector) & (f >= lo) & (f < hi)


LOW = lambda f: (f >= 20) & (f < 50)
TARGETS = {
    "Welch": WELCH,
    "BayesWave": hybrid(lambda d, f, r: np.ones(f.size, bool)),
    **{f"{d} {lo}-{hi} Hz": hybrid(band(d, lo, hi)) for d in ("H1", "L1") for lo, hi in BANDS},
    **{f"{d} {lo}-{hi} Hz": hybrid(band(d, lo, hi)) for d in ("H1", "L1") for lo, hi in FINE},
    "<50 Hz, |ratio-1| > 0.3": hybrid(lambda d, f, r: LOW(f) & (np.abs(r - 1) > 0.3)),
    "<50 Hz, |ratio-1| <= 0.3": hybrid(lambda d, f, r: LOW(f) & (np.abs(r - 1) <= 0.3)),
}
EXACT = {(name, model): exact_likelihood(
             RunConfig(label="_", model=model,
                       prior_kwargs=dict(rho_max_gcm3=2e8) if model == "environment" else {}),
             "data_cache/distance_lookup_phase.npz", ifos=ifos, trigger_time=TRIGGER)
         for name, ifos in TARGETS.items() for model in ("vacuum", "environment")}


def exact_lnl(task):
    key, params = task
    return float(EXACT[key].log_likelihood_ratio(params))


def ln_z(label, target, pool):
    rng = np.random.default_rng(zlib.crc32(label.encode()))   # same samples for every target
    result = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json"))
    post = result.posterior
    rows = post.iloc[rng.choice(len(post), N_SAMPLES, replace=False)]
    model = "environment" if "alpha_cloud" in post else "vacuum"
    tasks = [((target, model), {**r.to_dict(), "phase": 0.0}) for _, r in rows.iterrows()]
    delta = np.array(pool.map(exact_lnl, tasks)) - rows["log_likelihood"].to_numpy()
    return result.log_evidence + logsumexp(delta) - np.log(delta.size), delta.std()


if __name__ == "__main__":
    print("BayesWave / Welch PSD, median per band; % of bins below 50 Hz differing by > 30%")
    for w, c in zip(WELCH, CATALOGUE):
        f = w.frequency_array
        ratio = c.power_spectral_density_array / w.power_spectral_density_array
        print(f"  {w.name}  " + "  ".join(f"{lo}-{hi}: {np.median(ratio[(f >= lo) & (f < hi)]):.2f}"
                                          for lo, hi in BANDS)
              + f"   {np.mean(np.abs(ratio[LOW(f)] - 1) > 0.3):.0%}")
    offset = mass_cut_offset()
    print(f"\n{'PSD':26s}{'agnostic':>10s}{'SR 1e6':>10s}   max log-weight scatter")
    with mp.get_context("fork").Pool(NPOOL) as pool:
        for target in TARGETS:
            z = {name: ln_z(label, target, pool) for name, label in RUNS.items()}
            lnb = [z[name][0] - z["vacuum"][0] + offset for name in ("agnostic", "superradiance")]
            print(f"{target:26s}{lnb[0]:+10.2f}{lnb[1]:+10.2f}   "
                  f"{max(s for _, s in z.values()):.2f}", flush=True)
