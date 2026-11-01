"""Does Virgo's noise alone give GW190728 its rise with Virgo?

Adding Virgo raises GW190728's ln B by +1.08 (Welch spectra), and with the
catalogue's BayesWave spectra by +2.16, more than for any vacuum injection
in the surrounding data (Sec. V.E).  No coherent signal can supply this:
Virgo carries 2-3% of the environmental term, so whatever does it lies in
Virgo's data.  This measures how often Virgo's noise does it for the
event's own posterior.  It keeps the event's H1+L1 data and H1+L1 runs
(vacuum; the four agnostic runs; superradiance, tau_d = 1e6 yr) and
replaces Virgo's data by off-source Virgo noise, one 8-s segment at a
time, from the 8 h around the event, with a vacuum signal drawn from the
event's posterior added.  Each run is reweighted from H1+L1 to H1+L1+V1 on
every slide,

    Z_HLV = Z_RB E_post[ L_HLV,exact / L_RB ],

on the samples script 17 stored, so zero lag (the event's own Virgo data,
the same estimator) is the point of comparison.  The waveforms do not
depend on the slide, so each sample's detector responses are computed
once, and a slide costs Virgo's inner products and bilby's distance and
phase marginalisation (exact_likelihood's own lookup table).

Slides whose Virgo data fail a GWOSC flag (DATA, CBC_CAT1-3, a hardware
injection) over the segment or its 512-s PSD window are skipped, and each
slide keeps the whitened-strain statistics of script 09, so that glitches
can be set aside.  Spectra: Welch from the preceding 512 s, as for the
event, and the event's BayesWave spectrum (estimated on source, so less
apt far from the event).  Writes virgo_time_slides.json; downloads the
Virgo bulk files it lacks (~120 MB each), then ~15 min on 10 cores.
"""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
import h5py
from gwpy.timeseries import TimeSeries
from scalar_env import data
from scalar_env.inference import RunConfig, exact_likelihood, OUTDIR

bilby.core.utils.logger.setLevel("WARNING")
EVENT = "GW190728_064510"
T0 = data.EVENTS[EVENT]["gps"]
DURATION = 8
HLV = ("H1", "L1", "V1")
LOOKUP = "data_cache/distance_lookup_phase.npz"
RUNS = {  # the event's H1+L1 runs, reweighted on every slide
    "vacuum": ["gw190728_vacuum"],
    "agnostic": ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
                 "gw190728_env_rho_max2e8"],
    "superradiance": ["gw190728_env_sr_eq9_taud1e6_rho2e8"],
}
BLOCKS = [1248321536 + 4096 * k for k in range(7)]   # Virgo bulk files, -2.8 h to +5.2 h
FLAGS = ("DATA", "CBC_CAT1", "CBC_CAT2", "CBC_CAT3")
N_INJECTED = 500        # vacuum signals drawn from the event's posterior
GLITCH = dict(max_whitened=6.0, kurtosis=0.5)   # a slide above either is set aside
PAPER = {"agnostic": {"welch": 1.08, "gwtc21": 2.16},     # the event's quoted rises
         "superradiance": {"welch": 1.43, "gwtc21": 2.73}}
NPOOL = int(os.environ.get("NPOOL", 10))
OUT = Path("virgo_time_slides.json")

LIKE = {}      # (model, spectrum) -> exact likelihood of H1+L1+V1 at the event
SLIDES = {}    # Virgo data of every slide divided by its spectrum, and 1/spectrum
STRAIN = {}    # Virgo strain over BLOCKS, read by the slide workers


# --------------------------------------------------------------- Virgo data
def virgo_strain():
    """Virgo strain over BLOCKS and, per second, whether it passes FLAGS
    with no hardware injection.  Missing bulk files are downloaded once;
    a block Virgo did not record is left empty."""
    values, good = [], []
    for start in BLOCKS:
        try:
            values.append(data._strain("V1", start, start + 4096).value)
        except Exception as exc:
            print(f"V1 {start}: no data ({type(exc).__name__})", flush=True)
            values.append(np.full(4096 * data.SAMPLE_RATE, np.nan))
            good.append(np.zeros(4096, bool))
            continue
        with h5py.File(data._cached_file("V1", start, start + 4096), "r") as f:
            names = [n.decode() for n in f["quality/simple/DQShortnames"][()]]
            mask = f["quality/simple/DQmask"][()]
            no_injection = f["quality/injections/Injmask"][()]
            n_injection_bits = len(f["quality/injections/InjShortnames"][()])
        bits = sum(1 << names.index(flag) for flag in FLAGS)
        good.append(((mask & bits) == bits) & (no_injection == (1 << n_injection_bits) - 1))
    strain = TimeSeries(np.concatenate(values), t0=BLOCKS[0], sample_rate=data.SAMPLE_RATE)
    return strain, np.concatenate(good)


def slide_starts(good):
    """Starts of the 8-s slides: segment and PSD window flagged good, and
    at least 16 s from the event's own segment."""
    starts = np.arange(BLOCKS[0] + data.PSD_DURATION, BLOCKS[-1] + 4096 - DURATION, DURATION)
    return np.array([s for s in starts if abs(s - (T0 - 6)) >= 2 * DURATION
                     and good[s - data.PSD_DURATION - BLOCKS[0]:s + DURATION - BLOCKS[0]].all()])


def virgo_segment(start):
    """(frequency-domain strain, PSD array, whitened statistics) of Virgo's
    8 s from `start`, prepared as data.fetch_off_source_noise does."""
    strain = STRAIN["V1"].crop(start, start + DURATION)
    welch = STRAIN["V1"].crop(start - data.PSD_DURATION, start).psd(
        fftlength=4, overlap=2, method="median")
    ifo = bilby.gw.detector.get_empty_interferometer("V1")
    ifo.minimum_frequency, ifo.maximum_frequency = data.F_LOW, data.F_HIGH
    ifo.strain_data.set_from_gwpy_timeseries(strain)
    ifo.strain_data.roll_off = data.ROLL_OFF
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=welch.frequencies.value, psd_array=welch.value)
    mask = ifo.frequency_mask
    # script 09's statistics: the loudest whitened sample, and the excess
    # kurtosis at 20-100 Hz
    white = strain.whiten(asd=welch ** 0.5).crop(start + 0.5, start + DURATION - 0.5)
    band = white.bandpass(data.F_LOW, 100.0).value
    band = band / band.std()
    stats = (float(np.abs(white.value).max()), float(np.mean(band ** 4) - 3.0))
    return ifo.frequency_domain_strain[mask], ifo.power_spectral_density_array[mask], stats


# ------------------------------------------------------- per-sample terms
def responses(model, params):
    """Each detector's response to one sample, built as
    log_likelihood_ratio builds it."""
    like = LIKE[model, "welch"]
    polarizations = like.waveform_generator.frequency_domain_strain(params)
    params = {**params, **like.get_sky_frame_parameters(params)}
    return params, {ifo.name: like._compute_full_waveform(
        signal_polarizations=polarizations, interferometer=ifo, parameters=params)
        for ifo in like.interferometers}


def injected(params):
    """Virgo's response to a vacuum signal of the event's posterior, with
    the sample's own phase and distance."""
    virgo = LIKE["vacuum", "welch"].interferometers[2]
    return responses("vacuum", params)[1]["V1"][virgo.frequency_mask]


def sample_terms(task):
    """Marginalised log-likelihood ratios of one sample, per spectrum:
    H1+L1, H1+L1+V1 at zero lag, and H1+L1+V1 on every slide."""
    model, params = task
    params, h = responses(model, {**params, "phase": 0.0})
    norm = 4 / DURATION
    out = {}
    for spectrum in ("welch", "gwtc21"):
        like = LIKE[model, spectrum]
        ifos = {ifo.name: ifo for ifo in like.interferometers}
        d_hl = sum(ifos[n].inner_product(h[n]) for n in ("H1", "L1"))
        o_hl = sum(ifos[n].optimal_snr_squared(h[n]).real for n in ("H1", "L1"))
        d_v0 = ifos["V1"].inner_product(h["V1"])
        o_v0 = ifos["V1"].optimal_snr_squared(h["V1"]).real
        v = h["V1"][ifos["V1"].frequency_mask]
        d_v = norm * (SLIDES[spectrum, "data"] @ np.conj(v))
        o_v = norm * (SLIDES[spectrum, "inv_psd"] @ np.abs(v) ** 2)

        def lnl(d, o):
            return like.distance_marginalized_likelihood(d_inner_h=d, h_inner_h=o,
                                                         parameters=params)
        out[spectrum] = (float(lnl(d_hl, o_hl)), float(lnl(d_hl + d_v0, o_hl + o_v0)),
                         lnl(d_hl + d_v, o_hl + o_v))
    return out


# ------------------------------------------------------------- reweighting
def run_terms(label, pool):
    """One run's log(L / L_RB) at its stored samples: H1+L1 (Welch), and
    per spectrum H1+L1+V1 at zero lag and on every slide.  Checks the
    first against script 17, and zero lag against log_likelihood_ratio."""
    rows, ln_w = np.load("reweighted_samples.npz")[label].T
    post = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json")).posterior
    post = post.iloc[rows.astype(int)]
    model = "environment" if "alpha_cloud" in post else "vacuum"
    tasks = [(model, r.to_dict()) for _, r in post.iterrows()]
    terms = pool.map(sample_terms, tasks)
    direct = max(abs(LIKE[model, s].log_likelihood_ratio({**tasks[i][1], "phase": 0.0})
                     - terms[i][s][1]) for i in range(3) for s in ("welch", "gwtc21"))
    ln_rb = post["log_likelihood"].to_numpy()
    out = {"hl": np.array([t["welch"][0] for t in terms]) - ln_rb}
    out["check"] = (float(np.abs(out["hl"] - ln_w).max()), float(direct))
    for spectrum in ("welch", "gwtc21"):
        out[spectrum] = (np.array([t[spectrum][1] for t in terms]) - ln_rb,
                         np.array([t[spectrum][2] for t in terms]).T - ln_rb)
    return out


def ln_ratio(log_w):
    """ln E[L_new / L_RB] over the samples (last axis), and the effective
    sample size of the weights."""
    log_w = np.atleast_2d(log_w)
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    return (logsumexp(log_w, axis=1) - np.log(log_w.shape[1]),
            w.sum(axis=1) ** 2 / (w ** 2).sum(axis=1))


def shifts(terms, model, spectrum):
    """Shift of ln B on adding Virgo at zero lag and on every slide (the
    environmental part averaged over the model's runs), with the smallest
    effective sample size behind each."""
    parts, ess_zero, ess_slides = {}, [], []
    for name in (model, "vacuum"):
        zero, slides = [], []
        for label in RUNS[name]:
            t = terms[label]
            base = ln_ratio(t["hl"])[0][0]
            z0, e0 = ln_ratio(t[spectrum][0])
            zs, es = ln_ratio(t[spectrum][1])
            zero.append(z0[0] - base)
            slides.append(zs - base)
            ess_zero.append(e0[0])
            ess_slides.append(es)
        parts[name] = (np.mean(zero), np.mean(slides, axis=0))
    return (float(parts[model][0] - parts["vacuum"][0]), parts[model][1] - parts["vacuum"][1],
            float(min(ess_zero)), np.min(ess_slides, axis=0))


def tail(values, observed):
    """k of N at or above `observed`, with the 95% Clopper-Pearson interval."""
    k, n = int(np.sum(values >= observed)), values.size
    low = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    high = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return k, n, float(low), float(high)


if __name__ == "__main__":
    for model in ("vacuum", "environment"):
        for spectrum in ("welch", "gwtc21"):
            config = RunConfig(label="_", model=model, detectors=HLV, psd_source=spectrum,
                               prior_kwargs=dict(rho_max_gcm3=2e8) if model == "environment" else {})
            LIKE[model, spectrum] = exact_likelihood(config, LOOKUP)
    virgo_event = {s: LIKE["vacuum", s].interferometers[2] for s in ("welch", "gwtc21")}
    mask = virgo_event["welch"].frequency_mask

    STRAIN["V1"], good = virgo_strain()
    starts = slide_starts(good)
    print(f"{starts.size} slides in {len(BLOCKS) * 4096 / 3600:.1f} h of Virgo data "
          f"({good.mean():.0%} of it flagged good)", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(NPOOL) as pool:
        segments = pool.map(virgo_segment, [T0 - 6] + list(starts))
    event_rebuilt, segments = segments[0], segments[1:]
    difference = (np.abs(event_rebuilt[0] - virgo_event["welch"].frequency_domain_strain[mask]).max()
                  / np.abs(event_rebuilt[0]).max())
    print(f"event's Virgo segment rebuilt from the bulk files: largest relative difference "
          f"{difference:.1e}; its whitened statistics {event_rebuilt[2][0]:.2f}, "
          f"{event_rebuilt[2][1]:+.2f}", flush=True)
    stats = np.array([s[2] for s in segments])

    # every slide gets a vacuum signal drawn from the event's posterior
    rng = np.random.default_rng(20261027)
    rows, ln_w = np.load("reweighted_samples.npz")["gw190728_vacuum"].T
    post = bilby.result.read_in_result(
        str(OUTDIR / "gw190728_vacuum" / "gw190728_vacuum_result.json")).posterior
    weight = np.exp(ln_w - ln_w.max())
    draws = post.iloc[rows[rng.choice(rows.size, N_INJECTED, p=weight / weight.sum())].astype(int)]
    with ctx.Pool(NPOOL) as pool:
        signals = np.array(pool.map(injected, [r.to_dict() for _, r in draws.iterrows()]))
    which = rng.integers(N_INJECTED, size=starts.size)
    strain = np.array([s[0] for s in segments]) + signals[which]
    psd = np.array([s[1] for s in segments])
    bayeswave = virgo_event["gwtc21"].power_spectral_density_array[mask]
    SLIDES["welch", "data"], SLIDES["welch", "inv_psd"] = strain / psd, 1 / psd
    SLIDES["gwtc21", "data"], SLIDES["gwtc21", "inv_psd"] = strain / bayeswave, 1 / bayeswave
    del strain, psd, segments

    terms = {}
    with ctx.Pool(NPOOL) as pool:
        for label in sum(RUNS.values(), []):
            terms[label] = run_terms(label, pool)
            print(f"{label}: H1+L1 against script 17, largest difference "
                  f"{terms[label]['check'][0]:.1e}; zero lag against log_likelihood_ratio "
                  f"{terms[label]['check'][1]:.1e}", flush=True)

    glitch = (stats[:, 0] > GLITCH["max_whitened"]) | (np.abs(stats[:, 1]) > GLITCH["kurtosis"])
    print(f"\n{starts.size} slides, {glitch.sum()} set aside as glitches "
          f"(whitened sample above {GLITCH['max_whitened']} or |kurtosis| above {GLITCH['kurtosis']})")
    results = dict(starts=starts.tolist(), injected=which.tolist(),
                   max_whitened=stats[:, 0].round(3).tolist(), kurtosis=stats[:, 1].round(3).tolist(),
                   checks={label: t["check"] for label, t in terms.items()}, shifts={})
    for model in ("agnostic", "superradiance"):
        for spectrum in ("welch", "gwtc21"):
            zero, slide, ess_zero, ess = shifts(terms, model, spectrum)
            results["shifts"][f"{model}, {spectrum}"] = dict(
                zero_lag=zero, ess_zero_lag=ess_zero, slides=slide.round(4).tolist(),
                ess=ess.round(3).tolist())
            usable = (ess >= 30) & ~glitch
            x = slide[usable]
            name = "Virgo" if spectrum == "welch" else "Virgo + BayesWave"
            print(f"{model:13s} {name:17s} zero lag {zero:+.2f} (quoted {PAPER[model][spectrum]:+.2f}, "
                  f"ESS {ess_zero:.0f}) | {x.size} slides: median {np.median(x):+.2f}, "
                  f"sd {x.std(ddof=1):.2f}, 90% in [{np.quantile(x, 0.05):+.2f}, "
                  f"{np.quantile(x, 0.95):+.2f}], max {x.max():+.2f}")
            for title, values in (("all slides", slide), ("ESS >= 30, no glitch", x)):
                for threshold, what in ((zero, "zero lag"), (PAPER[model][spectrum], "quoted")):
                    k, n, low, high = tail(values, threshold)
                    print(f"    {title:21s} at or above {what:8s} ({threshold:+.2f}): "
                          f"{k} of {n}, p = {k / n:.4f} [{low:.4f}, {high:.4f}]")
    OUT.write_text(json.dumps(results))
    print(f"\nwrote {OUT}")
