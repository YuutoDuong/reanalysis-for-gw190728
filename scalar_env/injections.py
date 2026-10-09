"""
Sections IV.F and V.F -- Injection-recovery study (false-alarm characterisation).

Vacuum-GR signals with GW190728-like parameters are injected into real
off-source detector noise around the event; the full environmental vs
vacuum model comparison is then run on each.  The distribution of ln B
under the vacuum hypothesis calibrates the significance of the observed
value: the false-alarm probability is the fraction of vacuum injections
with ln B >= ln B(GW190728).
"""

from pathlib import Path
import json
import os
import time

import numpy as np

from . import data
from .inference import RunConfig, run_pe, ln_bayes_factor, mass_cut_offset

# Must match RHO_MAX in scripts 02 and 04: a false-alarm probability is
# only meaningful if the background is computed with the same statistic
# as the observation, and ln B carries an explicit -ln(rho_max) term.
RHO_MAX = 2e8

NPOOL = max(1, (os.cpu_count() or 2) - 2)

# GW190728-like fiducial source; components jittered per injection so the
# study samples a neighbourhood, not one point.  The masses are GWTC-2.1's
# *source-frame* medians.
FIDUCIAL_INJECTION = dict(
    mass_1=12.5, mass_2=8.0, a_1=0.25, a_2=0.05, tilt_1=0.4, tilt_2=1.0,
    phi_12=1.0, phi_jl=2.0, luminosity_distance=870.0, theta_jn=0.9,
    psi=1.6, phase=2.0, ra=5.0, dec=0.3)

JITTER = dict(mass_1=0.8, mass_2=0.6, a_1=0.1, luminosity_distance=150.0,
              theta_jn=0.3)


def _tag(detector_frame):
    """Suffix of the files and run labels of a campaign.  The first
    campaign (2026-08) injected the source-frame masses unredshifted, so
    its binaries were 1+z lighter than GW190728 in the detector frame; it
    carries no tag.  The second (2026-10) redshifts them."""
    return "_det" if detector_frame else ""


def draw_injection_parameters(n, seed=1234, detector_frame=True):
    """n vacuum parameter sets clustered around the fiducial source.

    The draws do not depend on detector_frame, so both campaigns share
    every source parameter and noise segment; detector_frame only
    multiplies the masses by 1 + z(d_L) of each injection's own distance.
    """
    from bilby.gw.conversion import luminosity_distance_to_redshift

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        p = dict(FIDUCIAL_INJECTION)
        for key, width in JITTER.items():
            p[key] = float(np.clip(p[key] + rng.normal(0, width),
                                   1e-3, None))
        p["mass_1"], p["mass_2"] = max(p["mass_1"], p["mass_2"]), \
                                   min(p["mass_1"], p["mass_2"])
        if detector_frame:
            z = float(luminosity_distance_to_redshift(p["luminosity_distance"]))
            p["mass_1"], p["mass_2"] = p["mass_1"] * (1 + z), p["mass_2"] * (1 + z)
        draws.append(p)
    return draws


def inject_into_noise(parameters, index, duration=8,
                      event="GW190728_064510", approximant="IMRPhenomXPHM",
                      detectors=data.DETECTORS):
    """Inject a vacuum signal into real noise; returns (ifos, trigger).

    Noise segments are taken at increasing offsets after the event so each
    injection sees an independent stretch of real (non-Gaussian,
    non-stationary) detector noise.  Adding a detector leaves the others'
    data unchanged.
    """
    import bilby

    offset = 384 + index * (duration + 8)     # clear of the real signal
    ifos, trigger = data.fetch_off_source_noise(event, offset, duration, detectors)

    # Plain full-grid source model: the PE runs use the relative-binning
    # source, which requires bin-edge arguments only the likelihood
    # supplies -- injection must use the ordinary generator instead.
    wfg = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=data.SAMPLE_RATE,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant=approximant,
                                reference_frequency=data.F_LOW,
                                minimum_frequency=data.F_LOW))
    injection = dict(parameters, geocent_time=trigger)
    ifos.inject_signal(waveform_generator=wfg, parameters=injection)
    return ifos, trigger, injection


def _fetch_with_retries(params, index, duration, attempts=4, wait=90):
    """inject_into_noise with retries: GWOSC hiccups should cost minutes,
    not the campaign."""
    for attempt in range(attempts):
        try:
            return inject_into_noise(params, index, duration)
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            print(f"  data fetch failed ({type(exc).__name__}: {exc}); "
                  f"retry {attempt + 1}/{attempts - 1} in {wait} s")
            time.sleep(wait)


def run_campaign(n_injections=50, duration=8, detector_frame=True):
    """Full campaign: for each injection run vacuum + environmental PE.

    Restartable: finished injections are read back from the results file
    (injection_lnb{tag}.json).  Returns the list of ln B values.
    """
    tag = _tag(detector_frame)
    path = Path(f"injection_lnb{tag}.json")
    done = json.loads(path.read_text()) if path.exists() else {}

    for i, params in enumerate(draw_injection_parameters(
            n_injections, detector_frame=detector_frame)):
        key = f"inj{i:03d}"
        if key in done:
            continue

        # GWOSC downloads over a multi-week campaign will hit transient
        # network timeouts; retry with a pause, and skip the injection
        # rather than kill the campaign if the outage persists.
        try:
            ifos, trigger, injection = _fetch_with_retries(params, i, duration)
        except Exception as exc:
            print(f"{key}: SKIPPED -- GWOSC fetch kept failing ({exc}); "
                  f"rerun the script later to fill this one in.")
            continue
        vac = run_pe(RunConfig(label=f"{key}{tag}_vac", model="vacuum", npool=NPOOL),
                     ifos=ifos, trigger_time=trigger, injection=injection)
        # A vacuum injection is the rho_phi = 0 point of the environmental
        # model; bilby's post-processing evaluates the env waveform at the
        # injection parameters, so the env-model keys must be present.
        env_injection = dict(injection, alpha_cloud=0.1, rho_phi=0.0)
        env = run_pe(RunConfig(label=f"{key}{tag}_env", model="environment",
                               npool=NPOOL,
                               prior_kwargs=dict(rho_max_gcm3=RHO_MAX)),
                     ifos=ifos, trigger_time=trigger, injection=env_injection)

        lnb, err = ln_bayes_factor(env, vac)
        done[key] = dict(lnb=lnb, err=err, parameters=params)
        path.write_text(json.dumps(done, indent=2))
        print(f"{key}: ln B = {lnb:+.2f} +/- {err:.2f}")

    return [v["lnb"] for v in done.values()]


# ------------------------------------------------ superradiance-prior subset
def select_sr_subset(agnostic_file="injection_lnb.json", n_high=3,
                     n_typical=10):
    """Choose injections for the superradiance-prior follow-up.

    The agnostic background calibrates the agnostic statistic, but Roy et
    al.'s headline ln B ~ 3.5 uses the superradiance prior.  Rather than a
    random subsample (N ~ 13 cannot measure a few-percent rate), we take
    the highest agnostic ln B draws plus a stratified sample of the rest,
    which answers the mechanistic question: does the SR prior amplify or
    suppress noise fluctuations that already look environmental?

    Returns [(key, record), ...] ordered high-to-low in agnostic ln B.
    """
    data = json.loads(Path(agnostic_file).read_text())
    ranked = sorted(data.items(), key=lambda kv: kv[1]["lnb"], reverse=True)
    high, rest = ranked[:n_high], ranked[n_high:]
    step = max(1, len(rest) // n_typical) if n_typical else 1
    return high + rest[::step][:n_typical]


def run_sr_campaign(tau_d_yr=1e6, duration=8, detector_frame=True,
                    **subset_kwargs):
    """Re-analyse selected vacuum injections under Roy et al.'s
    superradiance priors (their SM Eqs. 9-10) at fixed tau_d.

    The cached vacuum runs are reused, so only the environmental side is
    resampled.  Restartable like run_campaign, and returns only the
    selected subset, never stray entries left in the results file.
    """
    tag = _tag(detector_frame)
    path = Path(f"injection_sr_lnb{tag}.json")
    done = json.loads(path.read_text()) if path.exists() else {}
    subset = select_sr_subset(f"injection_lnb{tag}.json", **subset_kwargs)

    for key, record in subset:
        # Reuse a cached entry only if it was computed against the current
        # agnostic background.  A leftover from an earlier campaign carries
        # a different lnb_agnostic: exactly how 5 of the 13 were silently
        # skipped on 2026-09-27 (linearised-waveform, pre-fix SR entries).
        if done.get(key, {}).get("lnb_agnostic") == record["lnb"]:
            continue
        index, params = int(key[3:]), record["parameters"]

        try:
            ifos, trigger, injection = _fetch_with_retries(
                params, index, duration)
        except Exception as exc:
            print(f"{key}: SKIPPED -- GWOSC fetch kept failing ({exc})")
            continue

        vac = run_pe(RunConfig(label=f"{key}{tag}_vac", model="vacuum", npool=NPOOL),
                     ifos=ifos, trigger_time=trigger, injection=injection)
        env_injection = dict(injection, alpha_cloud=0.1, rho_phi=0.0)
        env = run_pe(RunConfig(label=f"{key}{tag}_env_sr", model="environment",
                               npool=NPOOL,
                               prior_kwargs=dict(
                                   superradiance_tau_d_yr=tau_d_yr,
                                   rho_max_gcm3=RHO_MAX)),
                     ifos=ifos, trigger_time=trigger, injection=env_injection)

        lnb, err = ln_bayes_factor(env, vac)
        done[key] = dict(lnb_sr=lnb, err=err, lnb_agnostic=record["lnb"],
                         parameters=params)
        path.write_text(json.dumps(done, indent=2))
        print(f"{key}: ln B (SR) = {lnb:+.2f} +/- {err:.2f}   "
              f"[agnostic: {record['lnb']:+.2f}]")

    return {key: done[key] for key, _ in subset if key in done}


def plot_sr_comparison(sr_results, observed_sr, name="injection_sr_comparison",
                       roy_lnb=3.5):
    """Agnostic vs superradiance-prior ln B for the same noise draws.

    Points below the diagonal mean the SR prior suppresses noise
    fluctuations; points above mean it amplifies them.
    """
    import matplotlib.pyplot as plt

    agn = np.array([v["lnb_agnostic"] for v in sr_results.values()])
    sr = np.array([v["lnb_sr"] for v in sr_results.values()])
    err = np.array([v["err"] for v in sr_results.values()])
    finite = np.abs(agn) < 100          # keep the glitch case off-scale

    fig, ax = plt.subplots(figsize=(5, 4.2))
    lim = (min(agn[finite].min(), sr[finite].min()) - 0.7,
           max(agn[finite].max(), sr[finite].max(), roy_lnb) + 0.7)
    ax.plot(lim, lim, ls="-", lw=0.8, color="grey", label="no change")
    ax.axhline(roy_lnb, ls=":", color="C1",
               label=f"Roy et al. (ln B = {roy_lnb})")
    ax.axhline(observed_sr, ls="--", color="C3",
               label=f"GW190728, SR prior ({observed_sr:+.2f})")
    ax.errorbar(agn[finite], sr[finite], yerr=err[finite], fmt="o",
                color="C0", capsize=3, label="vacuum injections")
    n_off = int((~finite).sum())
    if n_off:
        ax.annotate(f"+{n_off} off-scale (glitch segment)",
                    xy=(0.97, 0.04), xycoords="axes fraction",
                    ha="right", fontsize=8, color="C0")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(r"$\ln B$  (agnostic priors)")
    ax.set_ylabel(r"$\ln B$  (superradiance priors)")
    ax.legend(fontsize=8, loc="upper left")

    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    return fig


def background(tag=""):
    """{key: (agnostic ln B, superradiance ln B or None)} of a campaign as
    the paper quotes them: for the full-grid likelihood (H1+L1 entries of
    script 30's injection_virgo file) once those cover the campaign, else
    relative binning plus the mass-cut offset."""
    campaign = json.loads(Path(f"injection_lnb{tag}.json").read_text())
    exact = Path(f"injection_virgo{tag}.json")
    exact = json.loads(exact.read_text()) if exact.exists() else {}
    if set(exact) >= set(campaign):
        return {k: (v["H1+L1"]["agnostic"]["lnb"],
                    v["H1+L1"].get("superradiance", {}).get("lnb")) for k, v in exact.items()}
    offset = mass_cut_offset()
    sr = Path(f"injection_sr_lnb{tag}.json")
    sr = json.loads(sr.read_text()) if sr.exists() else {}
    return {k: (v["lnb"] + offset, sr[k]["lnb_sr"] + offset if k in sr else None)
            for k, v in campaign.items()}


def false_alarm_probability(lnb_values, threshold):
    """P(ln B >= threshold | vacuum) with a binomial standard error."""
    lnb = np.asarray(lnb_values)
    if lnb.size == 0:
        return np.nan, np.nan
    p = np.count_nonzero(lnb >= threshold) / lnb.size
    return p, np.sqrt(p * (1.0 - p) / lnb.size)


def plot_background(lnb_values, observed, name="injection_background",
                    x_range=(-5.0, 6.0), roy_lnb=3.5):
    """Background ln B distribution vs the observed GW190728 value.

    Zoomed to the bulk of the distribution; extreme outliers (e.g. the
    glitch-candidate segment) are counted in an annotation rather than
    allowed to flatten the histogram.
    """
    import matplotlib.pyplot as plt

    lnb = np.asarray(lnb_values)
    in_range = lnb[(lnb >= x_range[0]) & (lnb <= x_range[1])]
    n_above = int(np.sum(lnb > x_range[1]))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(in_range, bins=np.linspace(*x_range, 23), color="C0",
            alpha=0.7, label=f"vacuum injections (N={lnb.size})")
    ax.axvline(observed, color="C3", ls="--",
               label=f"GW190728 (ln B = {observed:.2f})")
    ax.axvline(roy_lnb, color="C1", ls=":",
               label=f"Roy et al. (ln B = {roy_lnb})")
    if n_above:
        ax.annotate(f"+{n_above} off-scale > {x_range[1]:g}\n"
                    "(glitch-candidate segment)",
                    xy=(0.97, 0.55), xycoords="axes fraction",
                    ha="right", fontsize=8, color="C0")
    ax.set_xlabel(r"$\ln B^{\rm env}_{\rm vac}$")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    return fig
