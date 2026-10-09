"""
Section IV.A -- Data selection and preparation.

Strain is fetched from GWOSC (16 kHz release, resampled to 4096 Hz).  The
analysis segment ends 2 s after the trigger; the PSD is estimated from the
stretch of off-source data *preceding* the segment with a median-Welch
average, so the signal never contaminates its own noise estimate.
"""

from pathlib import Path
import pickle
import time

import numpy as np

# Events, GPS triggers and analysis settings.  GW190728 is the primary
# target; the extended list favours low mass / moderate q / decent SNR,
# where the in-band inspiral is long enough for environmental dephasing.
EVENTS = {
    "GW190728_064510": dict(duration=8),    # primary target (Roy et al.)
    "GW190814":        dict(duration=16),   # flagged by Roy et al.
    "GW190924_021846": dict(duration=16),
    "GW191129_134029": dict(duration=16),
    "GW200202_154313": dict(duration=16),
    "GW200316_215756": dict(duration=16),
}

DETECTORS = ("H1", "L1")
SAMPLE_RATE = 4096
F_LOW, F_HIGH = 20.0, 1024.0
PSD_DURATION = 512                  # seconds of off-source data for Welch
# Tukey rise time [s] before the FFT.  bilby's default, 0.2 s, leaks the
# steep noise below 20 Hz into the lowest analysed bins: in L1, 8-s
# segments then hold 1.3-1.6x the Welch PSD at 20-30 Hz, against 1.04
# with the 0.4 s of bilby_pipe and the catalogue analyses.  ln B barely
# notices (script 27: -1.20 against -1.12), so 0.2 s stays the baseline.
ROLL_OFF = 0.2
GWOSC_TIMEOUT = 600                 # s; 4096-s GWOSC files can be slow to stream

# GWTC-2.1 parameter-estimation releases, whose BayesWave PSDs are the
# LVK's own noise model for each event (psd_source="gwtc21").
CATALOGUE = {"GW190728_064510":
             "IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_cosmo.h5"}


def _strain(detector, start, end, attempts=4, wait=60):
    """GWOSC open data, retried: downloads fail transiently (timeouts,
    truncated files) and a multi-day pipeline should not die of one."""
    from gwpy.timeseries import TimeSeries

    for attempt in range(attempts):
        try:
            return TimeSeries.fetch_open_data(detector, start, end,
                                              sample_rate=SAMPLE_RATE, cache=True,
                                              timeout=GWOSC_TIMEOUT)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(wait)


def catalogue_psd(event, detector, cache_dir="data_cache"):
    """(frequencies, PSD) of one detector from the GWTC-2.1 release."""
    import h5py

    with h5py.File(Path(cache_dir) / CATALOGUE[event], "r") as f:
        return f["C01:IMRPhenomXPHM/psds"][detector][()].T


def fetch_event(event, duration=None, detectors=DETECTORS, psd_fftlength=4,
                psd_source="welch", roll_off=ROLL_OFF, cache_dir="data_cache"):
    """Return (bilby InterferometerList, trigger gps) for a GWOSC event.

    psd_source: "welch" (median Welch average of the preceding 512 s) or
    "gwtc21" (the catalogue's BayesWave PSDs).  roll_off: rise time [s] of
    the Tukey window applied before the FFT.  Downloads are cached to
    disk so re-runs (robustness sweeps) are free.  The cache name carries
    every input: it once omitted the detectors, so the single-detector
    and H1+L1+V1 variants silently reloaded the H1+L1 data.
    """
    import bilby
    from gwosc.datasets import event_gps

    duration = duration or EVENTS[event]["duration"]
    name = f"{event}_{duration}s_fft{psd_fftlength}_{''.join(detectors)}"
    if psd_source != "welch":
        name += f"_{psd_source}"
    if roll_off != ROLL_OFF:
        name += f"_roll{roll_off}"
    cache = Path(cache_dir) / f"{name}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    trigger = event_gps(event)
    end = trigger + 2.0                       # post-trigger padding
    start = end - duration

    ifos = bilby.gw.detector.InterferometerList([])
    for det in detectors:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.minimum_frequency, ifo.maximum_frequency = F_LOW, F_HIGH

        strain = _strain(det, start, end)
        ifo.strain_data.set_from_gwpy_timeseries(strain)
        ifo.strain_data.roll_off = roll_off   # read when the FFT is first taken

        if psd_source == "gwtc21":
            freqs, psd = catalogue_psd(event, det, cache_dir)
        else:
            psd_strain = _strain(det, start - PSD_DURATION, start)
            welch = psd_strain.psd(fftlength=psd_fftlength,
                                   overlap=psd_fftlength / 2, method="median")
            freqs, psd = welch.frequencies.value, welch.value
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=freqs, psd_array=psd)
        ifos.append(ifo)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump((ifos, trigger), f)
    return ifos, trigger


def fetch_off_source_noise(event, offset, duration, detectors=DETECTORS,
                           psd_fftlength=4, roll_off=ROLL_OFF):
    """Interferometers holding pure noise `offset` seconds after the event.

    Used by the injection study (Section IV.F): real detector noise with no
    astrophysical signal, into which simulated vacuum signals are injected.
    """
    import bilby
    from gwosc.datasets import event_gps

    start = event_gps(event) + offset
    ifos = bilby.gw.detector.InterferometerList([])
    for name in detectors:
        ifo = bilby.gw.detector.get_empty_interferometer(name)
        ifo.minimum_frequency, ifo.maximum_frequency = F_LOW, F_HIGH

        strain = _strain(name, start, start + duration)
        ifo.strain_data.set_from_gwpy_timeseries(strain)
        ifo.strain_data.roll_off = roll_off

        psd_strain = _strain(name, start - PSD_DURATION, start)
        psd = psd_strain.psd(fftlength=psd_fftlength,
                             overlap=psd_fftlength / 2, method="median")
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value)
        ifos.append(ifo)
    return ifos, start + duration - 2.0       # nominal "trigger" time
