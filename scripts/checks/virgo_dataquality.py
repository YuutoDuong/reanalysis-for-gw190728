"""Is the strain of any detector unusual at GW190728?

The whitened-strain statistics that single out the glitch segment of
injection 028 (script 09), for H1, L1 and Virgo at the event and in the
50 injection segments: the largest whitened sample, the maximum and
excess kurtosis of the band-limited strain at 20-100 Hz and at 20-40 Hz,
where the environmental dephasing lives, and the peak Q-transform
energy.  Prints the event's value, its rank among the segments and their
median.  Needs GWOSC data (cached after the first run); ~15 min.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from scalar_env import data

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries

T0 = event_gps("GW190728_064510")
DURATION = 8
DETECTORS = ("H1", "L1", "V1")


def fetch(ifo, start, end):
    return TimeSeries.fetch_open_data(ifo, start, end, sample_rate=data.SAMPLE_RATE, cache=True)


def statistics(ifo, start):
    """(max |whitened|, 20-100 Hz max, 20-100 Hz kurtosis, 20-40 Hz max, 20-40 Hz kurtosis, peak Q)."""
    psd = fetch(ifo, start - data.PSD_DURATION, start).psd(fftlength=4, overlap=2, method="median")
    w = fetch(ifo, start, start + DURATION).whiten(asd=psd ** 0.5).crop(start + 0.5, start + DURATION - 0.5)
    out = [abs(w.value).max()]
    for lo, hi in ((20.0, 100.0), (20.0, 40.0)):
        x = w.bandpass(lo, hi).value
        x = x / x.std()
        out += [abs(x).max(), float(np.mean(x ** 4) - 3.0)]
    q = fetch(ifo, start - 2, start + DURATION + 2).q_transform(
        outseg=(start, start + DURATION), frange=(20, 512), qrange=(4, 64), whiten=True)
    return out + [float(q.max().value)]


NAMES = ("|w|max", "max 20-100", "kurt 20-100", "max 20-40", "kurt 20-40", "peak Q")
background = {d: [] for d in DETECTORS}
for i in range(50):
    start = T0 + 384 + 16 * i
    for d in DETECTORS:
        try:
            background[d].append(statistics(d, start))
        except Exception as exc:
            print(f"inj{i:03d} {d}: skipped ({type(exc).__name__})", flush=True)
print(f"{'':4s}" + "".join(f"{n:>14s}" for n in NAMES) + "   (event value; rank among the 50 segments, 1 = largest; their median)")
for d in DETECTORS:
    event = statistics(d, T0 - 6.0)
    bg = np.array(background[d])
    cells = [f"{v:6.2f} #{int((bg[:, j] > v).sum()) + 1:<2d} {np.median(bg[:, j]):5.2f}" for j, v in enumerate(event)]
    print(f"{d:4s}" + "".join(f"{c:>17s}" for c in cells) + f"   ({len(bg)} segments)")
