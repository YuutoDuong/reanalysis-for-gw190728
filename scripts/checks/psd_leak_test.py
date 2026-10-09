"""Is the Welch PSD biased against bilby-windowed 8-s segments OFF source?

Cut the 512 s that the Welch PSD is estimated from into 64 segments of 8 s,
window each as bilby does (Tukey, 0.2 s roll-off), and compare its power
with the Welch PSD band by band: (2/T)|d(f)|^2 / S(f) should average ~1
(0.97 with the window's power loss).  A ratio well above 1 at 20-30 Hz in
every segment means leakage/estimator bias, not something special about
the event segment.
"""
import sys
import numpy as np
from scipy.signal.windows import tukey

sys.path.insert(0, ".")
from scalar_env import data
from gwosc.datasets import event_gps

BANDS = [(20, 25), (25, 30), (30, 50), (50, 100), (100, 300), (300, 1024)]
T, FS = 8, data.SAMPLE_RATE
trigger = event_gps("GW190728_064510")
end = trigger + 2.0
start = end - T
for det in ("H1", "L1", "V1"):
    ts = data._strain(det, start - data.PSD_DURATION, start)
    welch = ts.psd(fftlength=4, overlap=2, method="median")
    x = ts.value
    n = T * FS
    f = np.fft.rfftfreq(n, 1 / FS)
    S = np.interp(f, welch.frequencies.value, welch.value)
    w = tukey(n, alpha=2 * 0.2 / T)
    ratios = []
    for k in range(len(x) // n):
        d = np.fft.rfft(x[k * n:(k + 1) * n] * w) / FS
        p = 2 / T * np.abs(d) ** 2 / S
        ratios.append([(p[(f >= lo) & (f < hi)].mean(), np.median(p[(f >= lo) & (f < hi)]) / np.log(2))
                       for lo, hi in BANDS])
    r = np.array(ratios)   # (segments, bands, 2)
    print(f"{det}: {len(r)} off-source segments; band means (mean/median) "
          + ", ".join(f"{lo}-{hi}" for lo, hi in BANDS))
    print("   average over segments: " + "  ".join(f"{m:.2f}/{md:.2f}" for m, md in r.mean(0)))
    print("   segment-to-segment sd: " + "  ".join(f"{m:.2f}/{md:.2f}" for m, md in r.std(0)))
    # the event segment itself, same window, same PSD (contains the signal)
    ev = data._strain(det, start, end).value
    d = np.fft.rfft(ev * w) / FS
    p = 2 / T * np.abs(d) ** 2 / S
    print("   event segment (with signal): "
          + "  ".join(f"{p[(f >= lo) & (f < hi)].mean():.2f}" for lo, hi in BANDS))
    # alternative: a wider roll-off (0.4 s, bilby_pipe default) on the same segments
    w4 = tukey(n, alpha=2 * 0.4 / T)
    r4 = []
    for k in range(len(x) // n):
        d = np.fft.rfft(x[k * n:(k + 1) * n] * w4) / FS
        p = 2 / T * np.abs(d) ** 2 / S
        r4.append([p[(f >= lo) & (f < hi)].mean() for lo, hi in BANDS])
    print("   roll-off 0.4 s, mean over segments: " + "  ".join(f"{m:.2f}" for m in np.mean(r4, 0)))
