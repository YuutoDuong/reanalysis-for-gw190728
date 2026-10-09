"""What is in the inj028 noise segment?

inj028 returns ln B ~ +295 for a pure vacuum injection, yet its segment
passes every GWOSC data-quality flag (DATA, CBC_CAT1/2/3, BURST_CAT2,
no hardware injection) in both detectors, as does its PSD window.  This
script characterises the segment directly: it whitens each injection's
8 s window with that window's own PSD and records the loudest excursion
and the excess power below 100 Hz, where the environmental dephasing
lives.  Produces a rank table and a Q-transform figure for the outlier.

Single-core and cache-only (the segments were downloaded by script 05),
so it is safe to run alongside the campaign.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env import data

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries

EVENT = "GW190728_064510"
T0 = event_gps(EVENT)
DURATION = 8
N_INJ = int(sys.argv[1]) if len(sys.argv) > 1 else 31
OUTLIER = 28
DETECTORS = ("H1", "L1")


def segment_start(index):
    return T0 + 384 + index * (DURATION + 8)


def whitened(ifo, start):
    """8 s of whitened strain, using the same 512 s PSD the PE run used."""
    strain = TimeSeries.fetch_open_data(ifo, start, start + DURATION,
                                        sample_rate=data.SAMPLE_RATE,
                                        cache=True)
    psd_strain = TimeSeries.fetch_open_data(ifo, start - data.PSD_DURATION,
                                            start,
                                            sample_rate=data.SAMPLE_RATE,
                                            cache=True)
    psd = psd_strain.psd(fftlength=4, overlap=2, method="median")
    return strain.whiten(asd=psd ** 0.5).crop(start + 0.5,
                                              start + DURATION - 0.5)


def statistics(w):
    """Loudest excursion, and band-limited excess power below 100 Hz."""
    band = w.bandpass(data.F_LOW, 100.0)
    x = band.value / band.value.std()
    return abs(w.value).max(), abs(x).max(), float(np.mean(x**4) - 3.0)


rows = []
for i in range(N_INJ):
    start = segment_start(i)
    try:
        stats = [statistics(whitened(d, start)) for d in DETECTORS]
    except Exception as exc:                 # not yet downloaded
        print(f"inj{i:03d}: skipped ({type(exc).__name__}: {exc})")
        continue
    rows.append((i, *[s for st in stats for s in st]))

cols = ("|w|max", "band", "kurt")
head = "  ".join(f"{d}_{c:<6}" for d in DETECTORS for c in cols)
print(f"\n{'inj':<6} {head}")
for r in rows:
    mark = "   <== ln B = +295" if r[0] == OUTLIER else ""
    print(f"inj{r[0]:03d}  " +
          "  ".join(f"{v:9.2f}" for v in r[1:]) + mark)

idx = [r[0] for r in rows]
if OUTLIER in idx:                      # skipped when only the figure is wanted
    arr = np.array([r[1:] for r in rows])
    print(f"\nrank of inj{OUTLIER:03d} among {len(rows)} segments "
          "(1 = loudest):")
    for j, name in enumerate(f"{d}_{c}" for d in DETECTORS for c in cols):
        order = np.argsort(-arr[:, j])
        rank = int(np.where(np.array(idx)[order] == OUTLIER)[0][0]) + 1
        print(f"   {name:<10} rank {rank:>3}   "
              f"value {arr[idx.index(OUTLIER), j]:8.2f}"
              f"   median {np.median(arr[:, j]):8.2f}")

# The same statistics for the real GW190728 segment (8 s ending 2 s after
# the trigger, as analysed in Sec. V.A).  If the environmental model is
# sensitive to unmodelled transients, the on-source data must be checked
# by the same standard, not only by the DQ flags.
onsource = T0 - 6.0
print("\nGW190728 on-source segment, same statistics:")
for d in DETECTORS:
    s = statistics(whitened(d, onsource))
    print(f"   {d}: |w|max = {s[0]:6.2f}   band = {s[1]:5.2f}   "
          f"kurt = {s[2]:7.2f}")

# Q-transform of the outlier segment.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

start = segment_start(OUTLIER)
fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
peaks = {}
for ax, ifo in zip(axes, DETECTORS):
    ts = TimeSeries.fetch_open_data(ifo, start - 2, start + DURATION + 2,
                                    sample_rate=data.SAMPLE_RATE, cache=True)
    q = ts.q_transform(outseg=(start, start + DURATION), frange=(20, 512),
                       qrange=(4, 64), whiten=True)
    peaks[ifo] = float(q.max().value)
    # A common colour scale: the contrast between the detectors is the
    # point, so the H1 transient must not be rescaled away.
    m = ax.imshow(q, vmin=0, vmax=50)
    ax.set_yscale("log")
    ax.set_ylabel(f"{ifo} frequency [Hz]")
    ticks = np.arange(0, DURATION + 1, 1.0)
    ax.set_xticks(start + ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks])
    ax.set_xlim(start, start + DURATION)
    ax.text(0.985, 0.88, f"peak {peaks[ifo]:.0f}", color="w", ha="right",
            transform=ax.transAxes)
    plt.colorbar(m, ax=ax, label="normalised energy", extend="max")
axes[-1].set_xlabel(f"time from GPS {start:.0f} [s]")
axes[0].set_title(f"inj{OUTLIER:03d}: vacuum injection, all DQ flags clean, "
                  r"$\ln\mathcal{B}=+295$")
Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/inj{OUTLIER:03d}_qscan.png", dpi=160,
            bbox_inches="tight")
print(f"\nfigures/inj{OUTLIER:03d}_qscan.png written.")
