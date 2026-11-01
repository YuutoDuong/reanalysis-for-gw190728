"""Section V.G -- What the pipeline returns when the environment is real.

The vacuum campaigns (scripts 05, 06) say how often noise alone produces
a given ln B.  They do not say what a real environment of the size
GW190728's posterior suggests would produce: whether the baseline's
ln B ~ -0.6 counts against it or only shows a lack of power, and whether
a real dephasing raises ln B as much as the event's does when Virgo is
added, although Virgo holds 3% of its power.

Inject the environmental waveform (waveform.scalar_cloud_bbh, harmonic
by harmonic), with the cloud at the posterior medians of GW190728, into
noise segments of the redshifted campaign with that campaign's source
parameters: the ten injections whose network signal-to-noise ratio is
nearest the event's 12.7 (11.6-14.0; scripts/checks/background_by_snr.py),
since ln B depends on loudness above all.  Each injection then has a
vacuum twin in the same noise, and is analysed exactly like it.  --sr re-analyses the same ten
under the superradiance prior at tau_d = 1e6 yr.  Script 30 --det --cloud
then gives the exact likelihood, the other networks and the gains in fit.

Expected wall time: ~2 days (--sr: ~1 day more); restartable.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import mass_cut_offset
from scalar_env.injections import run_campaign, run_sr_campaign

ENVIRONMENT = dict(alpha_cloud=0.12, rho_phi=1.9e7)   # GW190728, agnostic posterior medians
INDICES = (1, 2, 3, 5, 8, 16, 22, 24, 42, 43)         # as loud as the event; the glitch segment (28) excluded

parser = argparse.ArgumentParser(description="injections of environmental signals")
parser.add_argument("--sr", action="store_true", help="superradiance prior, tau_d = 1e6 yr")
if parser.parse_args().sr:
    results = run_sr_campaign(tau_d_yr=1e6, environment=ENVIRONMENT,
                              n_high=len(INDICES), n_typical=0)
    lnb = [v["lnb_sr"] for v in results.values()]
else:
    lnb = run_campaign(environment=ENVIRONMENT, indices=INDICES)

lnb = np.array(lnb) + mass_cut_offset()
print(f"\n{lnb.size} environmental injections (alpha = {ENVIRONMENT['alpha_cloud']}, "
      f"rho_phi = {ENVIRONMENT['rho_phi']:.1e} g/cm^3), ln B as sampled: "
      f"median {np.median(lnb):+.2f}, range [{lnb.min():+.2f}, {lnb.max():+.2f}]")
