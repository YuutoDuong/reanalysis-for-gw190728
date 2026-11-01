"""What the dominant-mode approximation costs in ln B.

The runs sample a model that gives every harmonic the dominant one's
environmental factor (waveform.scalar_cloud_bbh_dominant_mode); the
quoted evidences are for the correction harmonic by harmonic
(waveform.scalar_cloud_bbh).  Both on the full frequency grid and at the
same posterior samples, print the difference between the two evidences,

    ln Z_harmonics - ln Z_dominant = ln <L_harmonics / L_fast> - ln <L_dominant / L_fast>,

which is also the difference in ln B, the vacuum model being untouched.
First for GW190728, then for the vacuum injections of the superradiance
subset, whose background should not move.  ~10 min on 3 cores.  --det:
only the second campaign's subset (its injections as loud as the event),
where a posterior displaced from zero density can move as much as the
event's.
"""
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, ".")
import bilby
from scalar_env.inference import RunConfig, exact_likelihood, OUTDIR
from scalar_env.injections import inject_into_noise, RHO_MAX
from scalar_env.waveform import scalar_cloud_bbh_dominant_mode

bilby.core.utils.logger.setLevel("WARNING")
N_SAMPLES = 1000
NPOOL = int(os.environ.get("NPOOL", 3))
HLV = ("H1", "L1", "V1")
CATALOGUE = dict(detectors=HLV, psd_source="gwtc21")
RUNS = {  # GW190728 run: how its likelihood differs from the baseline
    "gw190728_env": {},
    "gw190728_env_sr_eq9_taud1e6_rho2e8": {},
    "gw190728_env_hlv_gwtc21psd": CATALOGUE,
    "gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8": CATALOGUE,
    "gw190728_env_hlv_gwtc21psd_phase_sampled": dict(CATALOGUE, phase_marginalization=False),
    "gw190728_env_hlv_gwtc21psd_phase_sampled_sr_eq9_taud1e6_rho2e8":
        dict(CATALOGUE, phase_marginalization=False),
}
GLITCH = "inj028"
LIKELIHOODS = {}   # of the data being processed, inherited by the forked pool


def build(changes, **data):
    """Both exact likelihoods of one data set."""
    config = RunConfig(label="_", model="environment", prior_kwargs=dict(rho_max_gcm3=RHO_MAX), **changes)
    lookup = f"data_cache/distance_lookup_{'phase' if config.phase_marginalization else 'nophase'}.npz"
    LIKELIHOODS["harmonics"] = exact_likelihood(config, lookup, **data)
    LIKELIHOODS["dominant"] = exact_likelihood(config, lookup, **data)
    LIKELIHOODS["dominant"].waveform_generator.frequency_domain_source_model = scalar_cloud_bbh_dominant_mode


def lnl(task):
    name, params = task
    return float(LIKELIHOODS[name].log_likelihood_ratio(params))


def shift(label):
    """(ln Z_harmonics - ln Z_dominant, bootstrap error, effective sample
    size of the harmonics weights) of one run."""
    rng = np.random.default_rng(zlib.crc32(label.encode()))        # script 17's samples
    post = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json")).posterior
    rows = post.iloc[rng.choice(len(post), min(N_SAMPLES, len(post)), replace=False)]
    fixed = dict(phase=0.0) if LIKELIHOODS["harmonics"].phase_marginalization else {}
    params = [{**r.to_dict(), **fixed} for _, r in rows.iterrows()]
    with mp.get_context("fork").Pool(NPOOL) as pool:
        delta = {name: np.array(pool.map(lnl, [(name, p) for p in params]))
                 - rows["log_likelihood"].to_numpy() for name in LIKELIHOODS}
    difference = lambda i: logsumexp(delta["harmonics"][i]) - logsumexp(delta["dominant"][i])
    n = len(rows)
    boot = [difference(rng.integers(0, n, n)) for _ in range(500)]
    w = np.exp(delta["harmonics"] - delta["harmonics"].max())
    return difference(slice(None)), np.std(boot), w.sum() ** 2 / (w ** 2).sum()


if __name__ == "__main__":
    tag = "_det" if "--det" in sys.argv[1:] else ""
    if not tag:
        print("GW190728: ln B, harmonic by harmonic minus dominant mode")
        previous = None
        for label, changes in RUNS.items():
            if changes != previous:
                build(changes)
                previous = changes
            value, error, ess = shift(label)
            print(f"  {label:66s} {value:+.2f} +- {error:.2f}   (ESS {ess:.0f})", flush=True)

    print(f"\nvacuum injections of the {'second' if tag else 'first'} campaign's superradiance subset (H1+L1, Welch)")
    campaign = json.loads(Path(f"injection_lnb{tag}.json").read_text())
    shifts = {"agnostic": [], "superradiance": []}
    for key in sorted(json.loads(Path(f"injection_sr_lnb{tag}.json").read_text())):
        ifos, trigger, _ = inject_into_noise(campaign[key]["parameters"], int(key[3:]))
        build({}, ifos=ifos, trigger_time=trigger)
        values = {prior: shift(f"{key}{tag}_{suffix}")
                  for prior, suffix in (("agnostic", "env"), ("superradiance", "env_sr"))}
        print(f"  {key}: " + ", ".join(f"{prior} {v:+.2f} (ESS {e:.0f})" for prior, (v, _, e) in values.items()),
              flush=True)
        if key != GLITCH:
            for prior, (v, _, _) in values.items():
                shifts[prior].append(v)
    for prior, s in shifts.items():
        print(f"  {prior}, {len(s)} clean: mean {np.mean(s):+.2f}, sd {np.std(s):.2f}, "
              f"range [{min(s):+.2f}, {max(s):+.2f}]")
