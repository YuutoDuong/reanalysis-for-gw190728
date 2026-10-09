"""Do Virgo and the catalogue spectra raise ln B for vacuum signals too?

For GW190728, adding Virgo raises ln B by ~1.2 although Virgo carries only
3% of the power of the environmental term, and the catalogue's BayesWave
spectra raise it by a further ~1.5.  Ask whether real noise does the same
to vacuum signals.  For every injection of a campaign, rebuild its data
with Virgo's off-source strain added (same signal injected), and reweight
the H1+L1 runs' posterior samples to the exact likelihood of each
network,

    Z_network = Z_RB * E_post[ L_exact,network / L_RB ],

on the same samples for every network:
  H1+L1                 the campaign's own data (full-grid likelihood);
  H1+L1+V1              Welch PSDs from the preceding 512 s, as for H1+L1;
  H1+L1+V1, BayesWave   the event's GWTC-2.1 BayesWave PSDs, standing in
                        for on-source estimates (the segments lie within
                        ~20 min of the event).
The first also puts the background on the exact likelihood quoted for
GW190728.  Writes injection_virgo{tag}.json (per injection, network and
prior: exact ln B, log-weight scatter, effective sample size) and prints
the shift distributions next to GW190728's own.  Resumable per network.
--det: the redshifted campaign.  ~1 h per network on 4 cores.
"""

import argparse
import json
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
from scalar_env.injections import inject_into_noise, RHO_MAX, _tag

bilby.core.utils.logger.setLevel("WARNING")
EVENT = "GW190728_064510"
N_SAMPLES = 300
NPOOL = int(os.environ.get("NPOOL", 4))
HLV = ("H1", "L1", "V1")
NETWORKS = {  # name: (detectors, PSD source)
    "H1+L1": (("H1", "L1"), "welch"),
    "H1+L1+V1": (HLV, "welch"),
    "H1+L1+V1, BayesWave": (HLV, "gwtc21"),
}
GW190728 = {  # network: GW190728's agnostic run at 2e8 (script 17 labels)
    "H1+L1+V1": "gw190728_env_hlv",
    "H1+L1+V1, BayesWave": "gw190728_env_hlv_gwtc21psd",
}
GLITCH = "inj028"
LIKELIHOODS = {}   # of the injection being processed, inherited by the forked pool


def injected(index, parameters, network):
    """The injection's data for one network."""
    detectors, psd_source = NETWORKS[network]
    ifos, trigger, _ = inject_into_noise(parameters, index, detectors=detectors)
    if psd_source == "gwtc21":
        for ifo in ifos:
            f, psd = data.catalogue_psd(EVENT, ifo.name)
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                frequency_array=f, psd_array=psd)
    return ifos, trigger


def likelihoods(index, parameters, networks):
    """Exact likelihoods of one injection, per network and model."""
    out = {}
    for network in networks:
        ifos, trigger = injected(index, parameters, network)
        for model in ("vacuum", "environment"):
            config = RunConfig(label="_", model=model, detectors=NETWORKS[network][0],
                               prior_kwargs=dict(rho_max_gcm3=RHO_MAX) if model == "environment" else {})
            out[network, model] = exact_likelihood(config, "data_cache/distance_lookup_phase.npz",
                                                   ifos=ifos, trigger_time=trigger)
    return out


def exact_lnl(task):
    key, params = task
    return float(LIKELIHOODS[key].log_likelihood_ratio(params))


def ln_z(label, network, pool):
    """(exact ln Z, log-weight scatter, effective sample size) of one run."""
    rng = np.random.default_rng(zlib.crc32(label.encode()))   # same samples for every network
    result = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json"))
    post = result.posterior
    rows = post.iloc[rng.choice(len(post), min(N_SAMPLES, len(post)), replace=False)]
    model = "environment" if "alpha_cloud" in post else "vacuum"
    tasks = [((network, model), {**r.to_dict(), "phase": 0.0}) for _, r in rows.iterrows()]
    delta = np.array(pool.map(exact_lnl, tasks)) - rows["log_likelihood"].to_numpy()
    w = np.exp(delta - delta.max())
    return (float(result.log_evidence + logsumexp(delta) - np.log(delta.size)),
            float(delta.std()), float(w.sum() ** 2 / (w ** 2).sum()))


def summary(results):
    lnb = json.loads(Path("reweighted_lnb.json").read_text())["lnb"]
    base = np.mean([lnb[l]["exact"] for l in ("gw190728_env", "gw190728_env_seed1",
                                              "gw190728_env_seed2", "gw190728_env_rho_max2e8")])
    clean = {k: v for k, v in results.items() if k != GLITCH}
    background = np.array([v["H1+L1"]["agnostic"]["lnb"] for v in clean.values()])
    print(f"\nH1+L1 background, full grid ({background.size} clean): median "
          f"{np.median(background):+.2f}, P(>= {base:+.2f}) = {np.mean(background >= base):.2f}")
    for network, label in GW190728.items():
        for prior in ("agnostic", "superradiance"):
            pairs = [(v[network][prior], v["H1+L1"][prior]) for v in clean.values()
                     if prior in v.get(network, {})]
            if not pairs:
                continue
            shift = np.array([a["lnb"] - b["lnb"] for a, b in pairs])
            reliable = np.array([a["ess"] >= 30 for a, _ in pairs])
            line = (f"{network:20s} {prior:13s} shift over {shift.size}: median "
                    f"{np.median(shift):+.2f}, 90% in [{np.quantile(shift, 0.05):+.2f}, "
                    f"{np.quantile(shift, 0.95):+.2f}], max {shift.max():+.2f}"
                    f" (ESS >= 30: median {np.median(shift[reliable]):+.2f}, N {reliable.sum()})")
            if prior == "agnostic":
                observed = lnb[label]["exact"] - base
                line += f"; GW190728 {observed:+.2f}, exceeded by {np.mean(shift >= observed):.0%}"
            print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virgo and PSD null test on an injection campaign")
    parser.add_argument("--det", action="store_true", help="the redshifted campaign")
    tag = _tag(parser.parse_args().det)
    out_path = Path(f"injection_virgo{tag}.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    offset = mass_cut_offset()
    campaign = json.loads(Path(f"injection_lnb{tag}.json").read_text())
    for key, record in sorted(campaign.items()):
        missing = [n for n in NETWORKS if n not in results.get(key, {})]
        if not missing:
            continue
        runs = {prior: f"{key}{tag}_{suffix}" for prior, suffix in
                (("vacuum", "vac"), ("agnostic", "env"), ("superradiance", "env_sr"))
                if (OUTDIR / f"{key}{tag}_{suffix}" / f"{key}{tag}_{suffix}_result.json").exists()}
        try:
            LIKELIHOODS.clear()
            LIKELIHOODS.update(likelihoods(int(key[3:]), record["parameters"], missing))
        except Exception as exc:
            print(f"{key}: SKIPPED ({type(exc).__name__}: {exc})", flush=True)
            continue
        entry = results.setdefault(key, {})
        with mp.get_context("fork").Pool(NPOOL) as pool:
            for network in missing:
                z = {prior: ln_z(label, network, pool) for prior, label in runs.items()}
                entry[network] = {prior: dict(lnb=z[prior][0] - z["vacuum"][0] + offset,
                                              scatter=max(z[prior][1], z["vacuum"][1]),
                                              ess=min(z[prior][2], z["vacuum"][2]))
                                  for prior in z if prior != "vacuum"}
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{key}: " + "; ".join(
            f"{network} " + ", ".join(f"{p} {v['lnb']:+.2f} (ESS {v['ess']:.0f})"
                                      for p, v in entry[network].items())
            for network in missing), flush=True)
    summary(results)
