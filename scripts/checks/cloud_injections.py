"""What the pipeline returns when the environment is real (Sec. V.G).

The campaign of environmental signals (script 31, then script 30 --det
--cloud) put a cloud at GW190728's agnostic posterior medians into ten
segments of the second vacuum campaign, with that campaign's source
parameters.  Next to their vacuum twins in the same noise and to
GW190728, print every number the paper quotes about it:

  ln B    exact, H1+L1, under both priors; the shifts with Virgo and with
          Virgo and the catalogue's spectra; the gain in fit;
  fit     the gain in posterior-mean log-likelihood of H1 and L1 (as
          sampled) against each detector's share w_d of the power of the
          injected environmental term, and the same for GW190728;
  cloud   recovery of the injected (alpha, rho_phi) under both priors, and
          the superradiance posterior against the agnostic one cut to the
          superradiance region (the truncation of Sec. V.C).

Reads results only, apart from one waveform per injection for w_d.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
import bilby
from scalar_env import data
from scalar_env.inference import exact_posterior
from scalar_env.injections import inject_into_noise
from scalar_env.priors import _env_conversion
from scalar_env.waveform import scalar_cloud_bbh

bilby.core.utils.logger.setLevel("WARNING")
PRIORS = ("agnostic", "superradiance")
NETWORKS = ("H1+L1", "H1+L1+V1", "H1+L1+V1, BayesWave")
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2", "gw190728_env_rho_max2e8"]
EVENT_RUNS = {  # network: GW190728's (agnostic, superradiance) runs, scripts 02, 04, 20
    "H1+L1": (BASE, ["gw190728_env_sr_eq9_taud1e6_rho2e8"]),
    "H1+L1+V1": (["gw190728_env_hlv"], []),     # superradiance: reweighted, below
    "H1+L1+V1, BayesWave": (["gw190728_env_hlv_gwtc21psd"],
                            ["gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8"]),
}
WFG = bilby.gw.WaveformGenerator(   # as injected (injections.inject_into_noise)
    duration=8, sampling_frequency=data.SAMPLE_RATE,
    frequency_domain_source_model=scalar_cloud_bbh,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                            reference_frequency=data.F_LOW, minimum_frequency=data.F_LOW))

load = lambda name: json.loads(Path(name).read_text())
EXACT = load("reweighted_lnb.json")["lnb"]
CLOUD, TWIN = load("injection_virgo_det_cloud.json"), load("injection_virgo_det.json")
PARAMETERS = {k: v["parameters"] for k, v in load("injection_lnb_det_cloud.json").items()}
KEYS = sorted(CLOUD)


def result(label):
    return load(f"outdir/{label}/{label}_result.json")


def posterior(label, columns):
    content = result(label)["posterior"]["content"]
    return {c: np.asarray(content[c], dtype=float) for c in columns}


def event(network, prior):
    runs = EVENT_RUNS[network][PRIORS.index(prior)]
    if not runs:    # never sampled: Table IV's reweighted cell (logs/noise_reweight_harmonics.log)
        return {("H1+L1+V1", "superradiance"): 2.07}[network, prior]
    return np.mean([EXACT[r]["exact"] for r in runs])


def shift(entry, network, prior):
    return entry[network][prior]["lnb"] - entry["H1+L1"][prior]["lnb"]


def loudness(key):
    snr = result(f"{key}_det_cloud_env")["meta_data"]["likelihood"]["interferometers"]
    return float(np.hypot(*[v["optimal_SNR"] for v in snr.values()]))


def shares(key):
    """Each detector's share w_d of the power of the injected environmental
    term h_env - h_vac, with the injection's own noise spectra."""
    params = PARAMETERS[key]
    ifos, _, injection = inject_into_noise(params, int(key[3:]), detectors=("H1", "L1", "V1"))
    h = WFG.frequency_domain_strain(injection)
    h0 = WFG.frequency_domain_strain({**injection, "rho_phi": 0.0})
    power = np.array([ifo.optimal_snr_squared(ifo.get_detector_response(h, injection)
                                              - ifo.get_detector_response(h0, injection)).real
                      for ifo in ifos])
    return power / power.sum()


def fit_by_detector(env_labels, vac_label):
    """Posterior-mean ln L_d, environment minus vacuum, for H1 and L1 (as sampled)."""
    columns = ["H1_log_likelihood", "L1_log_likelihood"]
    env = np.mean([[posterior(l, columns)[c].mean() for c in columns] for l in env_labels], axis=0)
    vac = np.array([posterior(vac_label, columns)[c].mean() for c in columns])
    return env - vac


def summary(name, values, observed=None, larger=True):
    values = np.asarray(values)
    line = (f"{name}: median {np.median(values):+.2f}, range [{values.min():+.2f}, "
            f"{values.max():+.2f}]")
    if observed is not None:
        n = np.sum(values >= observed) if larger else np.sum(values <= observed)
        line += f"; GW190728 {observed:+.2f}, {'reached' if larger else 'undercut'} by {n} of {values.size}"
    print(line)


if __name__ == "__main__":
    print("per injection (exact, offset included): SNR | twin agn, SR | cloud agn, SR | "
          "shift V, V+BW (agn) [ESS] | shift V, V+BW (SR)")
    for k in KEYS:
        c, t = CLOUD[k], TWIN[k]["H1+L1"]
        print(f"{k} {loudness(k):5.2f} | {t['agnostic']['lnb']:+.2f} "
              f"{t.get('superradiance', {}).get('lnb', np.nan):+.2f} | "
              f"{c['H1+L1']['agnostic']['lnb']:+.2f} {c['H1+L1']['superradiance']['lnb']:+.2f} | "
              + " ".join(f"{shift(c, n, 'agnostic'):+.2f} [{c[n]['agnostic']['ess']:.0f}]"
                         for n in NETWORKS[1:]) + " | "
              + " ".join(f"{shift(c, n, 'superradiance'):+.2f} [{c[n]['superradiance']['ess']:.0f}]"
                         for n in NETWORKS[1:]))

    print("\n--- two detectors")
    for prior in PRIORS:
        cloud = [CLOUD[k]["H1+L1"][prior]["lnb"] for k in KEYS]
        twin = [TWIN[k]["H1+L1"][prior]["lnb"] for k in KEYS]
        summary(f"{prior:13s} clouds", cloud, event("H1+L1", prior))
        summary(f"{prior:13s} twins ", twin, event("H1+L1", prior))
        summary(f"{prior:13s} cloud - twin", np.subtract(cloud, twin))
    gains = [CLOUD[k]["H1+L1"]["agnostic"]["gain"] for k in KEYS]
    median = lambda label: exact_posterior(label)["log_likelihood"].median()
    observed = np.mean([median(l) for l in BASE]) - median("gw190728_vacuum")
    summary("gain in median log-likelihood, agnostic", gains, observed)
    rise = [CLOUD[k]["H1+L1"]["superradiance"]["lnb"] - CLOUD[k]["H1+L1"]["agnostic"]["lnb"] for k in KEYS]
    summary("superradiance minus agnostic, clouds", rise,
            event("H1+L1", "superradiance") - event("H1+L1", "agnostic"))

    print("\n--- the rise with the network and the spectra (shift from H1+L1)")
    for network in NETWORKS[1:]:
        for prior in PRIORS:
            values = np.array([shift(CLOUD[k], network, prior) for k in KEYS])
            reliable = np.array([CLOUD[k][network][prior]["ess"] >= 30 for k in KEYS])
            summary(f"{network:20s} {prior:13s}", values,
                    event(network, prior) - event("H1+L1", prior))
            print(f"{'':35s}ESS >= 30: {reliable.sum()} of {reliable.size}, median "
                  f"{np.median(values[reliable]):+.2f}, max {values[reliable].max():+.2f}")

    print("\n--- recovery of the injected cloud (as sampled; truth inside the 90% interval?)")
    truth = {n: PARAMETERS[KEYS[0]][n] for n in ("alpha_cloud", "rho_phi")}
    print("injected:", truth)
    for suffix in ("env", "env_sr"):
        rows = []
        for k in KEYS:
            post = posterior(f"{k}_det_cloud_{suffix}", truth)
            rows.append([(np.median(post[n]), *np.quantile(post[n], [0.05, 0.95]),
                          np.mean(post[n] < v)) for n, v in truth.items()])
        rows = np.array(rows)          # (injection, parameter, [median, lo, hi, percentile])
        for i, (name, value) in enumerate(truth.items()):
            inside = np.sum((rows[:, i, 1] <= value) & (value <= rows[:, i, 2]))
            print(f"{suffix:6s} {name:11s}: medians {rows[:, i, 0].min():.3g}-{rows[:, i, 0].max():.3g} "
                  f"(mean {rows[:, i, 0].mean():.3g}), truth at the "
                  f"{rows[:, i, 3].min():.0%}-{rows[:, i, 3].max():.0%} percentiles, "
                  f"inside the 90% interval in {inside} of {len(KEYS)}")

    print("\n--- superradiance posterior = agnostic posterior cut to its region?")
    for k in KEYS:
        columns = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
        agn = posterior(f"{k}_det_cloud_env", columns)
        c = _env_conversion(dict(agn), tau_d_yr=1e6)
        keep = ((c["sr_window_lo"] >= 1) & (c["sr_window_hi"] <= 1) & (c["sr_density_ratio"] <= 1)
                & (c["alpha_bound_ratio"] <= 1) & (c["cloud_mass_ratio"] <= 1))
        sr = posterior(f"{k}_det_cloud_env_sr", columns)
        print(f"{k}: kept {keep.mean():.0%}; alpha {np.median(agn['alpha_cloud']):.3f} -> cut "
              f"{np.median(agn['alpha_cloud'][keep]):.3f}, run {np.median(sr['alpha_cloud']):.3f}; "
              f"rho {np.median(agn['rho_phi']):.2e} -> cut {np.median(agn['rho_phi'][keep]):.2e}, "
              f"run {np.median(sr['rho_phi']):.2e}")

    print("\n--- gain in fit by detector (posterior-mean ln L_d, env - vac, as sampled) [w_d]")
    splits, weights = [], []
    for k in KEYS:
        split, w = fit_by_detector([f"{k}_det_cloud_env"], f"{k}_det_cloud_vac"), shares(k)
        twin = fit_by_detector([f"{k}_det_env"], f"{k}_det_vac")
        splits.append(split)
        weights.append(w)
        print(f"{k}: cloud H1 {split[0]:+.2f}, L1 {split[1]:+.2f}  [w H1 {w[0]:.0%}, L1 {w[1]:.0%}, "
              f"V1 {w[2]:.1%}];  twin H1 {twin[0]:+.2f}, L1 {twin[1]:+.2f}")
    splits, weights = np.array(splits), np.array(weights)
    l1_share = splits[:, 1] / splits.sum(1)
    print(f"clouds: H1 median {np.median(splits[:, 0]):+.2f}, L1 median {np.median(splits[:, 1]):+.2f}; "
          f"L1's part of the H1+L1 gain {np.median(l1_share):.0%} (range {l1_share.min():.0%}-"
          f"{l1_share.max():.0%}) against w_L1/(w_H1+w_L1) "
          f"{np.median(weights[:, 1] / weights[:, :2].sum(1)):.0%}; L1 gains more than H1 in "
          f"{np.sum(splits[:, 1] > splits[:, 0])} of {len(KEYS)}; w_V1 {weights[:, 2].min():.1%}-"
          f"{weights[:, 2].max():.1%}")
    split = fit_by_detector(BASE, "gw190728_vacuum")
    print(f"GW190728 (same estimator, four baseline runs): H1 {split[0]:+.2f}, L1 {split[1]:+.2f} "
          f"(the paper's per-detector table, exact likelihood: +1.32, -0.69)")
