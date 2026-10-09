"""Which detector carries the environmental fit?

An environmental dephasing is a change of the source's phase evolution,
the same for every detector: it alters neither the arrival-time delays
nor the amplitude and phase ratios between detectors.  Whether it is real
or fitted to Gaussian noise, the better fit it brings to detector d is on
average proportional to w_d, that detector's share of the power of the
environmental term dh = h_env - h_vac, with fluctuations ~ sqrt(w_d).  A
detector whose gain far exceeds its share is fitting something else:
non-Gaussian noise, or a noise model that misdescribes its data.

For the vacuum and environmental posteriors of each network, evaluate the
exact, unmarginalised log-likelihood of each detector separately,
ln L_d = <d|h>_d - <h|h>_d / 2, at posterior samples (distance and phase
as reconstructed by bilby), and print per detector the change in its
posterior mean between the two models next to w_d and the SNR^2 share.

Also checks each PSD against the on-source data: the noise-weighted
power of the residual d - h per frequency bin (h the vacuum best fit),
halved, is exponentially distributed with mean 1 when the PSD is right.
Per band, the mean (what the likelihood feels) and the median / ln 2
(the bulk of the bins, blind to narrow lines) should both be ~1.

~10 min on 3 cores, alongside the injection campaign.
"""

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env import data
from scalar_env.inference import OUTDIR
from scalar_env.waveform import scalar_cloud_bbh

bilby.core.utils.logger.setLevel("WARNING")
EVENT = "GW190728_064510"
N_SAMPLES = 400
NPOOL = 3
BANDS = [(20, 30), (30, 50), (50, 100), (100, 300), (300, 1024)]
H1L1, HLV = ("H1", "L1"), ("H1", "L1", "V1")
NETWORKS = {  # name: (detectors, PSD source, vacuum run, environmental runs)
    "H1+L1, Welch PSDs": (H1L1, "welch", "gw190728_vacuum",
                          ["gw190728_env", "gw190728_env_sr_eq9_taud1e6_rho2e8"]),
    "H1+L1+V1, Welch PSDs": (HLV, "welch", "gw190728_vacuum_hlv", ["gw190728_env_hlv"]),
    "H1+L1+V1, GWTC-2.1 PSDs": (HLV, "gwtc21", "gw190728_vacuum_hlv_gwtc21psd",
                                ["gw190728_env_hlv_gwtc21psd",
                                 "gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8"]),
}
IFOS = {(dets, psd): data.fetch_event(EVENT, detectors=dets, psd_source=psd)[0]
        for dets, psd, *_ in NETWORKS.values()}
WFG = bilby.gw.WaveformGenerator(  # rho_phi = 0 gives the vacuum waveform exactly
    duration=data.EVENTS[EVENT]["duration"], sampling_frequency=data.SAMPLE_RATE,
    frequency_domain_source_model=scalar_cloud_bbh,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                            reference_frequency=data.F_LOW, minimum_frequency=data.F_LOW))


def responses(ifos, params):
    params = {"alpha_cloud": 0.1, "rho_phi": 0.0, **params}   # vacuum samples
    polarisations = WFG.frequency_domain_strain(params)
    return [ifo.get_detector_response(polarisations, params) for ifo in ifos]


def per_detector(task):
    """(ln L_d, SNR_d^2, <dh|dh>_d) of every detector at one posterior
    sample; dh, the environmental term, vanishes for vacuum samples."""
    key, params = task
    ifos = IFOS[key]
    full = responses(ifos, params)
    vacuum = responses(ifos, {**params, "rho_phi": 0.0}) if params.get("rho_phi", 0) > 0 else full
    out = []
    for ifo, h, h0 in zip(ifos, full, vacuum):
        snr2 = ifo.optimal_snr_squared(h).real
        out.append((ifo.inner_product(h).real - snr2 / 2, snr2,
                    ifo.optimal_snr_squared(h - h0).real))
    return out


def evaluate(label, key, pool, rng):
    """ln L_d, SNR_d^2 and <dh|dh>_d, each (samples, detectors), and the samples."""
    post = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json")).posterior
    rows = post.iloc[rng.choice(len(post), min(N_SAMPLES, len(post)), replace=False)]
    samples = [r.to_dict() for _, r in rows.iterrows()]
    values = np.array(pool.map(per_detector, [(key, s) for s in samples]))
    return values[..., 0], values[..., 1], values[..., 2], samples


def residual_power(ifos, params):
    """Per band, (mean, median / ln 2) of the noise-weighted residual
    power per bin, halved."""
    out = {}
    for ifo, h in zip(ifos, responses(ifos, params)):
        power = (2 / ifo.duration * np.abs(ifo.frequency_domain_strain - h) ** 2
                 / ifo.power_spectral_density_array)
        f = ifo.frequency_array
        bands = [power[(f >= lo) & (f < hi)] for lo, hi in BANDS]
        out[ifo.name] = [(p.mean(), np.median(p) / np.log(2)) for p in bands]
    return out


def share(x):
    return x.mean(0) / x.mean(0).sum()


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    with mp.get_context("fork").Pool(NPOOL) as pool:
        for name, (dets, psd, vac, envs) in NETWORKS.items():
            key = (dets, psd)
            lnl_vac, snr2, _, samples = evaluate(vac, key, pool, rng)
            print(f"\n{name}: optimal SNR "
                  + ", ".join(f"{d} {s ** 0.5:.1f}" for d, s in zip(dets, snr2.mean(0)))
                  + "; vacuum <ln L_d> "
                  + ", ".join(f"{d} {x:+.1f}" for d, x in zip(dets, lnl_vac.mean(0))))
            print("  residual power / expectation, mean/median, "
                  + ", ".join(f"{lo}-{hi}" for lo, hi in BANDS) + " Hz")
            best = samples[int(np.argmax(lnl_vac.sum(1)))]
            for d, row in residual_power(IFOS[key], best).items():
                print(f"    {d}  " + "  ".join(f"{m:.2f}/{md:.2f}" for m, md in row))
            for env in envs:
                lnl_env, _, dh2, _ = evaluate(env, key, pool, rng)
                diff = lnl_env.mean(0) - lnl_vac.mean(0)
                err = np.sqrt(lnl_env.var(0) / len(lnl_env) + lnl_vac.var(0) / len(lnl_vac))
                print(f"  {env}: posterior-mean ln L_d, environment - vacuum"
                      f"  [w_d, SNR^2 share]")
                for d, x, e, w, s in zip(dets, diff, err, share(dh2), share(snr2)):
                    print(f"    {d}  {x:+.2f} +- {e:.2f}   [{w:4.0%}, {s:4.0%}]")
                print(f"    all {diff.sum():+.2f}", flush=True)
