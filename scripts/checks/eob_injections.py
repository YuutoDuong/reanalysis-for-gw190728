"""Can an error of the vacuum waveform pass for an environment? (Sec. V.H)

Script 32 (then script 30 --det --approximant SEOBNRv4PHM) put vacuum
signals made with SEOBNRv4PHM into the ten segments of the environmental
campaign, aligned with their IMRPhenomXPHM twins, and recovered them with
IMRPhenomXPHM like every other injection.  Next to the twins in the same
noise and to GW190728, print every number the paper quotes about it:

  signal  the overlap with the twin before and after alignment, the
          loudness of both, and the norm of their difference, <dh|dh> =
          rho_1^2 + rho_2^2 - 2 rho_1 rho_2 O: two waveforms are
          distinguishable at this loudness once it exceeds about 1
          (Lindblom et al. 2008);
  ln B    exact, H1+L1, under both priors; the shifts with Virgo and with
          Virgo and the catalogue's spectra; each against the twin's;
  fit     the shift of the recovered chirp mass from the twin's, in units
          of the twin's posterior width, and the 90% bound on rho_phi.

Reads results only, apart from two waveforms per injection for the overlap
before alignment.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cloud_injections import NETWORKS, PRIORS, event, load, posterior, result, shift, summary
from bilby.gw.utils import noise_weighted_inner_product
from scalar_env.injections import draw_injection_parameters, inject_into_noise, waveform_generator

EOB, TWIN = load("injection_virgo_det_seobnrv4phm.json"), load("injection_virgo_det.json")
OVERLAP = {k: v["overlap"] for k, v in load("injection_alignment_det_seobnrv4phm.json").items()}
KEYS = sorted(EOB)


def loudness(label):
    snr = result(label)["meta_data"]["likelihood"]["interferometers"]
    return float(np.hypot(*[v["optimal_SNR"] for v in snr.values()]))


def unaligned_overlap(key):
    """Real network overlap of the EOB signal with its XPHM twin at the
    drawn parameters, as script 32 measures it, before the alignment."""
    index = int(key[3:])
    params = draw_injection_parameters(50)[index]
    ifos, trigger, _ = inject_into_noise(params, index)
    p = dict(params, geocent_time=trigger)
    h = [[ifo.get_detector_response(wfg.frequency_domain_strain(p), p) for ifo in ifos]
         for wfg in (waveform_generator(), waveform_generator("SEOBNRv4PHM"))]
    inner = lambda a, b: sum(noise_weighted_inner_product(
        x[ifo.frequency_mask], y[ifo.frequency_mask],
        ifo.power_spectral_density_array[ifo.frequency_mask], ifo.duration) for x, y, ifo in zip(a, b, ifos)).real
    return inner(h[0], h[1]) / np.sqrt(inner(h[0], h[0]) * inner(h[1], h[1]))


def paired(name, values):
    """Mean and standard error of EOB-minus-twin differences."""
    values = np.asarray(values)
    print(f"{name}: mean {values.mean():+.2f} +/- {values.std(ddof=1) / np.sqrt(values.size):.2f} "
          f"(N {values.size}), range [{values.min():+.2f}, {values.max():+.2f}]")


if __name__ == "__main__":
    print("per injection (exact, offset included): overlap | SNR EOB, twin | <dh|dh> | "
          "EOB (twin) agn, SR | shift V, V+BW, agn: EOB (twin) [ESS]")
    norms = []
    for k in KEYS:
        rho = loudness(f"{k}_det_seobnrv4phm_vac"), loudness(f"{k}_det_vac")
        norms.append(rho[0] ** 2 + rho[1] ** 2 - 2 * rho[0] * rho[1] * OVERLAP[k])
        e, t = EOB[k], TWIN[k]
        print(f"{k} {OVERLAP[k]:.4f} | {rho[0]:5.2f} {rho[1]:5.2f} | {norms[-1]:4.2f} | "
              + " ".join(f"{e['H1+L1'][p]['lnb']:+.2f} ({t['H1+L1'][p]['lnb']:+.2f})" for p in PRIORS)
              + " | " + " ".join(f"{shift(e, n, 'agnostic'):+.2f} ({shift(t, n, 'agnostic'):+.2f}) "
                                 f"[{e[n]['agnostic']['ess']:.0f}]" for n in NETWORKS[1:]))
    overlaps, norms = np.array([OVERLAP[k] for k in KEYS]), np.array(norms)
    print(f"overlap {overlaps.min():.4f}-{overlaps.max():.4f}; <dh|dh> {norms.min():.2f}-{norms.max():.2f}"
          f" (median {np.median(norms):.2f}), above 1 in {np.sum(norms > 1)} of {len(KEYS)}")
    before = np.array([unaligned_overlap(k) for k in KEYS])
    print("overlap before alignment: " + ", ".join(f"{k} {o:+.2f}" for k, o in zip(KEYS, before))
          + f"; range [{before.min():+.2f}, {before.max():+.2f}], negative in {np.sum(before < 0)} of {len(KEYS)}")

    print("\n--- two detectors")
    for prior in PRIORS:
        eob = [EOB[k]["H1+L1"][prior]["lnb"] for k in KEYS]
        summary(f"{prior:13s} EOB", eob, event("H1+L1", prior))
        paired(f"{prior:13s} EOB - twin", np.subtract(eob, [TWIN[k]["H1+L1"][prior]["lnb"] for k in KEYS]))
    paired("gain in median log-likelihood (agnostic), EOB - twin",
           [EOB[k]["H1+L1"]["agnostic"]["gain"] - TWIN[k]["H1+L1"]["agnostic"]["gain"] for k in KEYS])

    print("\n--- the rise with the network and the spectra (shift from H1+L1)")
    for network in NETWORKS[1:]:
        for prior in PRIORS:
            values = np.array([shift(EOB[k], network, prior) for k in KEYS])
            reliable = np.array([EOB[k][network][prior]["ess"] >= 30 for k in KEYS])
            summary(f"{network:20s} {prior:13s}", values, event(network, prior) - event("H1+L1", prior))
            print(f"{'':35s}ESS >= 30: {reliable.sum()} of {reliable.size}, median "
                  f"{np.median(values[reliable]):+.2f}, max {values[reliable].max():+.2f}")
            both = [k for k in KEYS if min(EOB[k][network][prior]["ess"], TWIN[k][network][prior]["ess"]) >= 30]
            paired(f"{'':35s}EOB - twin, ESS >= 30 in both",
                   [shift(EOB[k], network, prior) - shift(TWIN[k], network, prior) for k in both])

    print("\n--- what the recovery absorbs (as sampled)")
    shifts, bounds = [], []
    for k in KEYS:
        mc = [posterior(f"{k}_det{s}_vac", ["chirp_mass"])["chirp_mass"] for s in ("_seobnrv4phm", "")]
        rho = [np.quantile(posterior(f"{k}_det{s}_env", ["rho_phi"])["rho_phi"], 0.9) for s in ("_seobnrv4phm", "")]
        shifts.append((np.median(mc[0]) - np.median(mc[1])) / np.std(mc[1]))
        bounds.append(rho[0] / rho[1])
        print(f"{k}: chirp mass {np.median(mc[0]):.3f} against {np.median(mc[1]):.3f} ({shifts[-1]:+.2f} sigma); "
              f"90% bound on rho_phi {rho[0]:.2e} against {rho[1]:.2e}")
    shifts, bounds = np.array(shifts), np.array(bounds)
    print(f"chirp-mass shift: mean {shifts.mean():+.2f} sigma, range [{shifts.min():+.2f}, {shifts.max():+.2f}]; "
          f"density bound EOB/twin {bounds.min():.2f}-{bounds.max():.2f}")
