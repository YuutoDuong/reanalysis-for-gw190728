"""Section V.H -- Can an error of the vacuum waveform pass for an environment?

Every injection so far (scripts 05, 06, 31) was made and recovered with
one model, IMRPhenomXPHM, so none can contain an error of that model:
the explanation of GW190728's rise with Virgo and the catalogue's spectra
that the injections leave open besides noise.  Inject vacuum signals made
with SEOBNRv4PHM, the effective-one-body model with which GWTC-2.1 also
analysed GW190728, into the ten noise segments of script 31 with the
same source parameters, and recover them with IMRPhenomXPHM exactly like
the others.

The two models define the reference phase, the merger time and the spins
differently (EOB takes the spin directions at the start of its waveform,
about 18 Hz, not at 20 Hz), so the same parameters can give signals of
opposite sign in the detectors, which then meet the noise differently.
Each EOB signal is therefore given the phase, polarisation angle and
arrival time that bring it closest to its XPHM twin: the largest real
network overlap, noise-weighted with the segment's spectra (0.993-0.999
for the ten; before alignment -0.96 to +0.90, negative in six).  Each
segment then holds one source as an XPHM signal, an XPHM signal with a
cloud and an EOB signal, in identical noise, and the differences isolate
the cloud and the waveform error.  The alignment takes ~3 min per
injection and is kept in injection_alignment_det_seobnrv4phm.json.

--sr re-analyses the ten under the superradiance prior at tau_d = 1e6 yr.
Script 30 --det --approximant SEOBNRv4PHM then gives the exact
likelihood, the networks with Virgo and the gains in fit, and
scripts/checks/eob_injections.py every number of Sec. V.H.

Wall time on 10 cores: ~2 days (--sr: ~1 day more); restartable.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import mass_cut_offset
from scalar_env.injections import (draw_injection_parameters, inject_into_noise, run_campaign,
                                   run_sr_campaign, waveform_generator)

APPROXIMANT = "SEOBNRv4PHM"
INDICES = (1, 2, 3, 5, 8, 16, 22, 24, 42, 43)         # the segments of script 31
ALIGNMENT = Path("injection_alignment_det_seobnrv4phm.json")
LAGS = np.arange(-3e-3, 3e-3, 5e-5)                    # arrival-time offsets tried [s]


def align(params, index):
    """The phase, polarisation angle and arrival time that bring the EOB
    signal closest to its XPHM twin in the twin's detectors."""
    ifos, trigger, _ = inject_into_noise(params, index)          # the twin's segment and spectra
    p = dict(params, geocent_time=trigger)
    twin = waveform_generator().frequency_domain_strain(p)
    target = [ifo.get_detector_response(twin, p) for ifo in ifos]
    weight = [np.where(ifo.frequency_mask, 1 / ifo.power_spectral_density_array, 0) for ifo in ifos]
    delay = np.exp(-2j * np.pi * np.outer(LAGS, ifos[0].frequency_array))
    power = lambda h: sum(np.sum(np.abs(x) ** 2 * w) for x, w in zip(h, weight))
    eob = waveform_generator(APPROXIMANT)

    def best(phases, psis):
        """(overlap, (phase, psi, lag)) of the best combination on a grid."""
        found = (-2.0, None)
        for phase in phases:
            polarisations = eob.frequency_domain_strain(dict(p, phase=phase))
            for psi in psis:
                q = dict(p, phase=phase, psi=psi)
                h = [ifo.get_detector_response(polarisations, q) for ifo in ifos]
                cross = sum(np.conj(x) * y * w for x, y, w in zip(target, h, weight))
                overlap = (delay @ cross).real / np.sqrt(power(target) * power(h))
                i = int(np.argmax(overlap))
                found = max(found, (overlap[i], (phase, psi, LAGS[i])))
        return found

    _, (phase, psi, _) = best(np.linspace(0, 2 * np.pi, 24, endpoint=False),
                              np.linspace(0, np.pi, 24, endpoint=False))
    overlap, (phase, psi, lag) = best(phase + np.linspace(-0.26, 0.26, 9),
                                      psi + np.linspace(-0.13, 0.13, 9))
    return dict(phase=float(phase % (2 * np.pi)), psi=float(psi % np.pi),
                geocent_time=float(trigger + lag), overlap=float(overlap))


def aligned_parameters():
    """{index: the drawn parameters with the EOB signal aligned}, computed once."""
    done = json.loads(ALIGNMENT.read_text()) if ALIGNMENT.exists() else {}
    draws = draw_injection_parameters(50)
    for index in INDICES:
        key = f"inj{index:03d}"
        if key not in done:
            done[key] = align(draws[index], index)
            ALIGNMENT.write_text(json.dumps(done, indent=2))
            print(f"{key}: aligned, real overlap with the XPHM twin {done[key]['overlap']:.3f}",
                  flush=True)
    return {int(key[3:]): dict(draws[int(key[3:])], **{k: v for k, v in a.items() if k != "overlap"})
            for key, a in done.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vacuum injections made with a second waveform model")
    parser.add_argument("--sr", action="store_true", help="superradiance prior, tau_d = 1e6 yr")
    if parser.parse_args().sr:
        results = run_sr_campaign(tau_d_yr=1e6, approximant=APPROXIMANT,
                                  n_high=len(INDICES), n_typical=0)
        lnb = [v["lnb_sr"] for v in results.values()]
    else:
        lnb = run_campaign(approximant=APPROXIMANT, indices=INDICES, parameters=aligned_parameters())

    lnb = np.array(lnb) + mass_cut_offset()
    print(f"\n{lnb.size} vacuum injections made with {APPROXIMANT}, ln B as sampled: "
          f"median {np.median(lnb):+.2f}, range [{lnb.min():+.2f}, {lnb.max():+.2f}]")
