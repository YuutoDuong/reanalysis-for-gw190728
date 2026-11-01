"""The vacuum background, split by the loudness of the injected signal.

The injections scatter in distance and inclination, so their network
signal-to-noise ratio (SNR) ranges from 3 to 20 while GW190728 has 12.7.
A vacuum signal too weak to constrain the density gives ln B ~ 0 whatever
the noise does, and a loud one is pushed to negative ln B by the prior
volume it rules out, so the background of the event is the injections of
its own loudness.  Print ln B, the gain in median log-likelihood and the
shifts with network and spectra by SNR class, and for the injections
within MATCH of the event's SNR, next to GW190728's own values.
--det: the redshifted campaign.  Reads script 30's results and
injections.network_snr.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, ".")
import bilby
from scalar_env.injections import EVENT_SNR, GLITCH, MATCH, network_snr, _tag

bilby.core.utils.logger.setLevel("ERROR")
NETWORKS = {"H1+L1": "gw190728_env", "H1+L1+V1": "gw190728_env_hlv",
            "H1+L1+V1, BayesWave": "gw190728_env_hlv_gwtc21psd"}
SR = {"H1+L1": "gw190728_env_sr_eq9_taud1e6_rho2e8",
      "H1+L1+V1, BayesWave": "gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8"}
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2", "gw190728_env_rho_max2e8"]


def line(name, values, event):
    values = np.asarray(values)
    return (f"    {name:34s} N {values.size:2d}: median {np.median(values):+.2f}, range [{values.min():+.2f}, "
            f"{values.max():+.2f}]; GW190728 {event:+.2f}, reached by {(values >= event).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vacuum background by signal-to-noise ratio")
    parser.add_argument("--det", action="store_true", help="the redshifted campaign")
    tag = _tag(parser.parse_args().det)
    snr = network_snr(tag)
    results = json.loads(Path(f"injection_virgo{tag}.json").read_text())
    lnb = json.loads(Path("reweighted_lnb.json").read_text())["lnb"]
    event = {n: lnb[l]["exact"] for n, l in NETWORKS.items()}
    event["H1+L1"] = float(np.mean([lnb[l]["exact"] for l in BASE]))
    clean = sorted(k for k in results if k != GLITCH)
    x = np.array([snr[k] for k in clean])
    y = np.array([results[k]["H1+L1"]["agnostic"]["lnb"] for k in clean])
    print(f"{len(clean)} clean injections; network SNR median {np.median(x):.1f}, range [{x.min():.1f}, {x.max():.1f}]; "
          f"correlation of ln B with SNR {np.corrcoef(x, y)[0, 1]:+.2f}")
    print(f"at or above GW190728's ln B ({event['H1+L1']:+.2f}): "
          + ", ".join(f"{k} (SNR {snr[k]:.1f})" for k in clean if results[k]["H1+L1"]["agnostic"]["lnb"] >= event["H1+L1"]))

    classes = {"SNR < 7": x < 7, "SNR 7-10": (x >= 7) & (x < 10), "SNR >= 10": x >= 10,
               f"SNR within {MATCH} of {EVENT_SNR}": np.abs(x - EVENT_SNR) <= MATCH}
    for name, mask in classes.items():
        keys = [k for k, m in zip(clean, mask) if m]
        print(f"\n{name}")
        for network in NETWORKS:
            print(line(f"ln B, {network}", [results[k][network]["agnostic"]["lnb"] for k in keys], event[network]))
        print(line("gain in median ln L, H1+L1", [results[k]["H1+L1"]["agnostic"]["gain"] for k in keys], 2.49))
        for network in ("H1+L1+V1", "H1+L1+V1, BayesWave"):
            shift = [results[k][network]["agnostic"]["lnb"] - results[k]["H1+L1"]["agnostic"]["lnb"] for k in keys]
            print(line(f"shift to {network}", shift, event[network] - event["H1+L1"]))
        for network, label in SR.items():
            values = [results[k][network]["superradiance"]["lnb"] for k in keys if "superradiance" in results[k][network]]
            if values:
                print(line(f"superradiance ln B, {network}", values, lnb[label]["exact"]))
    matched = [k for k, m in zip(clean, classes[f"SNR within {MATCH} of {EVENT_SNR}"]) if m]
    print("\nmatched injections: " + ", ".join(
        f"{k} (SNR {snr[k]:.1f}, {results[k]['H1+L1']['agnostic']['lnb']:+.2f})" for k in matched))
    values = np.array([results[k]["H1+L1"]["agnostic"]["lnb"] for k in matched])
    n, k = values.size, int((values >= event["H1+L1"]).sum())
    chance = (f"none of {n} reaches it: the one-sided 95% upper limit on the chance of a vacuum signal doing so is "
              f"{1 - 0.05 ** (1 / n):.0%}" if k == 0 else
              f"{k} of {n} reach it: the chance of a vacuum signal doing so is {k / n:.2f}, 95% interval "
              f"[{beta.ppf(0.025, k, n - k + 1):.3f}, {beta.ppf(0.975, k + 1, n - k):.2f}]")
    print(f"GW190728 lies {(event['H1+L1'] - values.mean()) / values.std(ddof=1):.1f} standard deviations above their mean "
          f"({values.mean():+.2f} +- {values.std(ddof=1):.2f}); {chance}")
