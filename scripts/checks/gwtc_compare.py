"""Our vacuum posteriors against the GWTC-2.1 XPHM posterior of GW190728."""
import json
import h5py
import numpy as np

with h5py.File("data_cache/IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_cosmo.h5") as f:
    g = f["C01:IMRPhenomXPHM/posterior_samples"][()]
    print("GWTC columns with snr/likelihood:",
          [n for n in g.dtype.names if "snr" in n or "likelihood" in n])
    gw = {k: np.asarray(g[k]) for k in g.dtype.names}
    cfg = f["C01:IMRPhenomXPHM"]
    for key in ("config_file", "meta_data"):
        if key in cfg:
            print("has", key, list(cfg[key].keys())[:20])


def mag(col):
    return np.array([abs(complex(x["real"], x["imag"])) if isinstance(x, dict) else abs(x) for x in col])


def ours(label):
    with open(f"outdir/{label}/{label}_result.json") as fh:
        post = json.load(fh)["posterior"]["content"]
    out = {k: np.asarray(post[k], dtype=float) for k in
           ("chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance", "log_likelihood")}
    dets = [d for d in ("H1", "L1", "V1") if f"{d}_matched_filter_snr" in post]
    out["network_matched_filter_snr"] = np.sqrt(sum(mag(post[f"{d}_matched_filter_snr"]) ** 2 for d in dets))
    return out


def summary(x):
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return f"{med:8.3f} [{lo:7.3f}, {hi:7.3f}]"


runs = {"GWTC-2.1 XPHM": gw,
        "ours H1L1 Welch": ours("gw190728_vacuum"),
        "ours HLV Welch": ours("gw190728_vacuum_hlv"),
        "ours HLV GWTC PSD": ours("gw190728_vacuum_hlv_gwtc21psd")}
for k in ("chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance", "network_matched_filter_snr"):
    print(f"\n{k}")
    for name, r in runs.items():
        if k in r:
            print(f"  {name:20s} {summary(r[k])}")
