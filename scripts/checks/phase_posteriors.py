"""Phase-sampled vs phase-marginalised posteriors, against GWTC-2.1."""
import json
import h5py
import numpy as np

with h5py.File("data_cache/IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_cosmo.h5") as f:
    g = f["C01:IMRPhenomXPHM/posterior_samples"][()]
    gw = {k: np.asarray(g[k]) for k in g.dtype.names}


def ours(label):
    with open(f"outdir/{label}/{label}_result.json") as fh:
        r = json.load(fh)
    post = r["posterior"]["content"]
    out = {k: np.asarray(post[k], dtype=float) for k in
           ("chirp_mass", "mass_ratio", "chi_eff", "chi_p", "luminosity_distance", "log_likelihood")
           if k in post}
    out["ln_bf"] = r["log_bayes_factor"]
    return out


def s(x):
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return f"{med:7.3f} [{lo:6.3f}, {hi:6.3f}]"


runs = {"GWTC-2.1": gw,
        "vac marginalised": ours("gw190728_vacuum"),
        "vac phase sampled": ours("gw190728_vacuum_phase_sampled"),
        "env marginalised": ours("gw190728_env_seed1"),
        "env phase sampled": ours("gw190728_env_phase_sampled")}
for k in ("chirp_mass", "mass_ratio", "chi_eff", "chi_p", "luminosity_distance", "log_likelihood"):
    print(k)
    for n, r in runs.items():
        if k in r:
            print(f"  {n:20s} {s(r[k])}")
for n, r in runs.items():
    if "ln_bf" in r:
        print(f"{n:20s} ln BF (RB) {r['ln_bf']:.2f}")
