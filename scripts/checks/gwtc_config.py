"""Print the GWTC-2.1 PE settings for GW190728 stored in the release file."""
import h5py

KEYS = ("ifos", "seglen", "srate", "flow", "fhigh", "f-ref", "padding", "psd-length",
        "psd-start-time", "amporder", "approx", "distance-max", "q-min", "chirpmass-min",
        "chirpmass-max", "a_spin1-max", "enable-spline-calibration", "tukey",
        "roll-off", "window", "bayesline", "Niter", "Nchain", "channels", "types",
        "deltaLogP", "nlive", "sampler", "distance-marginalisation", "approximant")
with h5py.File("data_cache/IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_cosmo.h5") as f:
    cfg = f["C01:IMRPhenomXPHM/config_file"]
    for section in cfg:
        for k in cfg[section]:
            if any(key in k for key in KEYS):
                print(f"[{section}] {k} = {cfg[section][k][()]}")
    meta = f["C01:IMRPhenomXPHM/meta_data"]
    for group in meta:
        for k in meta[group]:
            v = meta[group][k]
            print(f"<{group}> {k} = {v[()] if isinstance(v, h5py.Dataset) else list(v)}"[:200])
