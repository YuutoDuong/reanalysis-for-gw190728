"""The 2x2 network x PSD table by reweighting, no sampling:
    Z_target = Z_source,RB * E_post,source[ L_target,exact / L_source,RB ].
Two independent sources per missing cell, and one cell that was also
sampled directly (H1+L1+V1, catalogue PSDs) as a check."""
import multiprocessing as mp
import sys
import zlib
import numpy as np
from scipy.special import logsumexp

mp.set_start_method("fork", force=True)
sys.path.insert(0, ".")
import bilby
from scalar_env.inference import RunConfig, exact_likelihood, mass_cut_offset, OUTDIR

bilby.core.utils.logger.setLevel("WARNING")
N = 300
LOOKUP = "data_cache/distance_lookup_phase.npz"
H1L1, HLV = ("H1", "L1"), ("H1", "L1", "V1")
NOISE = {"H1L1 Welch": (H1L1, "welch"), "HLV Welch": (HLV, "welch"),
         "H1L1 GWTC": (H1L1, "gwtc21"), "HLV GWTC": (HLV, "gwtc21")}
SOURCES = {  # source noise model: (vacuum run, agnostic run, SR run)
    "H1L1 Welch": ("gw190728_vacuum", "gw190728_env_seed1", "gw190728_env_sr_eq9_taud1e6_rho2e8"),
    "HLV Welch": ("gw190728_vacuum_hlv", "gw190728_env_hlv", None),
    "HLV GWTC": ("gw190728_vacuum_hlv_gwtc21psd", "gw190728_env_hlv_gwtc21psd",
                 "gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8"),
}
JOBS = [  # (source, target)
    ("H1L1 Welch", "H1L1 GWTC"), ("HLV GWTC", "H1L1 GWTC"),
    ("H1L1 Welch", "HLV Welch"), ("HLV GWTC", "HLV Welch"),
    ("HLV Welch", "HLV GWTC"), ("H1L1 Welch", "HLV GWTC"),
]
EXACT = {}
for name, (dets, psd) in NOISE.items():
    for model in ("vacuum", "environment"):
        cfg = RunConfig(label="_", model=model, detectors=dets, psd_source=psd,
                        prior_kwargs=dict(rho_max_gcm3=2e8) if model == "environment" else {})
        EXACT[name, model] = exact_likelihood(cfg, LOOKUP)


def lnl(task):
    key, params = task
    return float(EXACT[key].log_likelihood_ratio(params))


def reweight(label, target, pool):
    rng = np.random.default_rng(zlib.crc32((label + target).encode()))
    res = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json"))
    post = res.posterior
    rows = post.iloc[rng.choice(len(post), min(N, len(post)), replace=False)]
    model = "environment" if "alpha_cloud" in post else "vacuum"
    tasks = [((target, model), {**r.to_dict(), "phase": 0.0}) for _, r in rows.iterrows()]
    delta = np.array(pool.map(lnl, tasks)) - rows["log_likelihood"].to_numpy()
    w = np.exp(delta - delta.max())
    ess = w.sum() ** 2 / (w ** 2).sum()
    return res.log_evidence + logsumexp(delta) - np.log(delta.size), delta.std(), ess


offset = mass_cut_offset()
with mp.Pool(10) as pool:
    for source, target in JOBS:
        vac, agn, sr = SOURCES[source]
        zv, sv, ev = reweight(vac, target, pool)
        line = f"{source:10s} -> {target:10s}: vac sd {sv:.2f} ESS {ev:5.0f}"
        for name, label in (("agnostic", agn), ("SR", sr)):
            if label:
                z, s, e = reweight(label, target, pool)
                line += f" | {name} {z - zv + offset:+.2f} (sd {s:.2f}, ESS {e:4.0f})"
        print(line, flush=True)
