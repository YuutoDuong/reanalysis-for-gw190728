"""Exact-likelihood evidences, by reweighting the relative-binning runs.

Every run sampled the relative-binning likelihood L_RB.  Its evidence for
the exact, full-grid likelihood follows from its posterior samples alone,

    Z_exact / Z_RB = E_post[ L_exact / L_RB ],

an importance-sampling identity that needs no new sampling.  bilby fixes
the coalescence phase at 0 under phase marginalisation, so the stored
log-likelihoods refer to phase 0 and the exact one is evaluated there too
(runs that sample the phase are evaluated at their own); the printed
scatter of ln(L_exact / L_RB) checks that both refer to the same points
(a few tenths at most, if so).

Each run's exact likelihood uses that run's own data, PSD, approximant
and marginalisations, read off its label (VARIANTS).  Writes
reweighted_lnb.json: per run the reported ln Z, the correction and its
bootstrap error; per environmental run ln B for the exact likelihood
(mass-cut offset included).  A run whose result is unchanged since the
last call (same ln Z) keeps its stored correction, so only new or redone
runs cost time; move reweighted_lnb.json aside to redo them all.  ~30 min
for every run with 10 processes; set NPOOL to use fewer.
"""

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
from scalar_env.inference import RunConfig, exact_likelihood, mass_cut_offset, OUTDIR

bilby.core.utils.logger.setLevel("WARNING")
N_SAMPLES = 250
NPOOL = int(os.environ.get("NPOOL", max(1, (os.cpu_count() or 2) - 2)))
HLV = ("H1", "L1", "V1")
VACUUM = "gw190728_vacuum"
SR = [f"gw190728_env_sr_eq9_taud1e{e}" for e in (5, 6, 7, 8)]
PAIRS = {  # environmental run -> the vacuum run it is compared with
    **{l: VACUUM for l in (
        "gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
        "gw190728_env_rho_max2e8", "gw190728_env_rho_max1e6",
        "gw190728_env_rho_max1e7", "gw190728_env_rho_max1e8",
        "gw190728_env_log_alpha", "gw190728_env_fixfid",
        "gw190728_env_fixfid_sr_eq9_taud1e6_rho2e8",
        "gw190728_env_sr_eq9_taud1e6_rho2e8_accwalk60",
        "gw190728_env_sr_eq9_taud1e6_rho2p2e7", "gw190728_env_sr_eq9_taud1e6_rho3e6",
        "gw190728_env_sr_eq9_taud1e8_rho2p7e7",
        *SR, *[s + "_rho2e8" for s in SR])},
    "gw190728_env_nessai": "gw190728_vacuum_nessai",
    "gw190728_env_sr_eq9_taud1e6_rho2e8_nessai": "gw190728_vacuum_nessai",
    "gw190728_env_pymultinest": "gw190728_vacuum_pymultinest",
    **{f"gw190728_env_{v}": f"gw190728_vacuum_{v}" for v in (
        "imrphenompv2", "imrphenomxp", "psd8s", "psd16s", "only_H1", "only_L1",
        "hlv", "hlv_gwtc21psd", "h1l1_gwtc21psd", "phase_sampled",
        "hlv_gwtc21psd_phase_sampled")},
    "gw190728_env_hlv_sr_eq9_taud1e6_rho2e8": "gw190728_vacuum_hlv",
    "gw190728_env_h1l1_gwtc21psd_sr_eq9_taud1e6_rho2e8": "gw190728_vacuum_h1l1_gwtc21psd",
    **{f"gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e{e}_rho2e8{s}": "gw190728_vacuum_hlv_gwtc21psd"
       for e in (5, 6, 7, 8) for s in ("", "_seed1")},
    "gw190728_env_hlv_gwtc21psd_phase_sampled_sr_eq9_taud1e6_rho2e8":
        "gw190728_vacuum_hlv_gwtc21psd_phase_sampled",
}
VARIANTS = {  # label fragment -> how that run differs from the baseline
    "_hlv_gwtc21psd": dict(detectors=HLV, psd_source="gwtc21"),
    "_h1l1_gwtc21psd": dict(psd_source="gwtc21"),
    "_hlv": dict(detectors=HLV),
    "_only_H1": dict(detectors=("H1",)),
    "_only_L1": dict(detectors=("L1",)),
    "_psd8s": dict(psd_fftlength=8),
    "_psd16s": dict(psd_fftlength=16),
    "_imrphenompv2": dict(approximant="IMRPhenomPv2"),
    "_imrphenomxp": dict(approximant="IMRPhenomXP"),
    "_phase_sampled": dict(phase_marginalization=False),
}


def spec(label):
    """(model, fragments) that fix the exact likelihood of a run: every
    VARIANTS fragment in its label, except one inside a longer match
    ("_hlv" within "_hlv_gwtc21psd")."""
    model = "vacuum" if label.startswith(VACUUM) else "environment"
    found = []
    for fragment in VARIANTS:
        if fragment in label and not any(fragment in f for f in found):
            found.append(fragment)
    return model, tuple(found)


def likelihood(model, fragments):
    """Exact likelihood of a run.  Its distance table lives apart from
    bilby's default file, which the injection campaign reads meanwhile."""
    changes = {k: v for f in fragments for k, v in VARIANTS[f].items()}
    config = RunConfig(label="_", model=model, **changes,
                       prior_kwargs=dict(rho_max_gcm3=2e8) if model == "environment" else {})
    return exact_likelihood(config, distance_lookup(config.phase_marginalization))


def distance_lookup(phase_marginalization):
    return f"data_cache/distance_lookup_{'phase' if phase_marginalization else 'nophase'}.npz"


def done(label):
    return (OUTDIR / label / f"{label}_result.json").exists()


PAIRS = {e: v for e, v in PAIRS.items() if done(e) and done(v)}   # usable mid-campaign
LABELS = sorted(set(PAIRS) | set(PAIRS.values()))
EXACT = {s: likelihood(*s) for s in sorted({spec(l) for l in LABELS})}
OUTPUT = Path("reweighted_lnb.json")
STORED = json.loads(OUTPUT.read_text())["runs"] if OUTPUT.exists() else {}


def exact_lnl(task):
    key, params = task
    return float(EXACT[key].log_likelihood_ratio(params))


def reweight(label, pool):
    result = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json"))
    if STORED.get(label, {}).get("ln_z") == result.log_evidence:   # unchanged run
        return STORED[label]
    rng = np.random.default_rng(zlib.crc32(label.encode()))   # same draws in any run order
    post = result.posterior
    rows = post.iloc[rng.choice(len(post), min(N_SAMPLES, len(post)), replace=False)]
    key = spec(label)
    fixed = dict(phase=0.0) if EXACT[key].phase_marginalization else {}
    tasks = [(key, {**r.to_dict(), **fixed}) for _, r in rows.iterrows()]
    delta = np.array(pool.map(exact_lnl, tasks)) - rows["log_likelihood"].to_numpy()
    correction = logsumexp(delta) - np.log(delta.size)
    boot = [logsumexp(rng.choice(delta, delta.size)) - np.log(delta.size) for _ in range(500)]
    return dict(ln_z=result.log_evidence, ln_z_err=result.log_evidence_err,
                correction=float(correction), correction_err=float(np.std(boot)),
                delta_mean=float(delta.mean()), delta_sd=float(delta.std()))


if __name__ == "__main__":
    offset = mass_cut_offset()
    runs = {}
    with mp.get_context("fork").Pool(NPOOL) as pool:
        for label in LABELS:
            runs[label] = r = reweight(label, pool)
            print(f"{label:50s} ln(L_exact/L_RB): mean {r['delta_mean']:+.3f} "
                  f"sd {r['delta_sd']:.3f}  ->  ln Z correction "
                  f"{r['correction']:+.3f} +- {r['correction_err']:.3f}", flush=True)

    print("\nln B, relative binning vs exact likelihood (mass-cut offset included)")
    lnb = {}
    for env, vac in PAIRS.items():
        e, v = runs[env], runs[vac]
        raw = e["ln_z"] - v["ln_z"] + offset
        exact = raw + e["correction"] - v["correction"]
        err = np.sqrt(e["ln_z_err"]**2 + v["ln_z_err"]**2
                      + e["correction_err"]**2 + v["correction_err"]**2)
        lnb[env] = dict(vacuum=vac, relative_binning=raw, exact=exact, err=float(err))
        print(f"  {env:50s} {raw:+6.2f}  ->  {exact:+6.2f} +- {err:.2f}")
    OUTPUT.write_text(json.dumps(dict(runs=runs, lnb=lnb), indent=2))
    print(f"{OUTPUT} written")
