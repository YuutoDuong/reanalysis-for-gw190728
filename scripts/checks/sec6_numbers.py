"""Every number Section VI quotes, from the stored results (no sampling)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import bilby
bilby.core.utils.logger.setLevel("ERROR")
from scalar_env.inference import RunConfig, build_priors, savage_dickey_lnb
from scalar_env.waveform import scalar_mass_ev

TRIG = 1248331528.5
OUT = Path("outdir")
res = lambda l: bilby.result.read_in_result(str(OUT / l / f"{l}_result.json"))
q = lambda x, p=(0.05, 0.5, 0.95): np.quantile(np.asarray(x), p)
fmt = lambda x: "{:.3g} [{:.3g}, {:.3g}]".format(x[1], x[0], x[2])

print("=== A. vacuum")
vac = res("gw190728_vacuum")
print(f"lnZ_vac {vac.log_evidence:.2f} +/- {vac.log_evidence_err:.2f}; "
      f"lnZ_noise {vac.log_noise_evidence:.2f}; lnBF_sig {vac.log_bayes_factor:.2f}")
for f in sorted(glob.glob("outdir/gw190728_vacuum_seed*/*_result.json")):
    r = bilby.result.read_in_result(f)
    print("  seed run", Path(f).parent.name, f"{r.log_evidence:.2f}")
vp = vac.posterior
for c in ("chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance"):
    print(f"  vac {c:20s} {fmt(q(vp[c]))}")

print("\n=== B. agnostic environmental, rho_max = 2e8")
AGN = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2", "gw190728_env_rho_max2e8"]
lnbs = []
for l in AGN:
    r = res(l); lnbs.append(r.log_evidence - vac.log_evidence)
    print(f"  {l:28s} lnZ {r.log_evidence:.2f} +/- {r.log_evidence_err:.2f}  lnB {lnbs[-1]:+.2f}")
print(f"  mean lnB {np.mean(lnbs):+.2f}, sd {np.std(lnbs, ddof=1):.2f}, sem {np.std(lnbs, ddof=1)/2:.2f}")
post = pd.concat([res(l).posterior for l in AGN], ignore_index=True)
for c in ("alpha_cloud", "rho_phi", "chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance", "redshift"):
    print(f"  env {c:20s} {fmt(q(post[c]))}")
print(f"  mu_s source {fmt(q(scalar_mass_ev(post.alpha_cloud, post.total_mass_source)))} eV")
print(f"  alpha/(1+q) {fmt(q(post.alpha_cloud / (1 + post.mass_ratio)))}")
for thr in (1e5, 1e6):
    print(f"  P(rho < {thr:.0e}) = {np.mean(post.rho_phi < thr):.3f}")
print(f"  max lnL env {post.log_likelihood.max():.2f}, vac {vp.log_likelihood.max():.2f}, "
      f"gain {post.log_likelihood.max() - vp.log_likelihood.max():+.2f}")
pri = build_priors(RunConfig(label="_", model="environment",
                             prior_kwargs=dict(rho_max_gcm3=2e8)), TRIG)
for l in AGN[:1]:
    print(f"  Savage-Dickey (baseline run) {savage_dickey_lnb(res(l), pri):+.2f}")
comb = res(AGN[0]); comb.posterior = post
print(f"  Savage-Dickey (4 runs combined) {savage_dickey_lnb(comb, pri):+.2f}")

print("\n=== B2. other boxes")
for l in ("gw190728_env_rho_max1e6", "gw190728_env_rho_max1e7", "gw190728_env_rho_max1e8"):
    r = res(l); print(f"  {l:28s} lnB {r.log_evidence - vac.log_evidence:+.2f} "
                      f"+/- {np.hypot(r.log_evidence_err, vac.log_evidence_err):.2f}  "
                      f"max lnL {r.posterior.log_likelihood.max():.2f}")

print("\n=== C. robustness_lnb.json")
for lab, b, e in json.loads(Path("robustness_lnb.json").read_text()):
    print(f"  {lab:42s} {b:+.2f} +/- {e:.2f}")

print("\n=== C2. SR posteriors, rho_max = 2e8")
for e in (5, 6, 7, 8):
    l = f"gw190728_env_sr_eq9_taud1e{e}_rho2e8"; p = res(l).posterior
    print(f"  tau_d 1e{e}: rho {fmt(q(p.rho_phi))}  alpha {fmt(q(p.alpha_cloud))}  "
          f"Mc {fmt(q(p.chirp_mass))}  chi_eff {fmt(q(p.chi_eff))}  "
          f"mu_s {fmt(q(scalar_mass_ev(p.alpha_cloud, p.total_mass_source)))}")

print("\n=== D. background")
inj = json.loads(Path("injection_lnb.json").read_text())
v = np.array([inj[k]["lnb"] for k in sorted(inj)]); keys = sorted(inj)
clean = np.array([x for k, x in zip(keys, v) if k != "inj028"])
print(f"  N {len(v)}; median {np.median(v):+.2f}; 5/95% {q(v, (0.05, 0.95))}; "
      f"clean min/max {clean.min():+.2f} {clean.max():+.2f}; inj028 {inj['inj028']['lnb']:+.2f}")
for thr in (-1.29, 0.0, 1.0, 3.5):
    n = int((v >= thr).sum()); nc = int((clean >= thr).sum())
    print(f"  P(lnB >= {thr:+.2f}) = {n}/{len(v)} = {n/len(v):.3f} +/- "
          f"{np.sqrt(n/len(v)*(1-n/len(v))/len(v)):.3f};  clean {nc}/{len(clean)}")
print("  sorted top 6:", sorted(zip(v, keys))[-6:])
gains = {}
for k in keys:
    try:
        e = json.load(open(f"outdir/{k}_env/{k}_env_result.json"))["posterior"]["content"]["log_likelihood"]
        w = json.load(open(f"outdir/{k}_vac/{k}_vac_result.json"))["posterior"]["content"]["log_likelihood"]
        gains[k] = max(e) - max(w)
    except Exception as exc:
        print("  gain failed", k, type(exc).__name__)
g = np.array([gains[k] for k in keys if k in gains and k != "inj028"])
print(f"  best-fit gain (clean {len(g)}): median {np.median(g):+.2f}, 5/95% {q(g, (0.05, 0.95))}, "
      f"max {g.max():+.2f}; inj028 {gains.get('inj028', float('nan')):+.1f}")
gw_gain = post.log_likelihood.max() - vp.log_likelihood.max()
print(f"  P(gain >= GW190728's {gw_gain:+.2f}) = {np.mean(g >= gw_gain):.2f}")

print("\n=== E. SR subset")
sr = json.loads(Path("injection_sr_lnb.json").read_text())
for k in sorted(sr, key=lambda k: -sr[k]["lnb_agnostic"]):
    print(f"  {k}: agnostic {sr[k]['lnb_agnostic']:+7.2f}  SR {sr[k]['lnb_sr']:+6.2f} +/- {sr[k]['err']:.2f}  "
          f"diff {sr[k]['lnb_sr'] - sr[k]['lnb_agnostic']:+.2f}")
d = np.array([sr[k]["lnb_sr"] - sr[k]["lnb_agnostic"] for k in sr if k != "inj028"])
print(f"  clean diffs: median {np.median(d):+.2f}, range {d.min():+.2f} to {d.max():+.2f}")
