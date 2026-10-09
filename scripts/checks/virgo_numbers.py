"""Numbers for the paper from the finished Virgo/PSD null test (JSON reads only)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from scalar_env.inference import mass_cut_offset

OFF = mass_cut_offset()
V = json.loads(Path("injection_virgo.json").read_text())
RB = json.loads(Path("injection_lnb.json").read_text())
SRRB = json.loads(Path("injection_sr_lnb.json").read_text())
LNB = json.loads(Path("reweighted_lnb.json").read_text())["lnb"]
GLITCH = "inj028"
BASE = np.mean([LNB[l]["exact"] for l in ("gw190728_env", "gw190728_env_seed1",
                                          "gw190728_env_seed2", "gw190728_env_rho_max2e8")])
clean = sorted(k for k in V if k != GLITCH)
print(f"entries {len(V)}, clean {len(clean)}, campaign {len(RB)}, offset {OFF:+.3f}, base {BASE:+.3f}")

print("\n== A. H1+L1 background, full grid (agnostic)")
a = np.array([V[k]["H1+L1"]["agnostic"]["lnb"] for k in clean])
ess = np.array([V[k]["H1+L1"]["agnostic"]["ess"] for k in clean])
rb = np.array([RB[k]["lnb"] + OFF for k in clean])
k_ = int((a >= BASE).sum()); n = a.size
print(f"median {np.median(a):+.2f}, 90% [{np.quantile(a, .05):+.2f}, {np.quantile(a, .95):+.2f}], "
      f"max {a.max():+.2f} ({clean[a.argmax()]})")
print(f"P(>= {BASE:+.2f}) = {k_}/{n} = {k_/n:.3f} +- {np.sqrt(k_*(n-k_)/n**3):.3f}; percentile {100*np.mean(a < BASE):.0f}")
print("top 6:", ", ".join(f"{clean[i]} {a[i]:+.2f} (RB {rb[i]:+.2f}, ESS {ess[i]:.0f})" for i in np.argsort(-a)[:6]))
d = a - rb
print(f"exact - RB: mean {d.mean():+.3f}, sd {d.std():.3f}, max |d| {np.abs(d).max():.2f} ({clean[np.abs(d).argmax()]}); "
      f"ESS min {ess.min():.0f}, median {np.median(ess):.0f}, N(ESS<30) {(ess < 30).sum()}")
rb_k = int((rb >= BASE).sum())
print(f"RB for comparison: median {np.median(rb):+.2f}, 90% [{np.quantile(rb, .05):+.2f}, {np.quantile(rb, .95):+.2f}], "
      f"max {rb.max():+.2f}, P = {rb_k}/{n}")
g = V[GLITCH]["H1+L1"]
print(f"glitch {GLITCH}: " + ", ".join(f"{p} {v['lnb']:+.1f} (ESS {v['ess']:.0f}, RB {(RB[GLITCH]['lnb'] if p == 'agnostic' else SRRB[GLITCH]['lnb_sr']) + OFF:+.1f})" for p, v in g.items()))

print("\n== B. superradiance subset, full grid")
rows = []
for k in sorted(V):
    h = V[k]["H1+L1"]
    if "superradiance" in h:
        rows.append((k, h["agnostic"]["lnb"], h["superradiance"]["lnb"], h["superradiance"]["ess"],
                     SRRB[k]["lnb_agnostic"] + OFF, SRRB[k]["lnb_sr"] + OFF))
for k, ag, sr, e, ag_rb, sr_rb in rows:
    print(f"  {k}: agnostic {ag:+6.2f} SR {sr:+6.2f} raise {sr-ag:+5.2f} (ESS {e:.0f}) | RB {ag_rb:+6.2f} {sr_rb:+6.2f} raise {sr_rb-ag_rb:+5.2f}")
raise_ = np.array([sr - ag for k, ag, sr, *_ in rows if k != GLITCH])
raise_rb = np.array([s - g_ for k, _, _, _, g_, s in rows if k != GLITCH])
print(f"clean {raise_.size}: raise median {np.median(raise_):+.2f}, max {raise_.max():+.2f} | RB median {np.median(raise_rb):+.2f}, max {raise_rb.max():+.2f}")
sr_clean = np.array([sr for k, ag, sr, *_ in rows if k != GLITCH])
gw_sr = LNB["gw190728_env_sr_eq9_taud1e6_rho2e8"]["exact"]
print(f"GW190728 SR (H1+L1, 2e8, tau 1e6) {gw_sr:+.2f}; clean SR values >= it: {(sr_clean >= gw_sr).sum()}/{sr_clean.size}; max {sr_clean.max():+.2f}")

print("\n== C. network shifts relative to H1+L1 (full grid)")
gw = {("H1+L1+V1", "agnostic"): LNB["gw190728_env_hlv"]["exact"] - BASE,
      ("H1+L1+V1, BayesWave", "agnostic"): LNB["gw190728_env_hlv_gwtc21psd"]["exact"] - BASE,
      ("H1+L1+V1", "superradiance"): LNB.get("gw190728_env_hlv_sr_eq9_taud1e6_rho2e8", {}).get("exact", np.nan) - gw_sr,
      ("H1+L1+V1, BayesWave", "superradiance"): LNB["gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8"]["exact"] - gw_sr}
for (net, prior), obs in gw.items():
    keys = [k for k in clean if prior in V[k].get(net, {})]
    s = np.array([V[k][net][prior]["lnb"] - V[k]["H1+L1"][prior]["lnb"] for k in keys])
    e = np.array([V[k][net][prior]["ess"] for k in keys])
    big = [f"{keys[i]} {s[i]:+.2f} (ESS {e[i]:.0f})" for i in np.argsort(-s)[:4]]
    print(f"{net:20s} {prior:13s} N {s.size}: median {np.median(s):+.2f}, 90% [{np.quantile(s, .05):+.2f}, {np.quantile(s, .95):+.2f}], "
          f"sd {s.std():.2f}; GW190728 {obs:+.2f}, k(>=) {(s >= obs).sum()}; ESS<30: {(e < 30).sum()}; largest: {', '.join(big)}")
    if prior == "agnostic":
        r = e >= 30
        print(f"{'':34s} ESS>=30 only: N {r.sum()}, median {np.median(s[r]):+.2f}, 95% {np.quantile(s[r], .95):+.2f}, max {s[r].max():+.2f}, k(>=GW) {(s[r] >= obs).sum()}")
obs = LNB["gw190728_env_hlv_gwtc21psd"]["exact"] - LNB["gw190728_env_hlv"]["exact"]
s = np.array([V[k]["H1+L1+V1, BayesWave"]["agnostic"]["lnb"] - V[k]["H1+L1+V1"]["agnostic"]["lnb"] for k in clean])
print(f"{'spectra only (Virgo in both)':34s} N {s.size}: median {np.median(s):+.2f}, sd {s.std():.2f}, "
      f"90% [{np.quantile(s, .05):+.2f}, {np.quantile(s, .95):+.2f}]; GW190728 {obs:+.2f}, k(>=) {(s >= obs).sum()}")
print("\n== D. GW190728 exact values used")
for l in ("gw190728_env_hlv", "gw190728_env_hlv_gwtc21psd", "gw190728_env_hlv_sr_eq9_taud1e6_rho2e8",
          "gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e6_rho2e8", "gw190728_env_sr_eq9_taud1e6_rho2e8",
          "gw190728_env_phase_sampled", "gw190728_env_hlv_gwtc21psd_phase_sampled",
          "gw190728_env_hlv_gwtc21psd_phase_sampled_sr_eq9_taud1e6_rho2e8"):
    v = LNB.get(l)
    print(f"  {l:62s} " + (f"exact {v['exact']:+.2f} +- {v['err']:.2f} (RB {v['relative_binning']:+.2f})" if v else "not reweighted yet"))
