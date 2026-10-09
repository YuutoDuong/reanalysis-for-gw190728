"""How much can the unpublished density bound change ln B?  (No sampling.)

Roy et al. do not state rho_max.  Every bound below ours is a box
truncation of the 2e8 runs, so the truncation identity gives

    ln B(rho_max) = ln B(2e8) + ln(P / f),

with P (f) the fraction of the 2e8 posterior (prior) that the smaller box
keeps.  For the superradiance prior the 2e8 baseline is either the direct
superradiance run (its own posterior) or the agnostic run restricted to the
window (the identity value, our upper envelope).  Above 2e8 the posterior
has no support, so ln B can only fall.  The maximum over rho_max is the
largest Bayes factor that any choice of bound could have produced.

Writes bound_envelope.json and figures/bound_envelope_paper.{pdf,png}.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import bilby
from scalar_env.inference import RunConfig, build_priors, mass_cut_offset, OUTDIR

bilby.core.utils.logger.setLevel("WARNING")
TRIGGER = 1248331528.5
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
        "gw190728_env_rho_max2e8"]
ROY = {"agnostic": 0.4, 1e5: 3.4, 1e6: 3.5, 1e7: 3.5, 1e8: 2.8}
GRID = np.logspace(5, np.log10(2e8), 80)

OFFSET = mass_cut_offset()
LNB = {l: b + OFFSET for l, b, _ in json.loads(Path("robustness_lnb.json").read_text())}
priors = lambda **kw: build_priors(RunConfig(label="_", model="environment",
                                             prior_kwargs=dict(rho_max_gcm3=2e8, **kw)),
                                   TRIGGER)
inside = lambda pri, df: np.asarray(pri.evaluate_constraints(
    {k: np.asarray(df[k]) for k in KEYS}), dtype=bool)
posterior = lambda *labels: pd.concat([bilby.result.read_in_result(
    str(OUTDIR / l / f"{l}_result.json")).posterior for l in labels], ignore_index=True)

np.random.seed(3)
agn_pri = priors()
draws = {k: np.asarray(agn_pri[k].sample(2_000_000)) for k in KEYS}
agn_prior_ok = inside(agn_pri, draws)
agn_post = posterior(*BASE)
lnb_agn = np.mean([LNB[l] for l in BASE])


def curve(post_rho, prior_ok, lnb_ref, post_keep=None, prior_ref=None):
    """ln B(rho_max) = lnb_ref + ln(P/f) on GRID."""
    post_keep = np.ones(len(post_rho), bool) if post_keep is None else post_keep
    prior_ref = prior_ok if prior_ref is None else prior_ref
    out = []
    for rmax in GRID:
        p = np.mean(post_keep & (post_rho < rmax))
        f = np.mean(prior_ok & (draws["rho_phi"] < rmax)) / np.mean(prior_ref)
        out.append(lnb_ref + np.log(p / f) if p > 0 else np.nan)
    return np.array(out)


curves = {"agnostic": curve(agn_post["rho_phi"].to_numpy(), agn_prior_ok, lnb_agn)}
for tau in (1e5, 1e6, 1e7, 1e8):
    sr_pri = priors(superradiance_tau_d_yr=tau)
    sr_ok = inside(sr_pri, draws)
    label = f"gw190728_env_sr_eq9_taud1e{int(np.log10(tau))}_rho2e8"
    sr_post = posterior(label)
    curves[f"SR {tau:.0e} direct"] = curve(sr_post["rho_phi"].to_numpy(), sr_ok, LNB[label])
    curves[f"SR {tau:.0e} identity"] = curve(agn_post["rho_phi"].to_numpy(), sr_ok, lnb_agn,
                                             post_keep=inside(sr_pri, agn_post),
                                             prior_ref=agn_prior_ok)

summary = {}
print(f"mass-cut offset {OFFSET:+.3f}; bounds scanned 1e5 .. 2e8 g/cm^3\n")
for name, y in curves.items():
    i = np.nanargmax(y)
    at = lambda r: float(np.interp(np.log(r), np.log(GRID), y))
    summary[name] = dict(max=float(y[i]), argmax=float(GRID[i]), at_1e7=at(1e7),
                         at_2e8=float(y[-1]))
    print(f"{name:22s} max {y[i]:+.2f} at rho_max = {GRID[i]:.1e}   "
          f"(1e7: {at(1e7):+.2f}, 2e8: {y[-1]:+.2f})")

y = curves["agnostic"]
for shift, text in ((0.0, "our alpha box"), (0.354, "Roy's alpha box [0.008, 0.5]")):
    hit = np.where(np.diff(np.sign(y + shift - ROY["agnostic"])))[0]
    found = [f"{GRID[j]:.1e}" for j in hit]
    print(f"agnostic ln B = +0.4 (Roy) with {text}: rho_max = {found}")
    summary[f"bound_for_roy_agnostic ({text})"] = found

Path("bound_envelope.json").write_text(json.dumps(dict(
    grid=GRID.tolist(), curves={k: v.tolist() for k, v in curves.items()},
    summary=summary), indent=2))

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot(GRID, curves["agnostic"], color="C0", label="agnostic")
ax.plot(GRID, curves["SR 1e+06 direct"], color="C3",
        label=r"superradiance, $\tau_d=10^6$ yr (direct)")
ax.plot(GRID, curves["SR 1e+06 identity"], color="C3", ls="--",
        label=r"superradiance, $\tau_d=10^6$ yr (identity)")
ax.axhline(ROY[1e6], color="C3", ls=":", lw=1)
ax.axhline(ROY["agnostic"], color="C0", ls=":", lw=1)
ax.text(GRID[0], ROY[1e6] + 0.08, "Roy et al., superradiance", fontsize=7, color="C3")
ax.text(GRID[0], ROY["agnostic"] + 0.08, "Roy et al., agnostic", fontsize=7, color="C0")
ax.set_xscale("log")
ax.set_xlabel(r"density bound $\rho_{\max}$ [g cm$^{-3}$]")
ax.set_ylabel(r"$\ln\mathcal{B}^{\rm env}_{\rm vac}$")
ax.legend(fontsize=7, loc="lower left")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/bound_envelope_paper.pdf", bbox_inches="tight")
fig.savefig("figures/bound_envelope_paper.png", dpi=200, bbox_inches="tight")
print("\nbound_envelope.json and figures/bound_envelope_paper.* written")
