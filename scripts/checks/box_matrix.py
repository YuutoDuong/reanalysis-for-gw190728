"""Box-truncation identity between every pair of agnostic box runs:
predict ln B(small box) from the larger-box run's posterior and prior."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import bilby
from scalar_env.inference import RunConfig, build_priors, mass_cut_offset, OUTDIR
bilby.core.utils.logger.setLevel("ERROR")
KEYS = ("chirp_mass", "mass_ratio", "alpha_cloud", "rho_phi")
OFF = mass_cut_offset()
LNB = {l: b + OFF for l, b, _ in json.loads(Path("robustness_lnb.json").read_text())}
runs = {2e8: ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2", "gw190728_env_rho_max2e8"],
        1e8: ["gw190728_env_rho_max1e8"], 1e7: ["gw190728_env_rho_max1e7"],
        1e6: ["gw190728_env_rho_max1e6"]}
direct = {b: np.mean([LNB[l] for l in ls]) for b, ls in runs.items()}
np.random.seed(5)
for big, labels in runs.items():
    pri = build_priors(RunConfig(label="_", model="environment", prior_kwargs=dict(rho_max_gcm3=big)), 0.0)
    d = {k: np.asarray(pri[k].sample(2_000_000)) for k in KEYS}
    ok = np.asarray(pri.evaluate_constraints(d), bool)
    post = pd.concat([bilby.result.read_in_result(str(OUTDIR / l / f"{l}_result.json")).posterior
                      for l in labels], ignore_index=True)
    rho, a = post["rho_phi"].to_numpy(), post["alpha_cloud"].to_numpy()
    row = []
    for small in (1e8, 1e7, 3e6, 1e6, 3e5):
        if small >= big: continue
        p = np.mean(rho < small); f = np.mean(ok & (d["rho_phi"] < small)) / ok.mean()
        pred = direct[big] + np.log(p / f)
        tag = f"{direct[small]:+.2f}" if small in direct else "  -- "
        row.append(f"{small:.0e}: pred {pred:+.2f} (direct {tag}, n={int((rho < small).sum())})")
    print(f"from {big:.0e} (direct {direct[big]:+.2f}, N={len(rho)}): " + "; ".join(row))
    lo = rho < 1e6
    if lo.any():
        print(f"      alpha of samples with rho<1e6: median {np.median(a[lo]):.2f}, 90% [{np.quantile(a[lo],0.05):.2f}, {np.quantile(a[lo],0.95):.2f}]")
