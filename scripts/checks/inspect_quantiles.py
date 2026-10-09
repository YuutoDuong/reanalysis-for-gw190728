"""Which posterior do the quoted medians/90% intervals come from?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import bilby
from scalar_env.inference import OUTDIR
from scalar_env.waveform import scalar_mass_ev

bilby.core.utils.logger.setLevel("ERROR")
load = lambda l: bilby.result.read_in_result(str(OUTDIR / l / f"{l}_result.json")).posterior
BASE = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2", "gw190728_env_rho_max2e8"]


def q90(x):
    lo, mid, hi = np.quantile(x, [0.05, 0.5, 0.95])
    return f"{mid:.4g} +{hi - mid:.3g} -{mid - lo:.3g}"


sets = {"vacuum (single)": load("gw190728_vacuum"),
        "env (single)": load("gw190728_env"),
        "env (4 runs)": pd.concat([load(l) for l in BASE], ignore_index=True)}
for name, p in sets.items():
    row = f"{name:16s} Mc {q90(p['chirp_mass'])}   chi_eff {q90(p['chi_eff'])}"
    if "alpha_cloud" in p:
        mu = scalar_mass_ev(p["alpha_cloud"], p["total_mass_source"]) / 1e-13
        row += (f"   alpha {q90(p['alpha_cloud'])}   rho/1e7 {q90(p['rho_phi'] / 1e7)}"
                f"   m_phi/1e-13 {q90(mu)}")
    print(row)
