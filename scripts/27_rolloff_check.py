"""Does the window's roll-off decide the Bayes factor?

With bilby's default 0.2 s rise, the Tukey window leaks noise from below
20 Hz into the lowest analysed bins: L1's 8-s segments carry 1.3-1.6
times the Welch PSD at 20-30 Hz, on and off source alike, against 1.04
with a 0.4 s rise (the bilby_pipe default, as in the catalogue analyses).
The environmental dephasing lives in exactly those bins.

No new sampling is needed: the evidence for the 0.4 s data follows from
the posterior samples of the 0.2 s runs,

    Z_0.4 / Z_RB = E_post[ L_exact,0.4 / L_RB ],

reliable when ln(L_exact,0.4 / L_RB) scatters by well under 1 (printed).
The same estimate on 0.2 s data reproduces script 17.  ~5 min.
"""

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
ROLL_OFFS = (0.2, 0.4)
VACUUM = "gw190728_vacuum"
AGNOSTIC = ["gw190728_env", "gw190728_env_seed1", "gw190728_env_seed2",
            "gw190728_env_rho_max2e8"]
SR = "gw190728_env_sr_eq9_taud1e6_rho2e8"
EXACT = {(model, roll): exact_likelihood(
             RunConfig(label="_", model=model, roll_off=roll,
                       prior_kwargs=dict(rho_max_gcm3=2e8) if model == "environment" else {}),
             "data_cache/distance_lookup_phase.npz")   # the campaign reads bilby's default
         for model in ("vacuum", "environment") for roll in ROLL_OFFS}


def exact_lnl(task):
    key, params = task
    return float(EXACT[key].log_likelihood_ratio(params))


def ln_z(label, roll, pool):
    """(exact ln Z on data windowed with this roll-off, scatter of the log-weights)."""
    rng = np.random.default_rng(zlib.crc32(label.encode()))
    result = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json"))
    post = result.posterior
    rows = post.iloc[rng.choice(len(post), N_SAMPLES, replace=False)]
    model = "environment" if "alpha_cloud" in post else "vacuum"
    tasks = [((model, roll), {**r.to_dict(), "phase": 0.0}) for _, r in rows.iterrows()]
    delta = np.array(pool.map(exact_lnl, tasks)) - rows["log_likelihood"].to_numpy()
    return result.log_evidence + logsumexp(delta) - np.log(delta.size), delta.std()


if __name__ == "__main__":
    offset = mass_cut_offset()
    with mp.get_context("fork").Pool(NPOOL) as pool:
        for roll in ROLL_OFFS:
            vac, _ = ln_z(VACUUM, roll, pool)
            runs = {l: ln_z(l, roll, pool) for l in AGNOSTIC + [SR]}
            lnb = {l: z - vac + offset for l, (z, _) in runs.items()}
            spread = max(s for _, s in runs.values())
            print(f"roll-off {roll} s: agnostic ln B {np.mean([lnb[l] for l in AGNOSTIC]):+.2f} "
                  f"(4 runs), superradiance tau_d = 1e6 {lnb[SR]:+.2f}; "
                  f"log-weight scatter <= {spread:.2f}", flush=True)
