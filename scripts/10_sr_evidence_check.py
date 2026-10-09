"""Why do the direct superradiance evidences fall short of the identity?

Script 08 found ln B_SR - ln B_agn below the exact prediction ln(P/f) by
up to 0.93 at rho_max = 2e8 (tau_d = 1e6: measured +1.26, predicted
+2.19), although each SR posterior matches the agnostic posterior
truncated to its region.  The deficit therefore sits in the evidence
estimates of the constrained runs, not in their exploration.  Two
reruns of that case isolate the cause:

  nessai   an independent algorithm (normalising flows, no MCMC chains).
           bilby normalises its constrained prior explicitly
           (PriorDict.ln_prob), so the evidence is directly comparable.
  dynesty  Roy et al.'s setting, acceptance-walk with naccept = 60,
           against our ~4 accepted steps per chain (rwalk, nact = 2).

The prediction for each is its own agnostic ln B plus ln(P/f).  The
agnostic baseline is itself cross-validated by three samplers (dynesty,
MultiNest, nessai agree to 0.16), so a run landing on the prediction
means our short-chain SR evidences were biased low.  MultiNest is not
used here: bilby gives it the constraints only through the likelihood,
so its normalisation under a strong truncation is not comparable.

Expected wall time: nessai ~2 h, dynesty acceptance-walk ~1 day.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig, run_pe, ln_bayes_factor

NPOOL = max(1, (os.cpu_count() or 2) - 2)
SR = dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=1e6)
LN_P_OVER_F = 2.19        # script 08, tau_d = 1e6, rho_max = 2e8
DIRECT_RWALK = -0.03      # script 04, gw190728_env_sr_eq9_taud1e6_rho2e8

env = RunConfig(label="gw190728_env", model="environment", npool=NPOOL,
                prior_kwargs=dict(rho_max_gcm3=2e8))
vac = RunConfig(label="gw190728_vacuum", model="vacuum", npool=NPOOL)

# (name, SR run, its agnostic reference ln B, matching vacuum run)
nessai_vac = vac.variant("nessai", sampler="nessai")
checks = [
    ("nessai",
     env.variant("sr_eq9_taud1e6_rho2e8_nessai", sampler="nessai",
                 prior_kwargs=SR),
     ln_bayes_factor(run_pe(env.variant("nessai", sampler="nessai")),
                     run_pe(nessai_vac))[0],
     nessai_vac),
    ("dynesty, acceptance-walk 60",
     env.variant("sr_eq9_taud1e6_rho2e8_accwalk60", prior_kwargs=SR,
                 sampler_kwargs=dict(sample="acceptance-walk", naccept=60)),
     -1.29,                # mean of the four baseline rwalk runs
     vac),
]

print(f"tau_d = 1e6 yr, rho_max = 2e8.  Direct rwalk SR ln B = "
      f"{DIRECT_RWALK:+.2f}; identity predicts agnostic + {LN_P_OVER_F:.2f}.\n")
for name, sr_cfg, agnostic, vac_cfg in checks:
    lnb, err = ln_bayes_factor(run_pe(sr_cfg), run_pe(vac_cfg))
    predicted = agnostic + LN_P_OVER_F
    print(f"{name:28s} SR ln B = {lnb:+.2f} +/- {err:.2f}   predicted "
          f"{predicted:+.2f}   residual {lnb - predicted:+.2f}", flush=True)
