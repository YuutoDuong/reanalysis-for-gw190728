"""Roy et al.'s superradiance table in the catalogue noise model.

With the GWTC-2.1 noise model (H1+L1+V1, BayesWave PSDs) script 20 found
ln B = 2.84 at their headline delay time, 1e6 yr, against the 3.5 they
report.  Repeat that run with a second seed, and run the other three
delay times, so that their whole table can be compared in the
configuration they most likely used.  All at rho_max = 2e8.

Runs after script 24, on 4 cores at low priority: ~1-2 days.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig, run_pe, ln_bayes_factor, mass_cut_offset

NPOOL = 4
NOISE = dict(detectors=("H1", "L1", "V1"), psd_source="gwtc21", npool=NPOOL)
ROY = {5: 3.4, 6: 3.5, 7: 3.5, 8: 2.8}   # log10(tau_d / yr): their ln B


def superradiance(exponent):
    return RunConfig(label=f"gw190728_env_hlv_gwtc21psd_sr_eq9_taud1e{exponent}_rho2e8",
                     model="environment", **NOISE,
                     prior_kwargs=dict(rho_max_gcm3=2e8, superradiance_tau_d_yr=10.0**exponent))


if __name__ == "__main__":
    vac = run_pe(RunConfig(label="gw190728_vacuum_hlv_gwtc21psd", model="vacuum", **NOISE))
    runs = [("1e6, seed 1", 6, superradiance(6).variant("seed1", seed=1))]
    runs += [(f"1e{e}", e, superradiance(e)) for e in ROY]
    offset = mass_cut_offset()
    print(f"\nH1+L1+V1, GWTC-2.1 PSDs, superradiance, rho_max = 2e8"
          f"\n{'tau_d [yr]':14s}{'ln B':>16s}{'Roy et al.':>12s}")
    for name, exponent, config in runs:
        lnb, err = ln_bayes_factor(run_pe(config), vac)
        print(f"{name:14s}{lnb + offset:+10.2f} +- {err:.2f}{ROY[exponent]:12.1f}", flush=True)
