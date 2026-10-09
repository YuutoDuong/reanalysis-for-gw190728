"""Section V.A -- Vacuum PE reproduction on GW190728 (pipeline validation).

Runs standard IMRPhenomXPHM PE with relative binning, then overlays the
posterior on the published GWTC-3 samples.  The pipeline is trusted for
environmental runs only if this comparison passes.

Expected wall time: ~6-24 h (laptop, 1000 live points, relative binning).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.inference import RunConfig, run_pe, seed_variants, evidence_scatter
from scalar_env.analysis import compare_with_catalog

# Leave 1-2 cores free for the OS/WSL; each worker uses 1 BLAS thread (see
# the OMP/OPENBLAS/MKL_NUM_THREADS=1 exports in the README) so npool workers
# never oversubscribe the machine.
NPOOL = max(1, (os.cpu_count() or 2) - 2)

config = RunConfig(label="gw190728_vacuum", model="vacuum", npool=NPOOL)
result = run_pe(config)
print(f"ln Z_vac = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")

# Convergence check (Section IV.C): ln Z scatter across seeds must be < 0.5.
mean_lnz, std_lnz = evidence_scatter(seed_variants(config, n_seeds=3))
print(f"ln Z across seeds: {mean_lnz:.2f} +/- {std_lnz:.2f} "
      f"({'OK' if std_lnz < 0.5 else 'NOT CONVERGED -- increase nlive'})")

# Comparison with the published catalog posterior.  GW190728 is an O3a
# event, so it is in the GWTC-2.1 data release (Zenodo record 6513631):
#   wget -P data_cache https://zenodo.org/records/6513631/files/IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_cosmo.h5
catalog_file = Path("data_cache/IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_cosmo.h5")
if catalog_file.exists():
    from pesummary.io import read
    catalog = read(str(catalog_file)).samples_dict["C01:IMRPhenomXPHM"]
    compare_with_catalog(result, catalog)
    print("figures/vacuum_vs_gwtc.png written -- check for overlap before "
          "proceeding to the environmental analysis.")
else:
    print(f"(GWTC-3 samples not found at {catalog_file}; corner overlay skipped)")
