"""Decisive check: do the dead points of the vacuum run respect m2 >= 5
(evidence normalised to the cut prior) while the environmental runs do not?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import bilby
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses as comp
from scalar_env.inference import OUTDIR

bilby.core.utils.logger.setLevel("ERROR")
for label in ("gw190728_vacuum", "gw190728_vacuum_seed1", "gw190728_vacuum_nessai",
              "gw190728_env", "gw190728_env_fixfid",
              "gw190728_env_fixfid_sr_eq9_taud1e6_rho2e8"):
    r = bilby.result.read_in_result(str(OUTDIR / label / f"{label}_result.json"))
    ns = r.nested_samples
    m1, m2 = comp(ns["chirp_mass"].to_numpy(), ns["mass_ratio"].to_numpy())
    first = slice(0, 1000)                      # lowest-likelihood dead points ~ initial prior draws
    post_m2 = r.posterior["mass_2"].to_numpy()
    print(f"{label:44s} dead pts m2<5: {np.mean(m2 < 5):.4f} (first 1000: {np.mean(m2[first] < 5):.3f})"
          f"   posterior m2<5: {np.mean(post_m2 < 5):.5f}")
