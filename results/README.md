# Results

The paper's data products. ln B is ln Z_env − ln Z_vac. "Exact" means the full likelihood,
obtained by importance reweighting of the relative-binning runs (script 17). Where a file says
an ln B includes the mass-prior correction of Sec. IV.E, that is +0.180; raw values need it
added.

## GW190728

| File | Contents |
|---|---|
| `runs.csv` | one row per run of the paper. Settings: detectors, approximant, sampler, phase treatment, density bound and number of posterior samples. Evidences: ln Z (relative binning, raw) and its error, the correction to the full likelihood and its error. For environmental runs also the vacuum run it is compared with, and the exact ln B (correction included) with its error. |
| `reweighted_lnb.json` | the same evidences as script 17 writes them, with the scatter of the log-weights; every ln B includes the correction |
| `posteriors/<run>.csv.gz` | posterior samples, one row per sample; complex matched-filter SNRs are stored as moduli |
| `bound_envelope.json` | ln B against the density bound, from the truncation identity (script 16) |
| `shared_fiducial.json` | agnostic and superradiance runs sharing one relative-binning reference (script 13) |
| `logs/psd_band_attribution.log` | Table V (script 28) |
| `logs/detector_decomposition.log` | Table VI (script 23) |
| `logs/synthetic_test.log`, `logs/synthetic_test_samplers.log` | the synthetic test of Sec. V.C (script 18 and `scripts/checks/synth_samplers.py`) |

Run labels read `gw190728_<model>_<variants>`, where the model is `vacuum` or `env`. Each
environmental run is compared with a vacuum run with the same network, noise model, waveform and
phase treatment. `runs.csv` names that run in its `vacuum_run` column. The variants:

- `seed1`, `seed2`: repeats with other seeds.
- `rho_max<X>`, `rho<X>`: density bound X g cm⁻³ (`2p2e7` means 2.2×10⁷). `runs.csv` gives
  every run's bound.
- `sr_eq9_taud1e<N>`: the superradiance prior with delay time 10^N yr.
- `hlv`: H1+L1+V1. `only_H1`, `only_L1`: one detector.
- `gwtc21psd`: the GWTC-2.1 BayesWave noise spectra (`h1l1_gwtc21psd`: with H1+L1 only).
- `psd8s`, `psd16s`: Welch segment length (default 4 s).
- `imrphenompv2`, `imrphenomxp`: the vacuum approximant (default IMRPhenomXPHM).
- `phase_sampled`: the coalescence phase sampled instead of marginalised analytically.
- `nessai`, `pymultinest`: the sampler (default dynesty).
- `accwalk60`: dynesty's acceptance walk with 60 accepted steps.
- `log_alpha`: a log-uniform prior on the coupling α.
- `fixfid`: one fixed relative-binning reference, shared between runs.

## Injections

| File | Contents |
|---|---|
| `injection_lnb.json` | first campaign: 50 vacuum injections into off-source noise, with masses not redshifted (Sec. IV.F). Each injection's parameters and relative-binning ln B, raw. |
| `injection_sr_lnb.json` | the superradiance subset, 13 injections, raw |
| `injection_virgo.json` | each injection's exact ln B for three networks: H1+L1; H1+L1+V1; and H1+L1+V1 with the event's BayesWave spectra. Includes the correction, the log-weight scatter and the effective sample size (script 30). |
| `*_det.json` | the same for the second campaign, whose masses are redshifted |

## Figures

`figures/` holds the paper's figures, as PDF and PNG where both exist.
