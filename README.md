# Scalar-field environment in GW190728: an independent reanalysis

Code and results for

> Duong Ngoc Khoa, *Robustness of Scalar Dark Matter Signatures in GW190728: An Independent
> Reanalysis with Background Calibration* (2026), arXiv: TBU.

Roy et al. ([Phys. Rev. Lett. 136, 191402 (2026)](https://arxiv.org/abs/2510.17967)) reported
tentative evidence, ln B ≈ 3.5, for a scalar-field environment around the binary black hole
GW190728. In their model, rotating black holes grow a cloud of ultralight bosons by
superradiance. The companion ionizes the cloud, which drains orbital energy faster than
gravitational radiation alone. The extra torque enters the gravitational-wave phase as a term
that grows like f^(−37/6) toward low frequency.

This repository reimplements their model on top of IMRPhenomXPHM and reanalyses the event with
bilby. It also calibrates the Bayes factor against vacuum signals injected into real detector
noise.

## Main results

ln B of the environmental model against vacuum, at a density bound of 2×10⁸ g cm⁻³, for the
full (not relative-binning) likelihood:

| Analysis | Superradiance prior, τ_d = 10⁶ yr | Agnostic prior |
|---|---|---|
| Roy et al. | 3.5 | 0.4 (bound not stated) |
| H1+L1+V1, on-source BayesWave spectra, as in GWTC-2.1 | +2.86 ± 0.24 | +1.12 ± 0.24 |
| the same, coalescence phase sampled | +2.47 ± 0.25 | +0.75 ± 0.24 |
| H1+L1, off-source Welch spectra (baseline) | +0.12 ± 0.23 | −1.12 ± 0.10 |

- The noise-spectrum dependence arises below 50 Hz, where the dephasing lives and two standard
  spectral estimates differ by up to 70%.
- Virgo carries 3% of the power of the environmental term, yet adding it raises ln B by up to
  1.4. That is more than it raises ln B for any of 49 vacuum signals injected into the
  surrounding data.
- Analytic phase marginalisation, approximate for precessing, higher-multipole waveforms,
  inflates ln B by 0.4–0.6.
- In the baseline configuration GW190728 sits at the 78th percentile of the vacuum background
  (P = 0.22 ± 0.06).
- The superradiance prior applied to the most environment-like vacuum injection gives
  ln B = +3.26.

## Layout

```
scalar_env/       the package: waveform model, data, priors, inference, robustness, injections
scripts/          numbered analysis scripts (table below)
scripts/checks/   the ad-hoc checks behind specific statements of the paper (see its README)
results/          the paper's data products, posterior samples and figures (see its README)
```

## Requirements

Linux or macOS; on Windows, use WSL2. LALSuite has no native Windows build.

```
pip install -r requirements.txt
```

The code was tested with Python 3.14.4 and the versions pinned in `requirements.txt`. The
sampler comparison also needs nessai, which installs PyTorch, and PyMultiNest, which needs the
MultiNest library built from source. Nothing uses a GPU. The samplers run in parallel over
processes, so set `OMP_NUM_THREADS=1` (and likewise `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS`) to stop each process from also starting several BLAS threads.

## Data

- **Strain** is downloaded from the [Gravitational Wave Open Science Center](https://gwosc.org)
  on first use, so this needs an internet connection, and is cached locally.
- **The catalogue's noise spectra and posterior** come from the GWTC-2.1 parameter-estimation
  release (Zenodo record [6513631](https://zenodo.org/records/6513631)). Download
  `IGWN-GWTC2p1-v2-GW190728_064510_PEDataRelease_mixed_cosmo.h5` into `data_cache/`.

## Reproducing the paper

Run every script from the repository root. Runs are cached by label in `outdir/<label>/` and
reloaded once finished, so any script can be interrupted and rerun. Result files (`*.json`) are
written to the current directory.

Running times:
- one nested-sampling run takes 1–8 h on 10 cores;
- all the GW190728 runs take about two weeks;
- each injection campaign takes about ten days;
- scripts without sampling take minutes.

Script 17 reweights every finished run to the full likelihood. Scripts 11, 14 and 15 read the
results and do no sampling.

| Script | What it does | Paper |
|---|---|---|
| `00_validate_waveform.py` | the environmental dephasing (no data, no LALSuite needed) | Fig. 1 |
| `01_vacuum_gw190728.py` | vacuum analysis, compared with GWTC-2.1 | Sec. V.A, Fig. 2 |
| `02_environmental_gw190728.py` | environmental analysis, agnostic and superradiance priors | Secs. V.B–C |
| `04_robustness_suite.py` | samplers, waveforms, noise spectra, networks, priors | Sec. V.D, Table III |
| `05_injection_study.py` | vacuum signals injected into off-source noise | Secs. IV.F, V.F, App. B |
| `06_sr_injection_subset.py` | the superradiance prior on 13 of the injections | Sec. V.F |
| `07_check_relative_binning.py`, `12_relbin_vs_full.py`, `13_shared_fiducial_check.py` | accuracy and reference point of relative binning | Secs. IV.B, V.C |
| `08_evidence_normalisation.py` | the truncation identity ln(P/f) | Sec. V.C, Table II |
| `09_inj028_dataquality.py` | the glitch segment of injection 28 | Sec. VI.C, Fig. 8 |
| `10_sr_evidence_check.py`, `16_bound_envelope.py`, `18_synthetic_identity_check.py` | superradiance evidences: reruns, density bounds, synthetic test | Sec. V.C |
| `11_paper_figures.py` | figures | Figs. 3–7 |
| `14_injection_table.py` | the table of injections | App. B |
| `15_paper_numbers.py` | Bayes factors quoted in the text | |
| `17_full_likelihood_reweight.py` | full-likelihood evidence of every run, by reweighting | all tables |
| `19_phase_marginalisation_check.py` | runs with the coalescence phase sampled | Sec. V.E |
| `20_catalogue_noise_runs.py`, `29_catalogue_phase_sampled.py` | the catalogue's network and noise spectra; the same with the phase sampled | Sec. V.E |
| `21_bound_scan_runs.py` | superradiance runs at the most favourable density bounds | Sec. V.C |
| `22_injection_comparison.py` | the two injection campaigns compared | Sec. VI.E |
| `23_detector_decomposition.py` | the gain in fit, detector by detector | Sec. V.E, Table VI |
| `24_network_psd_matrix.py`, `25_catalogue_sr_table.py` | sampled runs for the network × spectrum table; unfinished, so Table IV uses `checks/noise_reweight.py` | |
| `26_phase_marginalisation_ratio.py` | numerical phase marginalisation; exploratory, superseded by scripts 19 and 29 | |
| `27_rolloff_check.py` | the window's roll-off | Sec. V.E |
| `28_psd_band_attribution.py` | the noise spectrum swapped band by band | Sec. V.E, Table V |
| `30_injections_with_virgo.py` | Virgo and catalogue-spectrum null tests on the injections; their full-likelihood background | Secs. V.E–F |

## Citation

If you use this code or its results, please cite the paper above.

## License

MIT; see `LICENSE`.

## Acknowledgements

This research has made use of data or software obtained from the Gravitational Wave Open
Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the Virgo
Collaboration, and KAGRA. The analysis uses bilby, dynesty, nessai, PyMultiNest, LALSuite,
GWpy, PESummary, NumPy, SciPy, pandas and Matplotlib.
