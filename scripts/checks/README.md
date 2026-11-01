# Checks

These are the ad-hoc checks behind specific statements of the paper, kept as they were run. Run them from the repository root after the numbered scripts have filled `outdir/` and written the results files. Most only read results. `noise_reweight.py` and `psd_leak_test.py` evaluate likelihoods and take about an hour on 10 cores; `dominant_mode.py` takes ten minutes on three.

| Script | Statement it supports |
|---|---|
| `harmonics.py` | the harmonic-by-harmonic waveform model: the pieces add up to the full waveform, power and dephasing per harmonic, cost (Sec. III.B) |
| `dominant_mode.py` | what applying the dominant harmonic's correction to every harmonic costs in ln B, for GW190728 and for vacuum injections; `--det` for the second campaign's injections as loud as the event (Secs. III.B, V.D, V.F, VI.B) |
| `sec6_numbers.py` | posterior summaries under the exact likelihood, Savage–Dickey estimate, prior and posterior mass at low density, gains in log-likelihood (Secs. V.A–C, VI.B) |
| `roy_figures.py` | the prior and posterior curves of Roy et al.'s Figs. 3 and 6 read as numbers, next to ours; the density bound they did not quote; the fraction of cycles the environment removes (Secs. IV.D, VI.A) |
| `density_scale.py` | the Bayes factors and densities Roy et al. would obtain if their code needed N times our density for the same waveform (Sec. VI.A) |
| `configuration_fingerprints.py` | each configuration's Bayes factors, likelihood gain, chirp mass and effective spin next to the values Roy et al. report (Sec. VI.A) |
| `inspect_quantiles.py` | which posterior each quoted interval comes from (Sec. V.B) |
| `assumption_checks.py` | the narrower α prior of Roy et al. as a truncation (Table III); the frame of the boson mass (Sec. V.B) |
| `inspect_masscut.py`, `inspect_norm.py`, `inspect_samplers.py`, `inspect_mn.py` | the component-mass cut that only the vacuum prior applied, and how each sampler normalises constrained priors (Sec. IV.E) |
| `identity_check2.py` | stability of P and f across samplers and Monte Carlo estimates (Sec. V.C) |
| `sr_vs_restricted.py`, `allparam_ks.py` | each superradiance posterior equals the agnostic posterior truncated to its region (Sec. V.C) |
| `box_matrix.py` | the truncation identity between density bounds (Sec. V.C) |
| `synth_exact_check.py`, `synth_samplers.py` | the synthetic test: its exact evidences, and nessai and the acceptance walk on it (Sec. V.C) |
| `gwtc_config.py` | the settings of the GWTC-2.1 analysis of GW190728 (Sec. V.E) |
| `gwtc_compare.py`, `phase_posteriors.py` | our posteriors against GWTC-2.1, with the phase sampled and marginalised (Sec. V.A) |
| `psd_leak_test.py` | leakage through the analysis window, off source (Sec. V.E) |
| `noise_reweight.py` | Table IV, the cells marked † |
| `background_by_snr.py` | the vacuum background split by the loudness of the injected signal, and GW190728 against the injections as loud as it is; `--det` for the second campaign, whose masses match the event's (Secs. V.E, V.F, VI.B) |
| `virgo_dataquality.py` | whitened-strain statistics of H1, L1 and Virgo at the event against the 50 injection segments: no detector is unusual (Sec. V.E) |
| `virgo_numbers.py` | the Virgo and noise-spectrum null tests and the full-likelihood background (Secs. V.E–F, VI.A–B) |
| `cloud_injections.py` | the environmental signals of script 31: their Bayes factors next to their vacuum twins and GW190728, their rise with Virgo and the catalogue's spectra, the recovery of the injected cloud, the superradiance posterior as the agnostic one cut to its region, and the gain in fit by detector (Sec. V.G) |
| `virgo_slides_summary.py` | the tail of script 33's Virgo time slides under each selection of slides (all, no glitch, effective sample size ≥ 30, within 1 h of the event), and the chance that one of 28 events shows such a rise (Sec. V.I) |
| `eob_injections.py` | the vacuum signals of script 32, made with SEOBNRv4PHM: their overlap with their IMRPhenomXPHM twins before and after alignment and the norm of the difference, their Bayes factors and rise with Virgo and the catalogue's spectra paired with the twins', and the chirp-mass shift and density bound of the recovery (Secs. IV.F, V.H); imports helpers from `cloud_injections.py` |
