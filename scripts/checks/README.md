# Checks

These are the ad-hoc checks behind specific statements of the paper, kept as they were run.
Run them from the repository root after the numbered scripts have filled `outdir/` and written
the results files. Most only read results. `noise_reweight.py` and `psd_leak_test.py` evaluate
likelihoods and take about an hour on 10 cores.

| Script | Statement it supports |
|---|---|
| `sec6_numbers.py` | posterior summaries, Savage–Dickey estimate, density-bound values, best-fit likelihood gains and their background (Secs. V.A–C, V.F, VI.B) |
| `sec6_precise.py` | quantiles of the chirp mass, effective spin and mass ratio; prior and posterior mass below 10⁶ g cm⁻³ (Sec. V.B) |
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
| `virgo_numbers.py` | the Virgo and noise-spectrum null tests and the full-likelihood background (Secs. V.E–F, VI.A–B) |
