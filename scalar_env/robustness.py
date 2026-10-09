"""
Section V.D -- Robustness tests.

Every test is expressed as a list of RunConfig variants of the baseline
environmental run, so the whole suite is: build configs -> run_pe each ->
tabulate ln B against the matching vacuum run.  Runs resume from disk,
so the suite can be executed in overnight batches.

Axes (mirroring the paper):
  1. prior sensitivity   tau_d ranges, flat vs log-flat alpha, SR prior
  2. sampler comparison  dynesty / pymultinest / nessai
  3. waveform systematics IMRPhenomXPHM / IMRPhenomPv2 / IMRPhenomXP
  4. noise modelling     Welch fftlength 4 / 8 / 16 s, single-detector
"""

from pathlib import Path

import numpy as np

from .inference import RunConfig, run_pe, ln_bayes_factor


def prior_variants(base):
    """Test 1: prior sensitivity.  Roy et al. found ln B = (3.4, 3.5,
    3.5, 2.8) for the superradiance prior with fixed tau_d = 1e5..1e8 yr;
    we repeat that sweep and add density-prior and alpha-prior variants."""
    variants = []
    # Faithful Roy et al. SR priors (SM Eqs. 9-10: spin-independent
    # alpha/(1+q) window + saturation density cap), at the baseline box
    # (rho_max = 2e8) and at the narrower 1e7 box.  Roy et al.'s GW190728
    # posteriors (their Figs. 5-6) extend to ~1e8 g/cm^3 and peak near 7e7
    # under the SR prior, so 2e8 is the matched comparison and 1e7 is the
    # prior-sensitivity point.  NOTE the "_rho1e7" labels carry an explicit
    # rho_max: run_pe caches on the label alone, so a variant whose box is
    # inherited from `base` would silently reload a result computed under a
    # different box if the baseline ever changes again.
    for exp in (5, 6, 7, 8):
        variants.append(base.variant(
            f"sr_eq9_taud1e{exp}_rho2e8",
            prior_kwargs=dict(base.prior_kwargs,
                              superradiance_tau_d_yr=10.0**exp,
                              rho_max_gcm3=2e8)))
    for exp in (5, 6, 7, 8):
        variants.append(base.variant(
            f"sr_eq9_taud1e{exp}",
            prior_kwargs=dict(base.prior_kwargs,
                              superradiance_tau_d_yr=10.0**exp,
                              rho_max_gcm3=1e7)))
    # Density-bound series.  With the posterior interior to the box, the
    # evidence carries an explicit -ln(rho_max) Occam term; this series
    # measures it.  rho_max2e8 duplicates the baseline configuration and
    # therefore also serves as a run-to-run scatter check.
    variants.append(base.variant(
        "rho_max2e8", prior_kwargs=dict(base.prior_kwargs, rho_max_gcm3=2e8)))
    variants.append(base.variant(
        "rho_max1e7", prior_kwargs=dict(base.prior_kwargs, rho_max_gcm3=1e7)))
    # NOTE: every SR run made before tau_d was bound into the conversion
    # function (priors.environmental_priors) silently skipped the SR
    # constraints and is an agnostic run in disguise -- including the
    # older merger-spin-conditioned gw190728_env_sr_taud1e{5..8} set.  The
    # "SR-prior implementation choices move ln B by several units" reading
    # of those runs does not stand.  Their outdirs are archived in
    # outdir_srbug/ and outdir_linearised/.
    variants.append(base.variant(
        "log_alpha", prior_kwargs=dict(base.prior_kwargs,
                                       alpha_prior="log-uniform",
                                       alpha_bounds=(0.01, 0.75))))
    variants.append(base.variant(
        "rho_max1e6", prior_kwargs=dict(base.prior_kwargs, rho_max_gcm3=1e6)))
    variants.append(base.variant(
        "rho_max1e8", prior_kwargs=dict(base.prior_kwargs, rho_max_gcm3=1e8)))
    return variants


def sampler_variants(base):
    """Test 2: independent nested samplers."""
    return [base.variant(s, sampler=s) for s in ("pymultinest", "nessai")]


def waveform_variants(base):
    """Test 3: vacuum-baseline systematics."""
    return [base.variant(a.lower(), approximant=a)
            for a in ("IMRPhenomPv2", "IMRPhenomXP")]


def noise_variants(base):
    """Test 4: PSD estimation, per-detector analyses, and the full
    H1+L1+V1 network (GW190728 was a three-detector event; Roy et al.
    presumably included Virgo, our baseline does not)."""
    variants = [base.variant(f"psd{n}s", psd_fftlength=n) for n in (8, 16)]
    variants += [base.variant(f"only_{d}", detectors=(d,)) for d in ("H1", "L1")]
    variants.append(base.variant("hlv", detectors=("H1", "L1", "V1")))
    return variants


def full_suite(env_base, vac_base):
    """All (environmental, matched-vacuum) config pairs for the suite.

    Each environmental variant is paired with a vacuum run sharing its
    sampler / approximant / PSD settings, so ln B compares like with like.
    """
    pairs = [(env_base, vac_base)]
    for env in prior_variants(env_base):
        pairs.append((env, vac_base))       # priors only affect M_env
    for env in sampler_variants(env_base):
        pairs.append((env, vac_base.variant(env.sampler, sampler=env.sampler)))
    for env in waveform_variants(env_base):
        pairs.append((env, vac_base.variant(env.approximant.lower(),
                                            approximant=env.approximant)))
    for env in noise_variants(env_base):
        pairs.append((env, vac_base.variant(
            env.label.replace(env_base.label + "_", ""),
            psd_fftlength=env.psd_fftlength, detectors=env.detectors)))
    return pairs


def run_suite(pairs):
    """Execute every pair; returns [(label, lnB, err), ...].

    A pair that fails (e.g. a GWOSC download that keeps timing out) is
    reported and skipped, so one outage cannot discard the whole table;
    rerunning the suite later fills it in from the cached runs.
    """
    rows = []
    for env_cfg, vac_cfg in pairs:
        try:
            lnb, err = ln_bayes_factor(run_pe(env_cfg), run_pe(vac_cfg))
        except Exception as exc:
            print(f"{env_cfg.label:40s}  FAILED ({type(exc).__name__}: {exc})")
            continue
        rows.append((env_cfg.label, lnb, err))
        print(f"{env_cfg.label:40s}  ln B = {lnb:+.2f} +/- {err:.2f}")
    return rows


def plot_lnb_robustness(rows, reference=3.5, name="lnb_robustness"):
    """Fig 3 of the paper: ln B across all systematic variations."""
    import matplotlib.pyplot as plt

    labels, lnbs, errs = zip(*rows)
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(rows) + 1))
    ax.errorbar(lnbs, y, xerr=errs, fmt="o", color="C0", capsize=3)
    ax.axvline(reference, ls="--", color="C3",
               label=f"Roy et al. (ln B = {reference})")
    ax.axvspan(-1, 1, alpha=0.15, color="grey", label="inconclusive")
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"$\ln B^{\rm env}_{\rm vac}$")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)

    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    return fig
