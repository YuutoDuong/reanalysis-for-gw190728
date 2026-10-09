"""Synthetic agnostic problem (exact ln Z = +0.09) through other samplers:
is dynesty-rwalk's +1.1 overestimate specific to the short random walk?"""
import importlib.util, sys
import multiprocessing as mp
mp.set_start_method("fork", force=True)
sys.path.insert(0, ".")
import bilby
spec = importlib.util.spec_from_file_location("s18", "scripts/18_synthetic_identity_check.py")
m = importlib.util.module_from_spec(spec); sys.modules["s18"] = m; spec.loader.exec_module(m)
bilby.core.utils.logger.setLevel("ERROR")

pri = m.make_priors(m.PRIORS["agnostic"])
exact = m.exact_ln_z(pri, n=1_000_000)
for name, kw in (("nessai", dict(sampler="nessai")),
                 ("dynesty acceptance-walk 60", dict(sampler="dynesty", sample="acceptance-walk", naccept=60))):
    r = bilby.run_sampler(likelihood=m.Synthetic(), priors=pri, nlive=1000, npool=4, dlogz=0.1,
                          outdir=f"/tmp/synth_{kw['sampler']}", label="s", plot=False,
                          resume=False, save=False, **kw)
    print(f"{name:28s} ln Z {r.log_evidence:+.3f} +- {r.log_evidence_err:.3f}   exact {exact:+.3f}"
          f"   error {r.log_evidence - exact:+.2f}", flush=True)
