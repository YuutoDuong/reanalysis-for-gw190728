"""Independent check of the synthetic test's exact evidences:
Z_4D = E_{constrained prior}[K(x) / pi_box(x)]  (dummy dims integrate to 1)."""
import importlib.util, sys
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("s18", "scripts/18_synthetic_identity_check.py")
m = importlib.util.module_from_spec(spec); sys.modules["s18"] = m; spec.loader.exec_module(m)

rng = np.random.default_rng(7)
for name, kw in m.PRIORS.items():
    pri = m.make_priors(kw)
    n = 400_000
    x = {k: np.asarray(m.BOX[k].sample(n)) for k in m.KEYS}
    ok = np.asarray(pri.evaluate_constraints(x), bool)
    xs = np.array([x[k][ok] for k in m.KEYS])
    ln_box = sum(np.asarray(m.BOX[k].ln_prob(x[k][ok])) for k in m.KEYS)
    lw = m.KDE.logpdf(xs) - ln_box
    ln_z = logsumexp(lw) - np.log(lw.size)
    w = np.exp(lw - lw.max()); ess = w.sum() ** 2 / (w ** 2).sum()
    # dummy-dimension factor, analytically: int_0^1 N(0.5, 0.05) du per dim
    print(f"{name:14s} prior-MC ln Z = {ln_z:+.3f} (ESS {ess:.0f} of {lw.size});"
          f" KDE-sampling exact {m.exact_ln_z(pri, n=1_000_000):+.3f}")
