"""Tuning strategies compared in the study.

All strategies see the same black box: choose gains u in [0, 1]^6, run one
experiment, observe (cost, safety). Every experiment counts, safe or not.

  RandomSearch   uniform sampling of the unit cube
  CMAES          covariance matrix adaptation evolution strategy (pycma)
  BO             GP surrogate on log cost + expected improvement
  SafeBO         two GPs (log cost, log safety); only evaluates gains whose safety
                 upper confidence bound is below the limit, starting from a known-safe
                 seed.  The safe set grows as data accumulate (SafeOpt-style).
  ContextualSafeBO
                 SafeBO whose GPs also take the payload as an input, so data from
                 earlier payloads warm-start tuning after a payload change.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .gp import GP
from .controller import to_gains
from .task import evaluate

DIM = 6


def expected_improvement(mu, sd, best):
    """EI for minimisation."""
    sd = np.maximum(sd, 1e-9)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


class Experiment:
    """Wraps the task and logs every evaluation."""

    def __init__(self, payload, seed):
        self.payload = payload
        self.rng = np.random.default_rng(seed)
        self.log = []                      # dicts with u, cost, safety, safe

    def __call__(self, u):
        r = evaluate(to_gains(u), self.payload, seed=int(self.rng.integers(1 << 30)))
        rec = dict(u=np.asarray(u, float).copy(), payload=self.payload,
                   cost=r["cost"], safety=r["safety"], safe=r["safe"])
        self.log.append(rec)
        return rec


# --------------------------------------------------------------------- baselines
def random_search(exp, budget, rng):
    for _ in range(budget):
        exp(rng.random(DIM))


def cmaes(exp, budget, rng, x0, sigma0=0.15):
    import cma
    es = cma.CMAEvolutionStrategy(np.asarray(x0), sigma0,
                                  dict(bounds=[0, 1], seed=int(rng.integers(1, 1 << 30)),
                                       verbose=-9, popsize=8))
    n = 0
    while n < budget:
        xs = es.ask()
        fs = []
        for x in xs:
            if n >= budget:
                break
            fs.append(np.log(exp(x)["cost"]))
            n += 1
        if len(fs) == len(xs):
            es.tell(xs, fs)


# --------------------------------------------------------------------- acquisition helpers
def _candidates(rng, centres, n_global, n_local, scales=(0.03, 0.08, 0.2)):
    C = [rng.random((n_global, DIM))]
    for c in centres:
        for s in scales:
            C.append(np.clip(c + s * rng.standard_normal((n_local, DIM)), 0, 1))
    return np.vstack(C)


def _refine(fun, x0):
    """Local polish of an acquisition maximiser (derivative-free, bounded)."""
    res = minimize(lambda x: -fun(x[None, :])[0], x0, method="L-BFGS-B",
                   bounds=[(0, 1)] * DIM, options=dict(maxiter=30))
    return np.clip(res.x, 0, 1) if -res.fun >= fun(x0[None, :])[0] else x0


# --------------------------------------------------------------------- standard BO
def bayes_opt(exp, budget, rng, n_init=5):
    gp = GP(DIM)
    for _ in range(n_init):
        exp(rng.random(DIM))
    for it in range(budget - n_init):
        X = np.array([r["u"] for r in exp.log])
        y = np.log([r["cost"] for r in exp.log])
        gp.fit(X, y, restarts=1, rng=rng)
        best = y.min()
        acq = lambda Z: expected_improvement(*gp.predict(Z), best)
        C = _candidates(rng, X[np.argsort(y)[:3]], 1500, 150)
        a = acq(C)
        x = _refine(acq, C[np.argmax(a)])
        exp(x)


# --------------------------------------------------------------------- safe BO
class SafeBOCore:
    """Shared machinery for (contextual) safe BO.

    inputs z = [u] or [u, context]; safety modelled as log(safety index) <= 0.
    """

    def __init__(self, dim, beta=2.5, radius=0.15):
        self.gp_f = GP(dim)
        # Safety model: pessimistic prior (log safety 0.7 ~ safety index 2, i.e. unsafe),
        # fixed output scale, capped length scales and a late hyper-parameter fit, so the
        # model never extrapolates safety far beyond verified data.
        self.gp_s = GP(dim, ls_bounds=(0.08, 0.35), sf_bounds=(0.5, 2.0),
                       prior_mean=0.7, prior_scale=1.0, min_fit=15)
        self.gp_s.theta = np.concatenate([np.full(dim, np.log(0.2)), [0.0, np.log(0.05)]])
        self.beta = beta
        self.radius = radius     # trust radius around verified-safe gains (unit-cube distance)

    def propose(self, Z, yf, ys, rng, context=None):
        self.gp_f.fit(Z, yf, restarts=1, rng=rng)
        self.gp_s.fit(Z, ys, restarts=1, rng=rng)
        U = Z[:, :DIM]
        safe_mask = ys <= 0.0
        if context is not None:
            here = np.isclose(Z[:, DIM], context)
            safe_mask &= here
        pool = U[safe_mask] if safe_mask.any() else U
        best_f = yf[safe_mask].min() if safe_mask.any() else yf.min()
        order = np.argsort(yf[safe_mask]) if safe_mask.any() else np.argsort(yf)
        centres = pool[order[:3]]
        C = _candidates(rng, centres, 800, 250)

        def aug(Cu):
            return Cu if context is None else np.hstack([Cu, np.full((len(Cu), 1), context)])

        safe_pts = pool if safe_mask.any() else U[:1]
        if context is not None:
            # historical gains predicted safe (with confidence) at the new payload also
            # serve as trust anchors and as starting points for the local search
            Zh = np.hstack([U, np.full((len(U), 1), context)])
            mu_h, sd_h = self.gp_s.predict(Zh)
            ok_h = (mu_h + self.beta * sd_h <= 0.0) & (ys <= 0.0)
            if ok_h.any():
                safe_pts = np.vstack([safe_pts, U[ok_h]])
                mu_fh = self.gp_f.predict(Zh[ok_h])[0]
                extra = U[ok_h][np.argsort(mu_fh)[:3]]
                C = np.vstack([C, _candidates(rng, extra, 0, 250)])

        def acq(Cu):
            Za = aug(Cu)
            mu_s, sd_s = self.gp_s.predict(Za)
            near = np.min(np.linalg.norm(Cu[:, None, :] - safe_pts[None, :, :], axis=2), axis=1) <= self.radius
            safe = (mu_s + self.beta * sd_s <= 0.0) & near
            mu_f, sd_f = self.gp_f.predict(Za)
            ei = expected_improvement(mu_f, sd_f, best_f)
            return np.where(safe, ei, -1.0 - (mu_s + self.beta * sd_s))

        a = acq(C)
        if a.max() > 0:
            x = _refine(acq, C[np.argmax(a)])
            if acq(x[None, :])[0] > 0:
                return x
            return C[np.argmax(a)]
        # no confidently-safe candidate improves: take the most confidently safe one
        return C[np.argmax(a)]


def safe_bayes_opt(exp, budget, rng, x_seed):
    core = SafeBOCore(DIM)
    exp(x_seed)
    for _ in range(budget - 1):
        Z = np.array([r["u"] for r in exp.log])
        yf = np.log([r["cost"] for r in exp.log])
        ys = np.log([r["safety"] for r in exp.log])
        exp(core.propose(Z, yf, ys, rng))


def contextual_safe_bayes_opt(exp, budget, rng, x_seed, history, ctx_scale=4.0):
    """history: log records from earlier payloads (warm start)."""
    core = SafeBOCore(DIM + 1)
    c_now = exp.payload / ctx_scale
    exp(x_seed)
    for _ in range(budget - 1):
        recs = history + exp.log
        Z = np.array([np.append(r["u"], r["payload"] / ctx_scale) for r in recs])
        yf = np.log([r["cost"] for r in recs])
        ys = np.log([r["safety"] for r in recs])
        exp(core.propose(Z, yf, ys, rng, context=c_now))
