"""Gaussian process regression, written from scratch with NumPy/SciPy.

Kernel : Matern-5/2 with automatic relevance determination (one length scale per input)

    k(x, x') = sf^2 (1 + sqrt(5) r + 5/3 r^2) exp(-sqrt(5) r),
    r^2 = sum_j ((x_j - x'_j) / l_j)^2

Hyper-parameters (log l_1..l_d, log sf, log sn) are fitted by maximising the log
marginal likelihood

    log p(y | X) = -1/2 y^T K^-1 y - sum log diag(L) - n/2 log 2 pi,   K = L L^T

with analytic gradients and L-BFGS-B restarts. Targets are standardised internally.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

SQRT5 = np.sqrt(5.0)


def _sqdist(A, B, ls):
    A = A / ls
    B = B / ls
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0.0)


def matern52(A, B, ls, sf2):
    r = np.sqrt(_sqdist(A, B, ls))
    return sf2 * (1 + SQRT5 * r + 5.0 / 3.0 * r * r) * np.exp(-SQRT5 * r)


class GP:
    def __init__(self, dim, ls_bounds=(0.03, 5.0), sf_bounds=(0.05, 5.0), sn_bounds=(1e-3, 0.7),
                 prior_mean=None, prior_scale=None, min_fit=6):
        """prior_mean / prior_scale: fix the GP prior mean and output scale instead of
        estimating them from data. A pessimistic prior mean makes unexplored regions look
        unsafe (used for the safety model). Hyper-parameters are only fitted once
        min_fit points are available; before that the initial values are used."""
        self.dim = dim
        self.prior_mean, self.prior_scale, self.min_fit = prior_mean, prior_scale, min_fit
        self.bounds = ([np.log(ls_bounds)] * dim + [np.log(sf_bounds), np.log(sn_bounds)])
        self.theta = np.concatenate([np.full(dim, np.log(0.4)), [0.0, np.log(0.05)]])

    # ---------------------------------------------------------------- likelihood
    def _nlml(self, theta, X, y):
        d = self.dim
        ls = np.exp(theta[:d])
        sf2 = np.exp(2 * theta[d])
        sn2 = np.exp(2 * theta[d + 1])
        n = len(y)
        diff = (X[:, None, :] - X[None, :, :]) / ls            # n x n x d
        r2 = (diff ** 2).sum(-1)
        r = np.sqrt(r2)
        e = np.exp(-SQRT5 * r)
        Kf = sf2 * (1 + SQRT5 * r + 5.0 / 3.0 * r2) * e
        K = Kf + (sn2 + 1e-8) * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        alpha = cho_solve((L, True), y)
        nlml = 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)
        # gradients: d nlml / d theta = -1/2 tr((alpha alpha^T - K^-1) dK/dtheta)
        Kinv = cho_solve((L, True), np.eye(n))
        W = np.outer(alpha, alpha) - Kinv
        grad = np.zeros_like(theta)
        # dK/dlog l_j = sf2 * 5/3 (1 + sqrt5 r) e * diff_j^2
        common = sf2 * 5.0 / 3.0 * (1 + SQRT5 * r) * e
        for j in range(d):
            grad[j] = -0.5 * np.sum(W * common * diff[:, :, j] ** 2)
        grad[d] = -0.5 * np.sum(W * 2 * Kf)
        grad[d + 1] = -0.5 * np.trace(W) * 2 * sn2
        return nlml, grad

    def fit(self, X, y, restarts=1, rng=None):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        self.ym = y.mean() if self.prior_mean is None else self.prior_mean
        self.ys = max(y.std(), 0.3) if self.prior_scale is None else self.prior_scale
        ys = (y - self.ym) / self.ys
        rng = rng or np.random.default_rng(0)
        if len(y) >= self.min_fit:
            starts = [self.theta] + [np.array([rng.uniform(lo, hi) for lo, hi in self.bounds])
                                     for _ in range(restarts)]
            best = None
            for t0 in starts:
                res = minimize(self._nlml, t0, args=(X, ys), jac=True, method="L-BFGS-B",
                               bounds=self.bounds, options=dict(maxiter=200))
                if best is None or res.fun < best.fun:
                    best = res
            self.theta = best.x
        self.X, self.y_std = X, ys
        d = self.dim
        self.ls = np.exp(self.theta[:d])
        self.sf2 = np.exp(2 * self.theta[d])
        self.sn2 = np.exp(2 * self.theta[d + 1])
        K = matern52(X, X, self.ls, self.sf2) + (self.sn2 + 1e-8) * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = cho_solve((self.L, True), ys)
        return self

    def predict(self, Xs):
        """Posterior mean and standard deviation of the latent function (original units)."""
        Xs = np.atleast_2d(Xs)
        Ks = matern52(Xs, self.X, self.ls, self.sf2)
        mu = Ks @ self.alpha
        v = solve_triangular(self.L, Ks.T, lower=True)
        var = np.maximum(self.sf2 - (v * v).sum(0), 1e-12)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys
