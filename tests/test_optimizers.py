"""Tests for the acquisition function and the SafeBO proposal logic.

Run: python -m pytest tests
"""
import math
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from autotune.controller import to_unit                                   # noqa: E402
from autotune.optimizers import (DIM, Experiment, SafeBOCore,             # noqa: E402
                                 _candidates, expected_improvement,
                                 safe_bayes_opt)

# Conservative day-one gains used as the known-safe seed in the experiments.
SEED_U = to_unit([600, 900, 15, 300, 500, 6])


def _ei_reference(mu, sd, best):
    """Independent EI computation via math.erf (no scipy)."""
    z = (best - mu) / sd
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (best - mu) * cdf + sd * pdf


# --------------------------------------------------------------------- EI maths
def test_ei_matches_closed_form():
    cases = [(0.0, 1.0, 0.5), (1.3, 0.4, 0.9), (-2.0, 2.5, -1.0),
             (0.7, 0.05, 0.7), (3.0, 1.5, -1.0), (-0.2, 0.8, 0.0)]
    for mu, sd, best in cases:
        got = float(expected_improvement(np.array([mu]), np.array([sd]), best)[0])
        assert math.isclose(got, _ei_reference(mu, sd, best), rel_tol=1e-10, abs_tol=1e-12)


def test_ei_zero_uncertainty_limit():
    """With sd -> 0, EI collapses to the deterministic improvement max(best - mu, 0)."""
    mu = np.array([0.3, 1.2, 0.9])
    ei = expected_improvement(mu, np.full(3, 1e-12), best=0.9)
    assert np.allclose(ei, [0.6, 0.0, 0.0], atol=1e-9)


def test_ei_at_incumbent_is_sd_over_sqrt_2pi():
    """At mu == best, EI = sd * phi(0) = sd / sqrt(2 pi): pure exploration value."""
    sd = np.array([0.1, 1.0, 2.7])
    ei = expected_improvement(np.zeros(3), sd, best=0.0)
    assert np.allclose(ei, sd / math.sqrt(2.0 * math.pi), rtol=1e-12)


def test_ei_nonnegative_and_increases_with_uncertainty():
    rng = np.random.default_rng(3)
    mu = rng.normal(size=200)
    sd = np.exp(rng.normal(size=200))
    assert np.all(expected_improvement(mu, sd, best=0.25) >= 0.0)
    # at the incumbent, more uncertainty must mean strictly more EI
    ei = expected_improvement(np.zeros(4), np.array([0.1, 0.5, 1.0, 2.0]), best=0.0)
    assert np.all(np.diff(ei) > 0)


# --------------------------------------------------------------------- candidates
def test_candidates_shape_bounds_and_centring():
    rng = np.random.default_rng(0)
    centres = np.array([[0.5] * DIM, [0.2] * DIM])
    C = _candidates(rng, centres, n_global=40, n_local=25)
    # 40 global + 2 centres x 3 scales x 25 local
    assert C.shape == (40 + 2 * 3 * 25, DIM)
    assert C.min() >= 0.0 and C.max() <= 1.0
    # local clouds are centred on their centre: with one centre at 0.5 and many
    # samples, the cloud mean must sit close to it (zero-mean Gaussian shells)
    rng = np.random.default_rng(1)
    C = _candidates(rng, np.array([[0.5] * DIM]), n_global=0, n_local=300)
    assert np.linalg.norm(C.mean(axis=0) - 0.5) < 0.03


# --------------------------------------------------------------------- SafeBO logic
def _synthetic_history(rng, tempting_unsafe=False):
    """Safe ball around 0.5, unsafe shell outside; log-cost bowl around the centre
    (or, if tempting_unsafe, an even better log-cost at an unsafe corner)."""
    centre = np.full(DIM, 0.5)
    safe_pts = np.clip(centre + 0.05 * rng.standard_normal((24, DIM)), 0, 1)
    unsafe_pts = np.clip(centre + rng.choice([-1, 1], size=(16, DIM)) * rng.uniform(0.3, 0.48, (16, DIM)), 0, 1)
    Z = np.vstack([safe_pts, unsafe_pts])
    ys = np.concatenate([np.full(24, -1.2), np.full(16, 1.2)]) + 0.05 * rng.standard_normal(40)
    yf = 0.5 + ((Z - centre) ** 2).sum(axis=1)
    if tempting_unsafe:
        bad = np.full((1, DIM), 0.97)
        Z = np.vstack([Z, bad])
        ys = np.append(ys, 2.0)          # clearly unsafe ...
        yf = np.append(yf, -3.0)         # ... but by far the best observed cost
    return Z, yf, ys, safe_pts


def _safety_ucb(core, x):
    mu, sd = core.gp_s.predict(np.atleast_2d(x))
    return float(mu[0] + core.beta * sd[0])


def test_safebo_proposal_is_confidently_safe_and_within_trust_radius():
    rng = np.random.default_rng(7)
    Z, yf, ys, safe_pts = _synthetic_history(rng)
    core = SafeBOCore(DIM)
    x = core.propose(Z, yf, ys, rng)
    assert x.shape == (DIM,) and x.min() >= 0.0 and x.max() <= 1.0
    # the proposal must pass the core's own safety test: UCB of log-safety <= 0
    assert _safety_ucb(core, x) <= 1e-6
    # and lie within the trust radius of a verified-safe point
    d = np.linalg.norm(safe_pts - x, axis=1).min()
    assert d <= core.radius + 1e-6


def test_safebo_proposal_ignores_tempting_unsafe_optimum():
    rng = np.random.default_rng(11)
    Z, yf, ys, _ = _synthetic_history(rng, tempting_unsafe=True)
    core = SafeBOCore(DIM)
    x = core.propose(Z, yf, ys, rng)
    # never propose the unsafe corner, however good its observed cost was
    assert np.linalg.norm(x - np.full(DIM, 0.97)) > 0.3
    assert _safety_ucb(core, x) <= 1e-6


# --------------------------------------------------------------------- end to end
def test_experiment_log_records_consistent_fields():
    exp = Experiment(payload=1.0, seed=5)
    rec = exp(SEED_U)
    assert len(exp.log) == 1 and exp.log[0] is rec
    assert np.allclose(rec["u"], SEED_U)
    assert rec["safe"] == bool(rec["safety"] <= 1.0)
    assert 0.0 < rec["cost"] <= 40.0 and rec["safety"] > 0.0
    # the documented day-one seed gains really are safe at this payload
    assert rec["safe"]


def test_safe_bayes_opt_small_budget():
    exp = Experiment(payload=1.0, seed=2)
    rng = np.random.default_rng(0)
    safe_bayes_opt(exp, budget=5, rng=rng, x_seed=SEED_U)
    assert len(exp.log) == 5
    assert np.allclose(exp.log[0]["u"], SEED_U) and exp.log[0]["safe"]
    for rec in exp.log:
        assert rec["u"].min() >= 0.0 and rec["u"].max() <= 1.0
        assert np.isfinite(rec["cost"]) and np.isfinite(rec["safety"])
