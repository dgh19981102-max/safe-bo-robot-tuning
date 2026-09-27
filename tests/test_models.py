"""Sanity checks for the arm model and the Gaussian process.  Run: python -m pytest tests"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from autotune.arm import ArmParams, TwoLinkArm          # noqa: E402
from autotune.gp import GP, matern52                     # noqa: E402
from autotune.controller import to_gains, to_unit        # noqa: E402


def test_energy_conserved_without_friction():
    """With friction and torques removed, total energy must stay constant (RK4, 0.2 ms)."""
    arm = TwoLinkArm(ArmParams(payload=1.5))
    q1, q2, w1, w2 = 0.3, 0.5, 0.0, 0.0
    e0 = arm.energy(q1, q2, w1, w2)
    dt = 2e-4
    f = lambda a, b, c, d: arm.accel(a, b, c, d, 0.0, 0.0, friction=False)
    for _ in range(10_000):
        k1 = (w1, w2, *f(q1, q2, w1, w2))
        k2 = (w1 + dt / 2 * k1[2], w2 + dt / 2 * k1[3], *f(q1 + dt / 2 * k1[0], q2 + dt / 2 * k1[1], w1 + dt / 2 * k1[2], w2 + dt / 2 * k1[3]))
        k3 = (w1 + dt / 2 * k2[2], w2 + dt / 2 * k2[3], *f(q1 + dt / 2 * k2[0], q2 + dt / 2 * k2[1], w1 + dt / 2 * k2[2], w2 + dt / 2 * k2[3]))
        k4 = (w1 + dt * k3[2], w2 + dt * k3[3], *f(q1 + dt * k3[0], q2 + dt * k3[1], w1 + dt * k3[2], w2 + dt * k3[3]))
        q1 += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        q2 += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        w1 += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        w2 += dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    assert abs(arm.energy(q1, q2, w1, w2) - e0) < 1e-6 * max(1.0, abs(e0)) + 1e-6


def test_static_gravity_torque():
    """Holding still, the required torque equals the analytical gravity torque."""
    p = ArmParams(payload=0.0, fc1=0.0, fc2=0.0)
    arm = TwoLinkArm(p)
    g1, g2 = arm.gravity(0.0, 0.0)               # arm horizontal
    expected2 = p.m2 * p.g * p.l2 / 2
    expected1 = p.m1 * p.g * p.l1 / 2 + p.m2 * p.g * (p.l1 + p.l2 / 2)
    assert np.isclose(g2, expected2) and np.isclose(g1, expected1)
    a1, a2 = arm.accel(0.0, 0.0, 0.0, 0.0, g1, g2)
    assert abs(a1) < 1e-12 and abs(a2) < 1e-12


def test_gain_mapping_roundtrip():
    g = np.array([600, 900, 15, 300, 500, 6.0])
    assert np.allclose(to_gains(to_unit(g)), g)


def test_gp_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    X = rng.random((15, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    y = (y - y.mean()) / y.std()
    gp = GP(3)
    th = rng.normal(size=5) * 0.3
    f0, g = gp._nlml(th, X, y)
    num = np.array([(gp._nlml(th + 1e-6 * e, X, y)[0] - f0) / 1e-6 for e in np.eye(5)])
    assert np.allclose(g, num, atol=1e-4)


def test_gp_matches_scikit_learn():
    """Same kernel and hyper-parameters -> same posterior as scikit-learn."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    rng = np.random.default_rng(1)
    X = rng.random((25, 2))
    y = np.cos(4 * X[:, 0]) * X[:, 1] + 0.01 * rng.standard_normal(25)
    gp = GP(2).fit(X, y, restarts=2, rng=rng)
    k = (ConstantKernel(gp.sf2, "fixed") * Matern(gp.ls, "fixed", nu=2.5)
         + WhiteKernel(gp.sn2 + 1e-8, "fixed"))
    sk = GaussianProcessRegressor(k, optimizer=None).fit(X, (y - gp.ym) / gp.ys)
    Xs = rng.random((10, 2))
    mu, sd = gp.predict(Xs)
    mu_sk, sd_sk = sk.predict(Xs, return_std=True)
    assert np.allclose(mu, mu_sk * gp.ys + gp.ym, atol=1e-6)
    # sklearn's std includes the white-noise term
    assert np.allclose(sd ** 2 + gp.sn2 * gp.ys ** 2, (sd_sk * gp.ys) ** 2, rtol=1e-4, atol=1e-8)
