"""The tuning task: one closed-loop experiment on the arm and its scores.

A single evaluation mimics one test run on a real robot:
  * the brake releases at the start pose with the integrators empty (gravity sag),
  * the arm follows a quintic point-to-point move and then holds position,
  * an external torque disturbance hits joint 1 while holding,
  * encoders are quantised and noisy, and the controller sees a one-sample delay.

Scores
  rmse_deg    RMS joint tracking error (deg)
  chatter     mean normalised torque increment |d tau| / tau_max (noise amplification)
  max_err_deg worst joint tracking error (deg)
  cost        rmse_deg + W_CHATTER * chatter   (quantity to minimise)
  safety      max(max_err_deg / 8 deg, chatter / 0.12); the run is safe if safety <= 1.
              Too-soft gains violate the error limit (the arm sags or overshoots into
              surroundings); too-aggressive gains violate the chatter limit (violent
              torque ripple that wears gearboxes).
"""
import math
import numpy as np

from .arm import ArmParams, TwoLinkArm
from .controller import JointPID

DT = 0.002              # control period (s), 500 Hz
T_END = 2.0             # experiment length (s)
Q_START = (-0.6, 0.9)   # start pose (rad)
Q_GOAL = (0.5, -0.3)    # goal pose (rad)
T_MOVE0, T_MOVE = 0.2, 0.8
DIST_T0, DIST_T1, DIST_TAU = 1.4, 1.5, 8.0   # disturbance window (s) and torque (N m)
ENC_RES = 2 * math.pi / 2 ** 16              # 16-bit encoder
NOISE_STD = 2e-4                             # rad
W_CHATTER = 40.0
SAFE_MAX_ERR_DEG = 8.0                       # safety limit on tracking error (deg)
SAFE_CHATTER = 0.12                          # safety limit on torque ripple
COST_CAP = 40.0


def _quintic(t, a, b):
    """Position and velocity of a quintic blend from a to b over [T_MOVE0, T_MOVE0+T_MOVE]."""
    s = min(max((t - T_MOVE0) / T_MOVE, 0.0), 1.0)
    p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    v = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T_MOVE if 0 < s < 1 else 0.0
    return a + (b - a) * p, (b - a) * v


def evaluate(gains, payload=1.0, seed=0, return_traj=False, params=None):
    """Run one closed-loop experiment. Returns a dict of scores."""
    p = params if params is not None else ArmParams(payload=payload)
    if params is None:
        p.payload = payload
    arm = TwoLinkArm(p)
    rng = np.random.default_rng(seed)
    kp1, ki1, kd1, kp2, ki2, kd2 = [float(g) for g in gains]
    c1 = JointPID(kp1, ki1, kd1, DT, p.tau_max1)
    c2 = JointPID(kp2, ki2, kd2, DT, p.tau_max2)

    n = int(round(T_END / DT))
    noise = rng.normal(0.0, NOISE_STD, size=(n, 2))
    q1, q2 = Q_START
    w1 = w2 = 0.0
    y1p, y2p = q1, q2                     # delayed measurement
    t1p = t2p = 0.0
    se = 0.0
    chatter = 0.0
    max_err = 0.0
    sat = 0
    traj = np.zeros((n, 7)) if return_traj else None
    ok = True

    for k in range(n):
        t = k * DT
        r1, rd1 = _quintic(t, Q_START[0], Q_GOAL[0])
        r2, rd2 = _quintic(t, Q_START[1], Q_GOAL[1])
        # measurement: quantised, noisy, one sample old
        m1 = round((y1p + noise[k, 0]) / ENC_RES) * ENC_RES
        m2 = round((y2p + noise[k, 1]) / ENC_RES) * ENC_RES
        tau1 = c1.step(r1, m1, rd1)
        tau2 = c2.step(r2, m2, rd2)
        if abs(tau1) >= p.tau_max1 or abs(tau2) >= p.tau_max2:
            sat += 1
        if k > 0:
            chatter += abs(tau1 - t1p) / p.tau_max1 + abs(tau2 - t2p) / p.tau_max2
        t1p, t2p = tau1, tau2
        y1p, y2p = q1, q2
        dist = DIST_TAU if DIST_T0 <= t < DIST_T1 else 0.0

        # RK4 integration over one control period (zero-order-hold torque)
        def f(a1, a2, b1, b2):
            return arm.accel(a1, a2, b1, b2, tau1 + dist, tau2)
        k1a, k1b = f(q1, q2, w1, w2)
        h = DT / 2
        k2a, k2b = f(q1 + h * w1, q2 + h * w2, w1 + h * k1a, w2 + h * k1b)
        k3a, k3b = f(q1 + h * (w1 + h * k1a), q2 + h * (w2 + h * k1b), w1 + h * k2a, w2 + h * k2b)
        k4a, k4b = f(q1 + DT * (w1 + h * k2a), q2 + DT * (w2 + h * k2b), w1 + DT * k3a, w2 + DT * k3b)
        dq1 = DT * (w1 + DT / 6 * (k1a + k2a + k3a))
        dq2 = DT * (w2 + DT / 6 * (k1b + k2b + k3b))
        w1 += DT / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        w2 += DT / 6 * (k1b + 2 * k2b + 2 * k3b + k4b)
        q1 += dq1
        q2 += dq2

        e1, e2 = r1 - q1, r2 - q2
        se += e1 * e1 + e2 * e2
        max_err = max(max_err, abs(e1), abs(e2))
        if return_traj:
            traj[k] = (t, r1, q1, r2, q2, tau1, tau2)
        if not (abs(w1) < 200 and abs(w2) < 200) or max_err > 1.5:
            ok = False                     # diverged: experiment aborted
            break

    steps = k + 1
    rmse = math.degrees(math.sqrt(se / steps / 2))
    chat = chatter / steps
    max_err_deg = math.degrees(max_err)
    cost = rmse + W_CHATTER * chat
    if not ok:
        cost, max_err_deg = COST_CAP, max(max_err_deg, 90.0)
    safety = max(max_err_deg / SAFE_MAX_ERR_DEG, chat / SAFE_CHATTER)
    if not ok:
        safety = max(safety, 10.0)
    out = dict(cost=min(cost, COST_CAP), rmse_deg=rmse, chatter=chat,
               max_err_deg=max_err_deg, sat_frac=sat / steps, safety=safety,
               safe=bool(safety <= 1.0), aborted=not ok)
    if return_traj:
        out["traj"] = traj[:steps]
    return out
