"""Discrete independent-joint PID controller, as found on industrial servo drives.

Features: derivative on measurement with a first-order filter, reference-rate
feedforward in the derivative term, integrator clamping anti-windup, torque
saturation. Six gains are tuned: (Kp, Ki, Kd) for each joint.
"""
import numpy as np

PARAM_NAMES = ["Kp1", "Ki1", "Kd1", "Kp2", "Ki2", "Kd2"]
# Search box, log-uniform.  Units: N m/rad, N m/(rad s), N m s/rad
LOWER = np.array([10.0, 1.0, 0.5, 10.0, 1.0, 0.5])
UPPER = np.array([3000.0, 5000.0, 60.0, 1500.0, 3000.0, 30.0])
_LOG_LO, _LOG_HI = np.log(LOWER), np.log(UPPER)


def to_gains(u):
    """Map a point of the unit cube [0, 1]^6 to PID gains (log scale)."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return np.exp(_LOG_LO + u * (_LOG_HI - _LOG_LO))


def to_unit(gains):
    """Inverse of to_gains."""
    return (np.log(np.asarray(gains, float)) - _LOG_LO) / (_LOG_HI - _LOG_LO)


class JointPID:
    def __init__(self, kp, ki, kd, dt, tau_max, tf=0.004):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt, self.tau_max = dt, tau_max
        self.alpha = dt / (tf + dt)          # first-order derivative filter
        self.i = 0.0
        self.d = 0.0
        self.y_prev = None

    def step(self, r, y, r_dot=0.0):
        e = r - y
        if self.y_prev is None:
            self.y_prev = y
        d_raw = r_dot - (y - self.y_prev) / self.dt
        self.d += self.alpha * (d_raw - self.d)
        self.y_prev = y
        u_unsat = self.kp * e + self.ki * self.i + self.kd * self.d
        u = max(-self.tau_max, min(self.tau_max, u_unsat))
        # clamping anti-windup: stop integrating while saturated in the same direction
        if u == u_unsat or (u_unsat > 0) != (e > 0):
            self.i += e * self.dt
        return u
