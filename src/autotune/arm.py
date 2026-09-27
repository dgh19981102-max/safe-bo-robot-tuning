"""Two-link planar robot arm moving in a vertical plane.

Links are modelled as uniform thin rods; an optional payload is a point mass at
the tip of link 2. The equation of motion is

    M(q) qdd + c(q, qd) + g(q) + f(qd) = tau

with viscous + Coulomb joint friction f and reflected motor inertia on the
diagonal of M. Angles are measured from the horizontal.
"""
from dataclasses import dataclass
import math


@dataclass
class ArmParams:
    l1: float = 0.50        # link lengths (m)
    l2: float = 0.40
    m1: float = 3.0         # link masses (kg)
    m2: float = 2.0
    payload: float = 1.0    # tip payload (kg)
    jm1: float = 0.05       # reflected motor inertia (kg m^2)
    jm2: float = 0.03
    fv1: float = 0.6        # viscous friction (N m s/rad)
    fv2: float = 0.4
    fc1: float = 1.2        # Coulomb friction (N m)
    fc2: float = 0.8
    tau_max1: float = 60.0  # torque limits (N m)
    tau_max2: float = 30.0
    g: float = 9.81


class TwoLinkArm:
    """Rigid-body dynamics of the arm with a tip payload."""

    def __init__(self, p: ArmParams):
        self.p = p
        m2e = p.m2 + p.payload                          # link 2 + payload
        lc1 = p.l1 / 2
        lc2e = (p.m2 * p.l2 / 2 + p.payload * p.l2) / m2e   # combined centre of mass
        I1 = p.m1 * p.l1 ** 2 / 12
        I2 = p.m2 * p.l2 ** 2 / 12
        # inertia of link 2 + payload about the combined centre of mass (parallel axis)
        I2e = I2 + p.m2 * (p.l2 / 2 - lc2e) ** 2 + p.payload * (p.l2 - lc2e) ** 2
        self._a1 = I1 + I2e + p.m1 * lc1 ** 2 + m2e * (p.l1 ** 2 + lc2e ** 2) + p.jm1
        self._a2 = m2e * p.l1 * lc2e
        self._a3 = I2e + m2e * lc2e ** 2
        self._g1 = (p.m1 * lc1 + m2e * p.l1) * p.g
        self._g2 = m2e * lc2e * p.g

    def mass_matrix(self, q2):
        c2 = math.cos(q2)
        m11 = self._a1 + 2 * self._a2 * c2
        m12 = self._a3 + self._a2 * c2
        m22 = self._a3 + self.p.jm2
        return m11, m12, m22

    def gravity(self, q1, q2):
        c12 = math.cos(q1 + q2)
        return self._g1 * math.cos(q1) + self._g2 * c12, self._g2 * c12

    def accel(self, q1, q2, w1, w2, t1, t2, friction=True):
        """Joint accelerations for state (q, qd) and applied torques."""
        p = self.p
        m11, m12, m22 = self.mass_matrix(q2)
        h = self._a2 * math.sin(q2)
        c1 = -h * (2 * w1 * w2 + w2 * w2)             # Coriolis / centrifugal
        c2 = h * w1 * w1
        g1, g2 = self.gravity(q1, q2)
        if friction:
            f1 = p.fv1 * w1 + p.fc1 * math.tanh(w1 / 0.02)
            f2 = p.fv2 * w2 + p.fc2 * math.tanh(w2 / 0.02)
        else:
            f1 = f2 = 0.0
        r1 = t1 - c1 - g1 - f1
        r2 = t2 - c2 - g2 - f2
        det = m11 * m22 - m12 * m12
        return (m22 * r1 - m12 * r2) / det, (m11 * r2 - m12 * r1) / det

    def energy(self, q1, q2, w1, w2):
        """Kinetic + potential energy (used to verify the model)."""
        m11, m12, m22 = self.mass_matrix(q2)
        ke = 0.5 * (m11 * w1 * w1 + 2 * m12 * w1 * w2 + m22 * w2 * w2)
        pe = self._g1 * math.sin(q1) + self._g2 * math.sin(q1 + q2)
        return ke + pe
