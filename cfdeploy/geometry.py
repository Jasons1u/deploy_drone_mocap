"""Angles and rotations (quaternions are (w, x, y, z), body -> world, Motive world Z up)."""

from __future__ import annotations

import math

import numpy as np


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def yaw_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tilt_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def clamp_cylinder(p, r_max: float, z_lo: float, z_hi: float) -> np.ndarray:
    """p moved radially to within r_max of the z axis and vertically into [z_lo, z_hi]."""
    q = np.array(p, float)
    r = math.hypot(q[0], q[1])
    if r > r_max:
        q[:2] *= r_max / r
    q[2] = min(max(q[2], z_lo), z_hi)
    return q


def quat_rotate(q, v) -> np.ndarray:
    """Rotate v by the unit quaternion q."""
    w, u = q[0], np.asarray(q[1:], float)
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)
