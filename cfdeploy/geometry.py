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


def quat_rotate(q, v) -> np.ndarray:
    """Rotate v by the unit quaternion q."""
    w, u = q[0], np.asarray(q[1:], float)
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)
