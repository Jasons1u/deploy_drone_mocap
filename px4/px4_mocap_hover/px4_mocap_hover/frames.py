"""Frame conversions and quaternion math.

PX4:          world NED (x North, y East, z Down), body FRD, q = FRD -> NED
This package: world ENU (x East,  y North, z Up),  body FLU, q = FLU -> ENU   (= Isaac Lab / REP-103)
Mocap world:  Z-up (see mocap_natnet.frames); it IS this package's ENU world, so "North" is
              simply mocap +Y and the PX4 local frame is the mocap frame expressed in NED.

Quaternions are [w, x, y, z] Hamilton. The quaternion helpers mirror isaaclab.utils.math (same
formulas, Euler XYZ in (-pi, pi]) so the policy observation matches training.
"""

from __future__ import annotations

import numpy as np

_S = np.sqrt(0.5)


# ---------------------------------------------------------------------------
# PX4 <-> ENU/FLU
# ---------------------------------------------------------------------------

def ned_to_enu(v) -> np.ndarray:
    """World vector NED -> ENU (also ENU -> NED: the map is its own inverse)."""
    x, y, z = v
    return np.array([y, x, -z], dtype=np.float64)


enu_to_ned = ned_to_enu


def frd_to_flu(v) -> np.ndarray:
    """Body vector FRD -> FLU (also FLU -> FRD)."""
    x, y, z = v
    return np.array([x, -y, -z], dtype=np.float64)


flu_to_frd = frd_to_flu


def frd_ned_to_flu_enu(q) -> np.ndarray:
    """PX4 attitude (FRD -> NED) -> FLU -> ENU. Also the inverse map: it is an involution."""
    w, x, y, z = q
    out = _S * np.array([w + z, x + y, x - y, w - z], dtype=np.float64)
    return out / (np.linalg.norm(out) + 1e-12)


flu_enu_to_frd_ned = frd_ned_to_flu_enu


# ---------------------------------------------------------------------------
# Quaternion helpers (isaaclab.utils.math semantics)
# ---------------------------------------------------------------------------

def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_apply(q, v) -> np.ndarray:
    """R(q) @ v."""
    qv = np.asarray(q[1:], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + q[0] * t + np.cross(qv, t)


def quat_apply_inverse(q, v) -> np.ndarray:
    """R(q)^T @ v, for unit q."""
    return quat_apply(quat_conj(q), v)


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)


def euler_xyz_from_quat(q) -> np.ndarray:
    """(roll, pitch, yaw) [rad], each in (-pi, pi]. Port of isaaclab euler_xyz_from_quat."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sin_pitch) if abs(sin_pitch) >= 1.0 else np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def yaw_from_quat(q) -> float:
    return float(euler_xyz_from_quat(q)[2])


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
