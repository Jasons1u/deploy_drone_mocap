"""Motive world frame -> this project's mocap world frame.

Everything downstream of `mocap_natnet` sees ONE convention:
    world: right-handed, Z up (x/y are whatever the Motive ground plane defines)
    body : FLU (x forward, y left, z up)
    quat : (w, x, y, z), rotation body -> world

Motive streams either Z-up (Streaming pane "Up Axis: Z Up") or its default Y-up. For Y-up the
world is rotated +90 deg about X:  (x, y, z)_zup = (x, -z, y)_yup. The rigid body frame is fixed
when the body is created in Motive (aligned with the Motive world axes at that moment), so it
gets the same change of basis: q_zup = q_R * q_yup * q_R^-1.

The drone must sit nose along world +X when its rigid body is created, or body x will not be
forward. Nothing here can detect that — check yaw ~ 0 in the smoke test.
"""

from __future__ import annotations

import math

UP_AXES = ("z", "y")

_S = math.sqrt(0.5)
_Q_YUP_TO_ZUP = (_S, _S, 0.0, 0.0)     # +90 deg about X
_Q_ZUP_TO_YUP = (_S, -_S, 0.0, 0.0)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def yaw_from_quat(q) -> float:
    """Yaw about world Z [rad], ZYX convention."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rpy_from_quat(q) -> tuple[float, float, float]:
    """(roll, pitch, yaw) [rad], ZYX convention (right-handed: pitch > 0 = nose down in FLU)."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch, yaw_from_quat(q)


def quat_from_rpy(roll: float, pitch: float, yaw: float):
    """ZYX (roll, pitch, yaw) [rad] -> (w, x, y, z); inverse of rpy_from_quat."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def rotate(q, v):
    """R(q) @ v."""
    return quat_mul(quat_mul(q, (0.0, *v)), quat_conj(q))[1:]


def shortest_arc(u, v):
    """Smallest rotation (w, x, y, z) taking direction u onto direction v (neither need be unit)."""
    nu, nv = math.sqrt(sum(a * a for a in u)), math.sqrt(sum(a * a for a in v))
    u, v = [a / nu for a in u], [a / nv for a in v]
    cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    w = 1.0 + sum(a * b for a, b in zip(u, v))
    n = math.sqrt(w * w + sum(c * c for c in cross))
    return (w / n, *(c / n for c in cross))


def to_z_up(position, quat, up_axis: str):
    """Motive (position, quat) -> Z-up world (position, quat)."""
    if up_axis == "z":
        return tuple(position), tuple(quat)
    if up_axis == "y":
        x, y, z = position
        return (x, -z, y), quat_mul(quat_mul(_Q_YUP_TO_ZUP, quat), _Q_ZUP_TO_YUP)
    raise ValueError(f"up_axis must be one of {UP_AXES}, got {up_axis!r}")


def from_z_up(position, quat, up_axis: str):
    """Inverse of to_z_up (used by the fake server to emit Y-up data)."""
    if up_axis == "z":
        return tuple(position), tuple(quat)
    if up_axis == "y":
        x, y, z = position
        return (x, z, -y), quat_mul(quat_mul(_Q_ZUP_TO_YUP, quat), _Q_YUP_TO_ZUP)
    raise ValueError(f"up_axis must be one of {UP_AXES}, got {up_axis!r}")
