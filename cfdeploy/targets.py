"""Policy target: a live parameter (launch spot by default, moved from RViz), kept inside what training covered."""

from __future__ import annotations

import math
import threading

import numpy as np

from cfdeploy.safe_cf_obs import DRONE_RADIUS, TARGET_KEEP_OUT_EXTRA, TARGET_RADIUS, TARGET_Z, UNSAFE_MARGIN


def clamp_to_region(t) -> np.ndarray:
    """Project a target into the trained target region (cylinder r <= TARGET_RADIUS, z in TARGET_Z)."""
    t = np.array(t, float)
    r = math.hypot(t[0], t[1])
    if r > TARGET_RADIUS:
        t[:2] *= TARGET_RADIUS / r
    t[2] = min(max(t[2], TARGET_Z[0]), TARGET_Z[1])
    return t


def target_problem(t, obstacles) -> str | None:
    """Training kept every obstacle centre at least r + drone + margin + clearance from the target."""
    for o in obstacles:
        keep = o.radius + DRONE_RADIUS + UNSAFE_MARGIN + TARGET_KEEP_OUT_EXTRA
        d = float(np.linalg.norm(np.asarray(t, float) - o.pos))
        if d < keep:
            return f"{d:.2f} m from an obstacle (training kept >= {keep:.2f} m)"
    return None


class TargetParam:
    """Policy target: reset at takeoff, moved live (clamped + checked) by requests; thread-safe."""

    def __init__(self):
        self.lock = threading.Lock()
        self.value, self.msg = None, ""

    def reset(self, xyz) -> None:
        with self.lock:
            self.value = clamp_to_region(xyz)

    def get(self):
        with self.lock:
            return None if self.value is None else self.value.copy()

    def request(self, xyz, obstacles) -> str:
        t = clamp_to_region(xyz)
        problem = target_problem(t, obstacles)
        with self.lock:
            if problem:
                self.msg = f"target refused: {problem}"
            else:
                self.value = t
                clamped = " (clamped into the trained region)" if np.linalg.norm(t - np.asarray(xyz, float)) > 1e-6 else ""
                self.msg = f"target ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}){clamped}"
            return self.msg
