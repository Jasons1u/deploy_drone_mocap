"""Host-side velocity from mocap: alpha-beta filter timed by frame numbers (the policy's velocity input)."""

from __future__ import annotations

import threading

import numpy as np


class AlphaBeta:
    """Velocity from mocap positions timed by frame numbers (exact), restarted after long gaps; thread-safe."""

    def __init__(self, mocap_cfg):
        f = mocap_cfg.velocity_filter
        self.rate, self.alpha, self.beta, self.restart_gap = mocap_cfg.rate, f.alpha, f.beta, f.restart_gap
        self.p = self.v = None
        self.frame = None
        self.lock = threading.Lock()

    def update(self, frame: int, pos) -> None:
        z = np.asarray(pos, float)
        with self.lock:
            if self.p is None or frame - self.frame > self.restart_gap * self.rate or frame <= self.frame:
                self.p, self.v = z.copy(), np.zeros(3)
            else:
                dt = (frame - self.frame) / self.rate
                pred = self.p + self.v * dt
                r = z - pred
                self.p = pred + self.alpha * r
                self.v = self.v + (self.beta / dt) * r
            self.frame = frame

    def velocity(self) -> np.ndarray:
        with self.lock:
            return np.zeros(3) if self.v is None else self.v.copy()
