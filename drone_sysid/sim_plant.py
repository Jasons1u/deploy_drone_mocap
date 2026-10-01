"""Simulated Crazyflie behind the appchannel: delayed mu -> acceleration -> motion, for testing without hardware."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class PlantParams:
    alpha: tuple = (0.633, 0.633, 0.562)   # mu -> acceleration gain per axis
    drag: tuple = (0.80, 0.80, 0.0)        # 1/s
    hover_mu: float = 3.0                  # mu_z where a_z = 0
    trim_acc: tuple = (0.0, 0.0)           # constant horizontal acceleration bias [m/s^2]
    a_z_max: float = 1.15                  # vertical acceleration ceiling [m/s^2]
    delay: float = 0.02                    # packet -> firmware transport delay [s]
    pose_noise: float = 0.0005             # mocap position noise std [m]
    vbat: float = 3.9


class SimPlant:
    def __init__(self, p: PlantParams, start=(0.0, 0.0, 0.0), seed: int = 0):
        self.p = p
        self.alpha = np.array(p.alpha, dtype=float)
        self.drag = np.array(p.drag, dtype=float)
        self.b = np.array([p.trim_acc[0], p.trim_acc[1], -p.alpha[2] * p.hover_mu])
        self.r = np.array(start, dtype=float)
        self.v = np.zeros(3)
        self.t = 0.0
        self.cmds: deque = deque([(-1.0, 0, np.zeros(3))])
        self.rng = np.random.default_rng(seed)

    def send(self, flag: int, mu) -> None:
        """Queue a packet; it takes effect after the transport delay."""
        self.cmds.append((self.t, int(flag), np.array(mu, dtype=float)))

    def _active(self) -> tuple[int, np.ndarray]:
        while len(self.cmds) > 1 and self.cmds[1][0] <= self.t - self.p.delay:
            self.cmds.popleft()
        return self.cmds[0][1], self.cmds[0][2]

    def step(self, dt: float) -> None:
        flag, mu = self._active()
        on_ground = self.r[2] <= 0.0
        if flag:
            a = self.alpha * mu + self.b
            a[2] = min(a[2], self.p.a_z_max)
            a = a - self.drag * self.v
            a[2] = max(a[2], -G)
        else:
            a = np.array([0.0, 0.0, -G]) - self.drag * self.v
        if on_ground and a[2] <= 0.0:
            a[:] = 0.0
            self.v[:] = 0.0
        self.v += a * dt
        self.r += self.v * dt
        if self.r[2] < 0.0:
            self.r[2], self.v[2] = 0.0, max(self.v[2], 0.0)
        self.t += dt

    def pose(self) -> np.ndarray:
        return self.r + self.rng.normal(0.0, self.p.pose_noise, 3)
