"""Policy observation of the safe_cf tasks (moving 26 / static 23 dims), mirroring isaac_drone_racer tasks/safe_cf/mdp.

World frame = mocap frame (Z up, origin = arena centre). Setpoint frame G = upright + target yaw (0 in training -> G = world).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ARENA_RADIUS = 6.0 * 0.3048        # safe_cf_env_cfg.ARENA_RADIUS
ARENA_HEIGHT = 9.5 * 0.3048        # safe_cf_env_cfg.ARENA_HEIGHT
TARGET_RADIUS = 1.4                 # safe_cf_env_cfg.SAMPLE_RADIUS: targets/spawns were drawn inside this cylinder
TARGET_Z = (0.5, 2.0)               # safe_cf_env_cfg.SAMPLE_Z
DRONE_RADIUS = 0.05                 # SafeCFCommandCfg.drone_radius
UNSAFE_MARGIN = 0.05                # SafeCFCommandCfg.unsafe_margin
BOUNDARY_UNSAFE_MARGIN = 0.10       # safe_cf_env_cfg.BOUNDARY_UNSAFE_MARGIN
TARGET_KEEP_OUT_EXTRA = 0.10        # SafeCFCommandCfg.spawn_clearance: obstacles kept r + drone + margin + this from the target
V_MAX = np.array([1.0, 1.0, 0.7])   # VelCtlActionCfg.v_max: action [-1, 1] -> m/s


@dataclass
class Obstacle:
    pos: np.ndarray                 # [m] world
    radius: float                   # [m] obstacle sphere radius (training U(0.08, 0.12))
    vel: np.ndarray = None          # [m/s] world

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)
        self.vel = np.zeros(3) if self.vel is None else np.asarray(self.vel, float)


class SafeCFObs:
    """Builds the policy obs; keeps the 3-step action history (oldest first, filled on reset like Isaac's CircularBuffer)."""

    def __init__(self, moving: bool = True, target_yaw: float = 0.0, history: int = 3, act_dim: int = 3):
        self.moving, self.history, self.act_dim = moving, history, act_dim
        c, s = math.cos(-target_yaw), math.sin(-target_yaw)
        self.rot_inv = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # world -> setpoint frame
        self.actions = np.zeros((history, act_dim))

    @property
    def dim(self) -> int:
        return 23 + (3 if self.moving else 0)

    def reset(self, action) -> None:
        self.actions[:] = np.asarray(action, float)

    def push(self, action) -> None:
        self.actions = np.roll(self.actions, -1, axis=0)
        self.actions[-1] = np.asarray(action, float)

    @staticmethod
    def closest(p, obstacles):
        """Obstacle with the smallest clearance to its unsafe sphere (SafeCFCommand.select_closest_obstacle)."""
        clear = [np.linalg.norm(p - o.pos) - (o.radius + DRONE_RADIUS + UNSAFE_MARGIN) for o in obstacles]
        return obstacles[int(np.argmin(clear))]

    def build(self, p, v, target, obstacles) -> np.ndarray:
        """p, v: drone world position / velocity; target: world setpoint; obstacles: list[Obstacle] (>= 1)."""
        p, v, target = np.asarray(p, float), np.asarray(v, float), np.asarray(target, float)
        R = self.rot_inv
        o = self.closest(p, obstacles)
        rel = R @ (o.pos - p)
        dist = np.linalg.norm(rel)
        unsafe = o.radius + DRONE_RADIUS + UNSAFE_MARGIN
        r = math.hypot(p[0], p[1])
        out_xy = p[:2] / r if r > 1e-6 else np.array([1.0, 0.0])
        out_g = (R @ np.array([out_xy[0], out_xy[1], 0.0]))[:2]
        clearance = (ARENA_RADIUS - DRONE_RADIUS - BOUNDARY_UNSAFE_MARGIN) - r
        parts = [R @ (p - target), R @ v, [p[2]], self.actions.reshape(-1), rel / max(dist, 1e-6),
                 [abs(dist - unsafe)], [out_g[0], out_g[1], clearance]]
        if self.moving:
            parts.append(R @ o.vel)
        return np.concatenate([np.asarray(x, float).reshape(-1) for x in parts])
