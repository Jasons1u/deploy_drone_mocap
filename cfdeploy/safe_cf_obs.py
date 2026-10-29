"""Policy observation of the safe_cf tasks (moving 26 / static 23 dims) in the mocap world frame (Z up, origin = arena centre)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cfdeploy.config import load

# #### // Training constants (config/deploy.yaml training section; must match Isaac config/crazyflie.yaml) // ####
T = load().training
FT = 0.3048
ARENA_RADIUS = T.arena_radius_ft * FT           # arena.radius_ft
ARENA_HEIGHT = T.arena_height_ft * FT           # arena.height_ft
TARGET_RADIUS = T.target_radius                 # sampling.radius: targets/spawns were drawn inside this cylinder
TARGET_Z = tuple(T.target_z)                    # sampling.z_range
DRONE_RADIUS = T.drone_radius                   # drone.radius
UNSAFE_MARGIN = T.unsafe_margin                 # obstacles.unsafe_margin
BOUNDARY_UNSAFE_MARGIN = T.boundary_unsafe_margin  # arena.boundary_unsafe_margin
TARGET_KEEP_OUT_EXTRA = T.target_keep_out_extra  # obstacles.spawn_clearance
V_MAX = np.array(T.v_max, float)                # rom.v_max: action [-1, 1] -> m/s
OBSTACLE_Z = tuple(T.obstacle_z)                # obstacle centres were sampled in this z range ...
OBSTACLE_R_MAX = T.obstacle_r_max               # ... within ~this of the arena centre
OBSTACLE_RADIUS = tuple(T.obstacle_radius)      # obstacles.radius_range
SLOT_CLEARANCE = tuple(T.slot_clearance)        # observations.slot_clearance (mdp.nearest_obstacles_geom clearance_range)
BASE_DIM, PER_OBSTACLE = 19, {False: 4, True: 7}   # pos 3 + vel 3 + height 1 + actions 9 + wall 3; dir 3 + clearance 1 (+ vel 3)


@dataclass
class Obstacle:
    pos: np.ndarray                 # [m] world
    radius: float                   # [m] obstacle sphere radius (training U(0.08, 0.12))
    vel: np.ndarray = None          # [m/s] world

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)
        self.vel = np.zeros(3) if self.vel is None else np.asarray(self.vel, float)


def obs_layout(obs_dim: int) -> dict:
    """SafeCFObs arguments for a policy input size: 23 static, 26 moving (closest obstacle), 33 moving (two nearest, Moving2)."""
    for moving in (True, False):
        n, rest = divmod(obs_dim - BASE_DIM, PER_OBSTACLE[moving])
        if rest == 0 and n in (1, 2):
            return dict(moving=moving, n_obstacles=n)
    raise ValueError(f"policy input of {obs_dim} values is not a safe_cf layout (23, 26 or 33)")


def history_clip_note(clip) -> str | None:
    return f"action history clipped to +-{clip:g}, as in training (clip_actions in the policy file)" if clip else None


class SafeCFObs:
    """Builds the policy obs; keeps the action history (oldest first, filled on reset like Isaac's CircularBuffer)."""

    def __init__(self, moving: bool = True, target_yaw: float = T.target_yaw, history: int = T.action_history, act_dim: int = 3,
                 history_clip: float | None = None, n_obstacles: int = 1):
        self.moving, self.history, self.act_dim, self.history_clip = moving, history, act_dim, history_clip
        self.n_obstacles = n_obstacles
        c, s = math.cos(-target_yaw), math.sin(-target_yaw)
        self.rot_inv = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # world -> setpoint frame G (upright + target yaw)
        self.actions = np.zeros((history, act_dim))

    @property
    def dim(self) -> int:
        return BASE_DIM + self.n_obstacles * PER_OBSTACLE[self.moving]

    def reset(self, action) -> None:
        self.actions[:] = np.asarray(action, float)

    def push(self, action) -> None:
        """Append the policy output; clipped first when the training wrapper clipped actions (policy clip_actions)."""
        c = self.history_clip
        self.actions = np.roll(self.actions, -1, axis=0)
        self.actions[-1] = np.asarray(np.clip(action, -c, c) if c else action, float)

    @staticmethod
    def closest(p, obstacles):
        """Obstacle with the smallest clearance to its unsafe sphere (SafeCFCommand.select_closest_obstacle)."""
        clear = [np.linalg.norm(p - o.pos) - (o.radius + DRONE_RADIUS + UNSAFE_MARGIN) for o in obstacles]
        return obstacles[int(np.argmin(clear))]

    def build(self, p, v, target, obstacles) -> np.ndarray:
        """p, v: drone world position / velocity; target: world setpoint; obstacles: list[Obstacle] (>= 1)."""
        p, v, target = np.asarray(p, float), np.asarray(v, float), np.asarray(target, float)
        R = self.rot_inv
        r = math.hypot(p[0], p[1])
        out_xy = p[:2] / r if r > 1e-6 else np.array([1.0, 0.0])
        out_g = (R @ np.array([out_xy[0], out_xy[1], 0.0]))[:2]
        wall = [out_g[0], out_g[1], (ARENA_RADIUS - DRONE_RADIUS - BOUNDARY_UNSAFE_MARGIN) - r]
        head = [R @ (p - target), R @ v, [p[2]], self.actions.reshape(-1)]
        parts = head + (self.closest_terms(p, obstacles, wall) if self.n_obstacles == 1 else self.slot_terms(p, obstacles, wall))
        return np.concatenate([np.asarray(x, float).reshape(-1) for x in parts])

    def closest_terms(self, p, obstacles, wall) -> list:
        """26 / 23 inputs: closest obstacle's direction, |distance - unsafe radius|, the arena wall, its velocity."""
        R, o = self.rot_inv, self.closest(p, obstacles)
        rel = R @ (o.pos - p)
        dist = np.linalg.norm(rel)
        unsafe = o.radius + DRONE_RADIUS + UNSAFE_MARGIN
        return [rel / max(dist, 1e-6), [abs(dist - unsafe)], wall] + ([R @ o.vel] if self.moving else [])

    def slot_terms(self, p, obstacles, wall) -> list:
        """33 inputs (mdp.nearest_obstacles_geom / _vel): wall, slots nearest first (dir, clipped signed clearance), slot velocities."""
        # an empty slot (fewer obstacles than slots) reads direction 0, clearance at the clip maximum, velocity 0
        R, (lo, hi) = self.rot_inv, SLOT_CLEARANCE
        clear = [np.linalg.norm(p - o.pos) - (o.radius + DRONE_RADIUS + UNSAFE_MARGIN) for o in obstacles]
        order = list(np.argsort(clear, kind="stable")[:self.n_obstacles])
        geom, vel = [], []
        for i in order:
            rel = R @ (obstacles[i].pos - p)
            geom += [rel / max(np.linalg.norm(rel), 1e-6), [min(max(clear[i], lo), hi)]]
            vel.append(R @ obstacles[i].vel)
        for _ in range(self.n_obstacles - len(order)):
            geom += [np.zeros(3), [hi]]
            vel.append(np.zeros(3))
        return [wall] + geom + vel
