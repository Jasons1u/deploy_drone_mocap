"""Obstacles the policy sees: mocap-tracked rigid bodies (sphere = pivot + offset) and virtual static spheres."""

from __future__ import annotations

import math
import threading

import numpy as np

from cfdeploy.estimation import AlphaBeta
from cfdeploy.geometry import quat_rotate
from cfdeploy.safe_cf_obs import OBSTACLE_R_MAX, OBSTACLE_RADIUS, OBSTACLE_Z, Obstacle


class ObstacleSet:
    """Tracked obstacles (alpha-beta velocity per body, updated from the mocap thread) plus static spheres."""

    def __init__(self, tracked, static, mocap_cfg):
        self.tracked = [dict(id=int(o["id"]), radius=float(o["radius"]), offset=np.asarray(o.get("offset", (0, 0, 0)), float))
                        for o in tracked]
        self.static = [Obstacle(o["pos"], float(o["radius"])) for o in static]
        self.est = {o["id"]: AlphaBeta(mocap_cfg) for o in self.tracked}
        self.centre, self.seen, self.rows = {}, {}, []
        self.lock = threading.Lock()

    def update(self, frame) -> None:
        """Called from the mocap thread for every frame."""
        for o in self.tracked:
            b = frame.bodies.get(o["id"])
            if b is None or not b.valid:
                continue
            c = np.asarray(b.position, float) + quat_rotate(b.quat, o["offset"])
            est = self.est[o["id"]]
            est.update(frame.number, c)
            with self.lock:
                self.centre[o["id"]], self.seen[o["id"]] = c, frame.arrival
                self.rows.append((frame.arrival, frame.number, o["id"], *c, *est.velocity()))

    def snapshot(self, now: float) -> tuple:
        """(obstacles, worst age [s] of the tracked ones, id of that one); never-seen bodies count as infinitely old."""
        out, worst, worst_id = list(self.static), 0.0, None
        with self.lock:
            for o in self.tracked:
                if o["id"] not in self.centre:
                    worst, worst_id = math.inf, o["id"]
                    continue
                age = now - self.seen[o["id"]]
                if age > worst:
                    worst, worst_id = age, o["id"]
                out.append(Obstacle(self.centre[o["id"]], o["radius"], self.est[o["id"]].velocity()))
        return out, worst, worst_id


def obstacle_ood(pos, slack=0.0) -> str | None:
    """Obstacle centre outside where training ever put one (slack < 0 shrinks the region, for hysteresis)."""
    r = math.hypot(pos[0], pos[1])
    if not OBSTACLE_Z[0] - slack <= pos[2] <= OBSTACLE_Z[1] + slack:
        return f"obstacle centre z {pos[2]:.2f} m outside the trained {OBSTACLE_Z[0]}-{OBSTACLE_Z[1]} m"
    if r > OBSTACLE_R_MAX + slack:
        return f"obstacle centre {r:.2f} m from the arena centre (trained <= {OBSTACLE_R_MAX} m)"
    return None


def load_obstacles(cfg) -> tuple:
    """cfg.obstacles -> (ObstacleSet, warnings); ValueError when nothing is configured or the drone is listed."""
    tracked, static = cfg.obstacles.tracked or [], cfg.obstacles.static or []
    if any(int(o["id"]) == cfg.mocap.drone_body for o in tracked):
        raise ValueError(f"obstacle id {cfg.mocap.drone_body} is the drone's own rigid body")
    if not tracked and not static:
        raise ValueError("no obstacles configured: the policy needs at least one. Track one (obstacles.tracked) or add a static "
                         f"sphere at z {OBSTACLE_Z[0]}-{OBSTACLE_Z[1]} m within {OBSTACLE_R_MAX} m of the centre (obstacles.static)")
    warnings = []
    for o in tracked + static:
        if not OBSTACLE_RADIUS[0] <= float(o["radius"]) <= OBSTACLE_RADIUS[1]:
            warnings.append(f"warning: obstacle radius {o['radius']} m is outside the trained {OBSTACLE_RADIUS[0]}-{OBSTACLE_RADIUS[1]} m")
    for o in static:
        why = obstacle_ood(o["pos"])
        if why:
            warnings.append(f"warning: static {why}; the policy hovers off target with such inputs")
    return ObstacleSet(tracked, static, cfg.mocap), warnings
