"""Desktop simulation world: the sysID drone model, hand-dragged or scripted obstacles, mocap sampling, the policy at 100 Hz."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from cfdeploy.drone_model import VelocityModel
from cfdeploy.estimation import AlphaBeta
from cfdeploy.logs import write_csv
from cfdeploy.mlp_policy import obs_for
from cfdeploy.obstacles import obstacle_ood
from cfdeploy.safe_cf_obs import ARENA_HEIGHT, ARENA_RADIUS, DRONE_RADIUS, V_MAX, Obstacle, SafeCFObs
from cfdeploy.safety import Hysteresis, braking_point, left_arena, outside_soft
from cfdeploy.targets import TargetParam, target_problem
from cfdeploy.viz_link import arena_state, obstacle_state

SIM_HEADER = ["t", "x", "y", "z", "vx", "vy", "vz", "vx_est", "vy_est", "vz_est", "tx", "ty", "tz", "a0", "a1", "a2",
              "cmd_vx", "cmd_vy", "cmd_vz", "obst", "ox", "oy", "oz", "ovx_est", "ovy_est", "ovz_est", "gap"]


# #### // Sim world: obstacles dragged like hand-held objects // ####
class SimObstacle:
    """Sphere chasing its RViz handle like a hand-moved object: critically damped, speed capped."""

    def __init__(self, pos, radius, omega, v_cap, mocap_cfg):
        self.home = np.array(pos, float)
        self.radius, self.omega, self.v_cap, self.mocap_cfg = float(radius), float(omega), float(v_cap), mocap_cfg
        self.reset()

    def reset(self) -> None:
        self.goal, self.p, self.prev, self.v = self.home.copy(), self.home.copy(), self.home.copy(), np.zeros(3)
        self.est, self.meas = AlphaBeta(self.mocap_cfg), self.home.copy()
        self.path, self.t = None, 0.0

    def follow(self, path) -> None:
        """Move exactly along path(t) [s since the sim started] instead of chasing the handle."""
        self.path = path

    def step(self, dt) -> None:
        if self.path is not None:
            self.t += dt
            self.prev, self.p = self.p, np.array(self.path(self.t), float)
            self.v = (self.p - self.prev) / dt
            return
        w = self.omega
        self.v = self.v + (w * w * (self.goal - self.p) - 2.0 * w * self.v) * dt
        s = float(np.linalg.norm(self.v))
        if s > self.v_cap:
            self.v *= self.v_cap / s
        self.prev, self.p = self.p, self.p + self.v * dt


def clamp_to_arena(xyz, limits) -> np.ndarray:
    q = np.array(xyz, float)
    r, r_max = math.hypot(q[0], q[1]), ARENA_RADIUS - limits.hard_wall_margin
    if r > r_max:
        q[:2] *= r_max / r
    q[2] = min(max(q[2], 0.0), ARENA_HEIGHT)
    return q


def clamp_drone(p) -> np.ndarray:
    """The drone stays inside the arena (walls, floor, ceiling)."""
    q = np.array(p, float)
    r, r_max = math.hypot(q[0], q[1]), ARENA_RADIUS - DRONE_RADIUS
    if r > r_max:
        q[:2] *= r_max / r
    q[2] = min(max(q[2], DRONE_RADIUS), ARENA_HEIGHT - DRONE_RADIUS)
    return q


# #### // Sim: policy at the training rate on the model at the physics rate; drone + obstacles seen through mocap // ####
class Sim:
    def __init__(self, cfg, policy, randomize: bool = False):
        self.cfg, self.policy, c = cfg, policy, cfg.sim
        self.dt = cfg.training.physics_dt
        self.decimation = int(round(cfg.training.policy_dt / self.dt))
        self.rng = np.random.default_rng(c.seed) if randomize else None
        self.noise_rng = np.random.default_rng(c.seed + 1)
        self.start = np.array(c.start, float)
        self.obstacles = [SimObstacle(o[:3], o[3], c.obstacle_omega, c.obstacle_max_speed, cfg.mocap) for o in c.obstacles]
        self.target = TargetParam()
        self.target.reset(self.start)
        self.obs = obs_for(policy)
        self.model = VelocityModel(cfg.drone_model, self.dt)
        self.t, self.frame, self.paused = 0.0, 0, False
        self.events, self.rows, self.flags = [], [], {}
        self.reset_drone()

    def event(self, what) -> None:
        self.events.append((round(self.t, 4), what))
        print(f"  [{self.t:7.2f} s] {what}")

    def reset_drone(self) -> None:
        self.model.reset(self.start, self.rng)
        self.p_prev, self.p_meas, self.est = self.start.copy(), self.start.copy(), AlphaBeta(self.cfg.mocap)
        self.obs.reset(np.zeros(3))
        self.cmd, self.action = np.zeros(3), np.zeros(3)
        self.min_gap, self.collisions, self.flags, self.ood = math.inf, 0, {}, Hysteresis()
        desc = "sysID fit" if self.rng is None else ", ".join(f"{k} {v:.3g}" for k, v in self.model.params.items())
        self.event(f"drone at ({self.start[0]:+.2f}, {self.start[1]:+.2f}, {self.start[2]:+.2f}), ROM {desc}")

    def reset_obstacles(self) -> None:
        for o in self.obstacles:
            o.reset()
        self.event("obstacles back to their start positions")

    def seen(self) -> list:
        return [Obstacle(o.meas, o.radius, o.est.velocity()) for o in self.obstacles]

    def handle(self, msg) -> None:
        """Requests from rviz_bridge.py: {"target": xyz} or {"obstacle": i, "pos": xyz}."""
        if "target" in msg and len(msg["target"]) == 3:
            self.event(f"RViz: {self.target.request(msg['target'], self.seen())}")
        elif "obstacle" in msg and len(msg.get("pos", ())) == 3 and 0 <= int(msg["obstacle"]) < len(self.obstacles):
            self.obstacles[int(msg["obstacle"])].goal = clamp_to_arena(msg["pos"], self.cfg.arena_limits)

    def tick(self) -> None:
        self.policy_step()
        for _ in range(self.decimation):
            self.physics_step()

    def policy_step(self) -> None:
        seen, target, v_est = self.seen(), self.target.get(), self.est.velocity()
        o = self.obs.build(self.p_meas, v_est, target, seen)
        a = self.policy(o)
        if not np.all(np.isfinite(a)):
            self.event("non-finite policy output: commanding zero velocity")
            a = np.zeros(3)
        self.obs.push(a)
        self.action, self.cmd = a, np.clip(a, -1.0, 1.0) * V_MAX
        gap, k = self.monitor(seen, target, v_est)
        c = seen[k]
        self.rows.append((round(self.t, 4), *self.model.p, *self.model.v, *v_est, *target, *a, *self.cmd, k, *self.obstacles[k].p,
                          *c.vel, gap))

    def physics_step(self) -> None:
        rate = self.cfg.mocap.rate
        self.p_prev = self.model.p.copy()
        self.model.step(self.cmd)
        self.model.p = clamp_drone(self.model.p)
        for o in self.obstacles:
            o.step(self.dt)
        self.t += self.dt
        while (self.frame + 1) / rate <= self.t + 1e-9:
            self.frame += 1
            w = 1.0 - (self.t - self.frame / rate) / self.dt   # interpolate to the frame time
            noise = lambda: self.noise_rng.normal(0.0, self.cfg.sim.mocap_noise, 3)
            self.p_meas = self.p_prev + w * (self.model.p - self.p_prev) + noise()
            self.est.update(self.frame, self.p_meas)
            for o in self.obstacles:
                o.meas = o.prev + w * (o.p - o.prev) + noise()
                o.est.update(self.frame, o.meas)

    def monitor(self, seen, target, v_est) -> tuple:
        """Warnings shown in RViz and logged once when they appear: what fly_policy would do, collisions, out-of-training."""
        flags, p, limits = {}, self.p_meas, self.cfg.arena_limits
        soft = outside_soft(braking_point(p, v_est, limits), limits)
        if soft:
            flags["soft"] = f"fly_policy would switch to manual here (would stop {soft})"
        if left_arena(p, limits):
            flags["arena"] = "fly_policy would E-STOP here (left the arena)"
        if p[2] < DRONE_RADIUS + 0.01:
            flags["floor"] = "drone on the floor"
        gaps = [float(np.linalg.norm(self.model.p - o.p)) - o.radius - DRONE_RADIUS for o in self.obstacles]
        k = int(np.argmin(gaps))
        self.min_gap = min(self.min_gap, gaps[k])
        if gaps[k] < 0.0:
            flags["hit"] = f"COLLISION with obstacle {k}"
        pos, slack, was_on = SafeCFObs.closest(p, seen).pos, -self.cfg.policy_flight.ood_hysteresis, self.ood.on
        self.ood.update(obstacle_ood(pos) is not None, obstacle_ood(pos, slack=slack) is None)
        if self.ood.on:
            flags["ood"] = f"closest {obstacle_ood(pos, slack=slack if was_on else 0.0)}"
        keep = target_problem(target, seen)
        if keep:
            flags["keepout"] = f"obstacle in the target keep-out, never seen in training ({keep})"
        if "hit" in flags and "hit" not in self.flags:
            self.collisions += 1
        for key in flags.keys() - self.flags.keys():
            self.event(flags[key])
        self.flags = flags
        return gaps[k], k

    def state(self) -> dict:
        target = self.target.get()
        lines = [self.target.msg or f"target ({target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f})", *self.flags.values(),
                 f"min gap {self.min_gap * 100:.0f} cm, collisions {self.collisions}"]
        return {
            "mode": "sim paused" if self.paused else "sim", "cmd": list(self.cmd), "hold": None,
            "drone": {"pos": list(self.model.p), "quat": [1.0, 0.0, 0.0, 0.0], "age": 0.0},
            "target": list(target), "msg": "\n".join(lines), "sim": True,
            "obstacles": obstacle_state([Obstacle(o.p, o.radius, o.est.velocity()) for o in self.obstacles]),
            "obstacle_goals": [list(o.goal) for o in self.obstacles], "vbat": None, "arena": arena_state(self.cfg.arena_limits),
        }

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "sim.csv", SIM_HEADER, self.rows)
        write_csv(out / "events.csv", ["t", "event"], self.events)
        meta = {"config": self.cfg, "policy_meta": {k: str(v) for k, v in self.policy.meta.items()}, "rom": self.model.params}
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
