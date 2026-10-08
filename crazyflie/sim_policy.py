#!/usr/bin/env python3
"""Desktop sim of a policy flight: the sysID drone model behind the deploy observation chain; drag target / obstacles in RViz."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # repo root: cfdeploy
from cfdeploy.config import add_override_args, apply_overrides, load, repo_path  # noqa: E402
from cfdeploy.drone_model import VelocityModel  # noqa: E402
from cfdeploy.estimation import AlphaBeta  # noqa: E402
from cfdeploy.logs import run_dir, write_csv  # noqa: E402
from cfdeploy.mlp_policy import MlpPolicy  # noqa: E402
from cfdeploy.obstacles import obstacle_ood  # noqa: E402
from cfdeploy.safe_cf_obs import (ARENA_HEIGHT, ARENA_RADIUS, DRONE_RADIUS, V_MAX, Obstacle, SafeCFObs, history_clip_note,  # noqa: E402
                                  obs_layout)
from cfdeploy.safety import braking_point, left_arena, outside_soft  # noqa: E402
from cfdeploy.targets import TargetParam, target_problem  # noqa: E402
from cfdeploy.viz_link import VizSocket, arena_state, obstacle_state  # noqa: E402

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

    def step(self, dt) -> None:
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
        self.obs = SafeCFObs(**obs_layout(policy.obs_dim), history_clip=getattr(policy, "clip_actions", None))
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
        self.min_gap, self.collisions, self.flags = math.inf, 0, {}
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
        closest = SafeCFObs.closest(p, seen)
        why = obstacle_ood(closest.pos, slack=-self.cfg.policy_flight.ood_hysteresis if "ood" in self.flags else 0.0)
        if why:
            flags["ood"] = f"closest {why}"
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


# #### // Terminal keys and the RViz process // ####
class Keys:
    """Single key presses from the terminal, no Enter needed."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def get(self) -> str | None:
        if select.select([self.fd], [], [], 0)[0]:
            return os.read(self.fd, 1).decode(errors="ignore").lower()
        return None

    def close(self) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def launch_rviz(cfg):
    """crazyflie/viz/run_rviz.sh (bridge + RViz) in its own process group on our ports; output to logs/rviz.log."""
    if not os.environ.get("DISPLAY"):
        print("no DISPLAY: not starting RViz (run crazyflie/viz/run_rviz.sh on a machine with a screen)")
        return None
    log = repo_path(cfg.paths.log_dir) / "rviz.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        proc = subprocess.Popen([str(HERE / "viz/run_rviz.sh"), "--state-port", str(cfg.viz.state_port), "--cmd-port", str(cfg.viz.cmd_port)],
                                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"RViz starting (log {log})")
    return proc


def stop_rviz(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=wait)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def interrupt(signum, frame):
    raise KeyboardInterrupt   # SIGTERM / SIGHUP (terminal closed) clean up like Ctrl+C, so RViz is not left behind


# #### // CLI // ####
def trained_obstacle_speed(policy) -> float:
    try:
        return float(str(policy.meta["obstacle_speed_range"]).strip("()[] ").split(",")[1])
    except (KeyError, IndexError, ValueError):
        return 1.6


def parse_args(argv=None) -> tuple:
    ap = argparse.ArgumentParser(description=__doc__, epilog="Starts RViz too (crazyflie/viz/run_rviz.sh); closing it ends the sim. "
                                 "Keys: r reset drone, o reset obstacles, p pause, q quit.")
    ap.add_argument("--policy", help="exported .npz (default policy_flight.policy)")
    ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "Z"), help="drone start and first target (default sim.start)")
    ap.add_argument("--obstacle", nargs=4, type=float, action="append", metavar=("X", "Y", "Z", "R"),
                    help="obstacle start, repeat for more (default sim.obstacles)")
    ap.add_argument("--randomize", action="store_true", help="draw model params, drift and kicks from the training ranges each reset")
    ap.add_argument("--seed", type=int, help="(default sim.seed)")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (0.5 = slow motion)")
    ap.add_argument("--duration", type=float, default=None, help="[s] sim time, then quit (default: until q / Ctrl+C)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--no-rviz", action="store_true", help="don't start RViz (e.g. it is already running)")
    add_override_args(ap)
    opts = ap.parse_args(argv)
    try:
        cfg = apply_overrides(load(), opts.set)
    except ValueError as exc:
        ap.error(str(exc))
    cfg.policy_flight.policy = str(Path(opts.policy).resolve() if opts.policy else repo_path(cfg.policy_flight.policy))
    for key, value in (("start", opts.start), ("obstacles", opts.obstacle), ("seed", opts.seed)):
        if value is not None:
            cfg.sim[key] = value
    return cfg, opts


def main(argv=None) -> int:
    cfg, opts = parse_args(argv)
    policy = MlpPolicy(cfg.policy_flight.policy)
    note = history_clip_note(policy.clip_actions)
    if note:
        print(note)
    if cfg.sim.obstacle_max_speed is None:
        cfg.sim.obstacle_max_speed = trained_obstacle_speed(policy)
    for o in cfg.sim.obstacles:
        why = obstacle_ood(o[:3])
        if why:
            print(f"warning: start {why}")
    try:
        link = VizSocket(cfg.viz)
    except OSError as exc:
        print(f"ABORT: udp:{cfg.viz.cmd_port} busy ({exc}); is fly_policy.py --viz running?", file=sys.stderr)
        return 1
    print(f"policy {Path(cfg.policy_flight.policy).name}; obstacle speed cap {cfg.sim.obstacle_max_speed:g} m/s; "
          f"{'randomized' if opts.randomize else 'sysID-fit'} ROM")
    sim = Sim(cfg, policy, opts.randomize)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, interrupt)
        except ValueError:
            pass   # not the main thread (tests)
    rviz = None if opts.no_rviz else launch_rviz(cfg)
    keys = Keys() if sys.stdin.isatty() else None
    print("keys: r reset drone, o reset obstacles, p pause, q quit")
    period, n = cfg.training.policy_dt / opts.speed, 0
    every = max(1, int(round(1.0 / (cfg.viz.rate * cfg.training.policy_dt))))
    next_t = time.monotonic()
    try:
        while opts.duration is None or sim.t < opts.duration:
            for msg in link.poll():
                sim.handle(msg)
            k = keys.get() if keys else None
            if k == "q":
                break
            if rviz is not None and rviz.poll() is not None:
                print(f"RViz closed (exit {rviz.returncode}): stopping the sim")
                break
            if k == "p":
                sim.paused = not sim.paused
                print("  paused" if sim.paused else "  running")
            elif k == "r":
                sim.reset_drone()
            elif k == "o":
                sim.reset_obstacles()
            if not sim.paused:
                sim.tick()
            if n % every == 0:
                link.send(sim.state())
            n += 1
            next_t += period
            lag = next_t - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.1:
                next_t = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if keys:
            keys.close()
        stop_rviz(rviz)
        link.close()
    print(f"min gap {sim.min_gap * 100:.1f} cm, collisions {sim.collisions} (since the last reset)")
    if not opts.no_log:
        out = run_dir(cfg, "sim")
        sim.save(out)
        print(f"log: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
