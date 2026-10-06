#!/usr/bin/env python3
"""Desktop sim of a policy flight: Isaac's sysID velocity ROM + deploy obs chain; target and obstacles dragged in RViz."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import signal
import socket
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "policy"))
import fly_policy as fp  # noqa: E402
from mlp_policy import MlpPolicy  # noqa: E402
from safe_cf_obs import (ARENA_HEIGHT, ARENA_RADIUS, DRONE_RADIUS, TARGET_KEEP_OUT_EXTRA, TARGET_RADIUS, TARGET_Z,  # noqa: E402
                         UNSAFE_MARGIN, V_MAX, Obstacle, SafeCFObs)

PHYS_DT, DECIMATION = 1 / 400, 4      # Isaac safe_cf: sim.dt 1/400, policy every 4 steps (100 Hz)
G = 9.81
KV = np.array([4.0, 4.0, 6.0])        # VelCtlActionCfg.kv
NOMINAL = dict(kv_scale=1.0, tau_xy=0.077, tau_z=0.041, d_cmd_xy=0.022, d_cmd_z=0.020, d_fb=0.020, a_xy=3.53, drag_z=0.45)
RANGES = dict(kv_scale=(0.9, 1.1), tau_xy=(0.06, 0.095), tau_z=(0.03, 0.06), d_cmd_xy=(0.012, 0.035), d_cmd_z=(0.003, 0.035),
              d_fb=(0.015, 0.03), a_xy=(3.3, 3.7), drag_z=(0.3, 0.6))
A_Z_UP = A_Z_DOWN = 6.0
DRIFT_SIGMA, DRIFT_CLIP = np.array([0.04, 0.04, 0.02]), np.array([0.08, 0.08, 0.04])
KICK_PROB, KICK_STD = 0.0025, np.array([0.05, 0.05, 0.03])
DEFAULT_POLICY = HERE.parent / "policy/models/safecf_moving_good.npz"


class Rom:
    """Isaac VelCtlAction for one env: a_cmd = kv*(cmd(t-d_cmd) - v(t-d_fb)), capped, lagged by tau_a; v' = a - drag*v."""

    BUF = 32

    def reset(self, p0, rng=None) -> "Rom":
        d = {k: float(rng.uniform(*r)) for k, r in RANGES.items()} if rng is not None else dict(NOMINAL)
        steps = lambda s: int(round(s / PHYS_DT))
        self.params, self.rng = d, rng
        self.kv = KV * d["kv_scale"]
        self.tau = np.array([d["tau_xy"], d["tau_xy"], d["tau_z"]])
        self.a_xy, self.drag = d["a_xy"], np.array([0.0, 0.0, d["drag_z"]])
        self.d_cmd = np.array([steps(d["d_cmd_xy"]), steps(d["d_cmd_xy"]), steps(d["d_cmd_z"])])
        self.d_fb = steps(d["d_fb"])
        self.bias = np.zeros(3) if rng is None else np.clip(rng.normal(0.0, DRIFT_SIGMA), -DRIFT_CLIP, DRIFT_CLIP)
        self.cmd_buf, self.v_buf, self.i = np.zeros((self.BUF, 3)), np.zeros((self.BUF, 3)), 0
        self.p, self.v, self.a = np.array(p0, float), np.zeros(3), np.zeros(3)
        return self

    def step(self, cmd) -> None:
        i = self.i
        self.cmd_buf[i], self.v_buf[i] = cmd, self.v
        c = self.cmd_buf[(i - self.d_cmd) % self.BUF, np.arange(3)]
        v_fb = self.v_buf[(i - self.d_fb) % self.BUF]
        self.i = (i + 1) % self.BUF
        a_cmd = self.kv * (c - v_fb)
        h = math.hypot(a_cmd[0], a_cmd[1])
        if h > self.a_xy:
            a_cmd[:2] *= self.a_xy / h
        a_cmd[2] = min(max(a_cmd[2], -min(A_Z_DOWN, G)), A_Z_UP)
        self.a = self.a + (a_cmd - self.a) * (PHYS_DT / self.tau)
        self.v = np.clip(self.v + (self.a - self.drag * self.v) * PHYS_DT, -V_MAX, V_MAX)
        if self.rng is not None and self.rng.random() < KICK_PROB:
            self.v = np.clip(self.v + self.rng.normal(0.0, KICK_STD), -V_MAX, V_MAX)
        self.p = self.p + (self.v + self.bias) * PHYS_DT


class SimObstacle:
    """Sphere chasing its RViz handle like a hand-moved object: critically damped, speed capped."""

    def __init__(self, pos, radius, omega, v_cap):
        self.home = np.array(pos, float)
        self.radius, self.omega, self.v_cap = float(radius), float(omega), float(v_cap)
        self.reset()

    def reset(self) -> None:
        self.goal, self.p, self.prev, self.v = self.home.copy(), self.home.copy(), self.home.copy(), np.zeros(3)
        self.est, self.meas = fp.AlphaBeta(), self.home.copy()

    def step(self, dt) -> None:
        w = self.omega
        self.v = self.v + (w * w * (self.goal - self.p) - 2.0 * w * self.v) * dt
        s = float(np.linalg.norm(self.v))
        if s > self.v_cap:
            self.v *= self.v_cap / s
        self.prev, self.p = self.p, self.p + self.v * dt


def clamp_to_arena(xyz) -> np.ndarray:
    q = np.array(xyz, float)
    r, r_max = math.hypot(q[0], q[1]), ARENA_RADIUS - 0.05
    if r > r_max:
        q[:2] *= r_max / r
    q[2] = min(max(q[2], 0.0), ARENA_HEIGHT)
    return q


class Sim:
    """Policy at 100 Hz on the ROM at 400 Hz; drone and obstacles seen through 240 Hz mocap + the runner's alpha-beta."""

    def __init__(self, args, policy, runner):
        self.args, self.policy, self.runner = args, policy, runner
        self.rng = np.random.default_rng(args.seed) if args.randomize else None
        self.noise_rng = np.random.default_rng(args.seed + 1)
        self.start = np.array(args.start, float)
        self.obstacles = [SimObstacle(o[:3], o[3], args.obstacle_omega, args.obstacle_max_speed) for o in args.obstacle]
        self.target = fp.TargetParam()
        self.target.reset(self.start)
        self.obs = SafeCFObs(moving=policy.obs_dim == 26)
        self.rom = Rom()
        self.t, self.frame, self.paused = 0.0, 0, False
        self.events, self.rows, self.flags = [], [], {}
        self.reset_drone()

    def event(self, what) -> None:
        self.events.append((round(self.t, 4), what))
        print(f"  [{self.t:7.2f} s] {what}")

    def reset_drone(self) -> None:
        self.rom.reset(self.start, self.rng)
        self.p_prev, self.p_meas, self.est = self.start.copy(), self.start.copy(), fp.AlphaBeta()
        self.obs.reset(np.zeros(3))
        self.cmd, self.action = np.zeros(3), np.zeros(3)
        self.min_gap, self.collisions, self.flags = math.inf, 0, {}
        desc = "sysID fit" if self.rng is None else ", ".join(f"{k} {v:.3g}" for k, v in self.rom.params.items())
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
            self.obstacles[int(msg["obstacle"])].goal = clamp_to_arena(msg["pos"])

    def tick(self) -> None:
        self.policy_step()
        for _ in range(DECIMATION):
            self.physics_step()

    def policy_step(self) -> None:
        seen, target, v_est = self.seen(), self.target.get(), self.est.velocity()
        o = self.obs.build(self.p_meas, v_est, target, seen)
        a = self.policy(o)
        if not np.all(np.isfinite(a)):
            self.event("non-finite policy output: commanding zero velocity")
            a = np.zeros(3)
        hc = self.runner.history_clip
        self.obs.push(np.clip(a, -hc, hc) if hc else a)
        self.action, self.cmd = a, np.clip(a, -1.0, 1.0) * V_MAX
        gap, k = self.monitor(seen, target, v_est)
        c = seen[k]
        self.rows.append((round(self.t, 4), *self.rom.p, *self.rom.v, *v_est, *target, *a, *self.cmd, k, *self.obstacles[k].p,
                          *c.vel, gap))

    def physics_step(self) -> None:
        self.p_prev = self.rom.p.copy()
        self.rom.step(self.cmd)
        self.rom.p = self.clamp_drone(self.rom.p)
        for o in self.obstacles:
            o.step(PHYS_DT)
        self.t += PHYS_DT
        while (self.frame + 1) / fp.RATE <= self.t + 1e-9:
            self.frame += 1
            w = 1.0 - (self.t - self.frame / fp.RATE) / PHYS_DT   # interpolate to the frame time
            noise = lambda: self.noise_rng.normal(0.0, self.args.mocap_noise, 3)
            self.p_meas = self.p_prev + w * (self.rom.p - self.p_prev) + noise()
            self.est.update(self.frame, self.p_meas)
            for o in self.obstacles:
                o.meas = o.prev + w * (o.p - o.prev) + noise()
                o.est.update(self.frame, o.meas)

    @staticmethod
    def clamp_drone(p) -> np.ndarray:
        q = np.array(p, float)
        r, r_max = math.hypot(q[0], q[1]), ARENA_RADIUS - DRONE_RADIUS
        if r > r_max:
            q[:2] *= r_max / r
        q[2] = min(max(q[2], DRONE_RADIUS), ARENA_HEIGHT - DRONE_RADIUS)
        return q

    def monitor(self, seen, target, v_est) -> tuple:
        """Warnings shown in RViz and logged once when they appear: what fly_policy would do, collisions, out-of-training."""
        flags, p = {}, self.p_meas
        soft = fp.outside_soft(fp.braking_point(p, v_est, self.runner), self.runner)
        if soft:
            flags["soft"] = f"fly_policy would switch to manual here (would stop {soft})"
        if math.hypot(p[0], p[1]) > ARENA_RADIUS - 0.05 or p[2] > ARENA_HEIGHT - 0.1:
            flags["arena"] = "fly_policy would E-STOP here (left the arena)"
        if p[2] < DRONE_RADIUS + 0.01:
            flags["floor"] = "drone on the floor"
        gaps = [float(np.linalg.norm(self.rom.p - o.p)) - o.radius - DRONE_RADIUS for o in self.obstacles]
        k = int(np.argmin(gaps))
        self.min_gap = min(self.min_gap, gaps[k])
        if gaps[k] < 0.0:
            flags["hit"] = f"COLLISION with obstacle {k}"
        closest = SafeCFObs.closest(p, seen)
        why = fp.obstacle_ood(closest.pos, slack=-0.05 if "ood" in self.flags else 0.0)
        if why:
            flags["ood"] = f"closest {why}"
        keep = fp.target_problem(target, seen)
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
        a = self.runner
        return {
            "mode": "sim paused" if self.paused else "sim", "cmd": list(self.cmd), "hold": None,
            "drone": {"pos": list(self.rom.p), "quat": [1.0, 0.0, 0.0, 0.0], "age": 0.0},
            "target": list(target), "msg": "\n".join(lines), "sim": True,
            "obstacles": [{"pos": list(o.p), "radius": o.radius, "vel": list(o.est.velocity())} for o in self.obstacles],
            "obstacle_goals": [list(o.goal) for o in self.obstacles], "vbat": None,
            "arena": {"radius": ARENA_RADIUS, "height": ARENA_HEIGHT, "target_radius": TARGET_RADIUS, "target_z": list(TARGET_Z),
                      "soft_wall": a.soft_wall, "soft_top": a.soft_top, "soft_bottom": a.soft_bottom,
                      "unsafe_extra": DRONE_RADIUS + UNSAFE_MARGIN, "keep_out_extra": DRONE_RADIUS + UNSAFE_MARGIN + TARGET_KEEP_OUT_EXTRA},
        }

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "sim.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "vx_est", "vy_est", "vz_est", "tx", "ty", "tz", "a0", "a1", "a2",
                        "cmd_vx", "cmd_vy", "cmd_vz", "obst", "ox", "oy", "oz", "ovx_est", "ovy_est", "ovz_est", "gap"])
            w.writerows(self.rows)
        with open(run_dir / "events.csv", "w", newline="") as f:
            csv.writer(f).writerows([("t", "event"), *self.events])
        meta = {"args": vars(self.args), "policy_meta": {k: str(v) for k, v in self.policy.meta.items()}, "rom": self.rom.params}
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


class SimViz:
    """State JSON to rviz_bridge.py on fly_policy's --viz ports; target / obstacle drags come back."""

    def __init__(self, state_port, cmd_port):
        self.dst = ("127.0.0.1", state_port)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.inp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.inp.bind(("127.0.0.1", cmd_port))
        self.inp.setblocking(False)

    def send(self, state) -> None:
        try:
            self.out.sendto(json.dumps(state).encode(), self.dst)
        except OSError:
            pass

    def poll(self) -> list:
        msgs = []
        while True:
            try:
                data, _ = self.inp.recvfrom(65535)
            except (BlockingIOError, OSError):
                return msgs
            try:
                msgs.append(json.loads(data))
            except ValueError:
                pass

    def close(self) -> None:
        self.inp.close()
        self.out.close()


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


def launch_rviz(args):
    """crazyflie/viz/run_rviz.sh (bridge + RViz) in its own process group on our ports; output to logs/rviz.log."""
    if not os.environ.get("DISPLAY"):
        print("no DISPLAY: not starting RViz (run crazyflie/viz/run_rviz.sh on a machine with a screen)")
        return None
    log = Path(args.log_dir) / "rviz.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        proc = subprocess.Popen([str(HERE / "viz/run_rviz.sh"), "--state-port", str(args.viz_port), "--cmd-port", str(args.viz_cmd_port)],
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, epilog="Starts RViz too (crazyflie/viz/run_rviz.sh); closing it ends the sim. "
                                 "Keys: r reset drone, o reset obstacles, p pause, q quit.")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY), help="exported .npz (policy/export_policy.py)")
    ap.add_argument("--start", nargs=3, type=float, default=[0.0, 0.0, 1.0], metavar=("X", "Y", "Z"),
                    help="drone start and first target (world, m)")
    ap.add_argument("--obstacle", nargs=4, type=float, action="append", metavar=("X", "Y", "Z", "R"),
                    help="obstacle start (world, m), repeat for more; default one at (0.8, 0, 1.0) r 0.10")
    ap.add_argument("--obstacle-omega", type=float, default=8.0, help="[rad/s] how tightly an obstacle follows its handle")
    ap.add_argument("--obstacle-max-speed", type=float, default=None,
                    help="[m/s] obstacle speed cap; default the trained maximum from the policy file (else 1.6)")
    ap.add_argument("--randomize", action="store_true", help="draw ROM params, drift and kicks from the training ranges each reset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mocap-noise", type=float, default=0.0003, help="[m] per-axis mocap position noise")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (0.5 = slow motion)")
    ap.add_argument("--duration", type=float, default=None, help="[s] sim time, then quit (default: until q / Ctrl+C)")
    ap.add_argument("--viz-port", type=int, default=47800, help="state -> rviz_bridge.py (localhost UDP)")
    ap.add_argument("--viz-cmd-port", type=int, default=47801, help="rviz_bridge.py -> target / obstacle requests")
    ap.add_argument("--log-dir", default=str(HERE / "logs"))
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--no-rviz", action="store_true", help="don't start RViz (e.g. it is already running)")
    args = ap.parse_args(argv)
    args.obstacle = args.obstacle or [[0.8, 0.0, 1.0, 0.10]]
    return args


def trained_obstacle_speed(policy) -> float:
    try:
        return float(str(policy.meta["obstacle_speed_range"]).strip("()[] ").split(",")[1])
    except (KeyError, IndexError, ValueError):
        return 1.6


def main(argv=None) -> int:
    args = parse_args(argv)
    policy = MlpPolicy(args.policy)
    runner = fp.parse_args(["--policy", args.policy])
    problem = fp.configure_history(runner, policy)
    if problem:
        print(f"ABORT: {problem}", file=sys.stderr)
        return 1
    if args.obstacle_max_speed is None:
        args.obstacle_max_speed = trained_obstacle_speed(policy)
    for o in args.obstacle:
        why = fp.obstacle_ood(o[:3])
        if why:
            print(f"warning: start {why}")
    try:
        link = SimViz(args.viz_port, args.viz_cmd_port)
    except OSError as exc:
        print(f"ABORT: udp:{args.viz_cmd_port} busy ({exc}); is fly_policy.py --viz running?", file=sys.stderr)
        return 1
    print(f"policy {Path(args.policy).name}; obstacle speed cap {args.obstacle_max_speed:g} m/s; "
          f"{'randomized' if args.randomize else 'sysID-fit'} ROM")
    sim = Sim(args, policy, runner)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, interrupt)
        except ValueError:
            pass   # not the main thread (tests)
    rviz = None if args.no_rviz else launch_rviz(args)
    keys = Keys() if sys.stdin.isatty() else None
    print("keys: r reset drone, o reset obstacles, p pause, q quit")
    period, n = 0.01 / args.speed, 0
    next_t = time.monotonic()
    try:
        while args.duration is None or sim.t < args.duration:
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
            if n % 3 == 0:
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
    if not args.no_log:
        run_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S_sim")
        sim.save(run_dir)
        print(f"log: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
