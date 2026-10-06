#!/usr/bin/env python3
"""Manual gamepad flight (velocity mode, mocap position hold) with RB toggling a trained safe_cf policy.

    python3 crazyflie/fly_policy.py --policy policy/models/safecf_moving_cf21bl_ent001.npz

LT = deadman (release = motors off), RB = manual <-> policy (flies to the target: launch spot, or set in RViz with --viz),
A = land; --loop: Y = fly again, B = quit. Right stick: horizontal velocity
(heading-relative, --room-frame for mocap x/y); left stick: vertical = climb, horizontal = yaw rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "policy"))
sys.path.insert(0, str(HERE.parent / "mocap"))
from hover import (Gamepad, Keyboard, NoKeyboard, Telemetry, arm_with_deadman, configure, connect,  # noqa: E402
                   preflight, save_logs, tilt_deg, yaw_deg)
from mlp_policy import MlpPolicy  # noqa: E402
from natnet import NatNetClient  # noqa: E402
from safe_cf_obs import (ARENA_HEIGHT, ARENA_RADIUS, DRONE_RADIUS, TARGET_KEEP_OUT_EXTRA, TARGET_RADIUS, TARGET_Z,  # noqa: E402
                         UNSAFE_MARGIN, V_MAX, Obstacle, SafeCFObs)

RATE = 240.0                 # mocap frame rate
AB_ALPHA, AB_BETA = 0.628, 0.305   # host alpha-beta (Kalata lambda 0.5): velocity for the policy obs


class AlphaBeta:
    """Velocity from mocap positions timed by frame numbers (exact), restarted after gaps > 0.1 s."""

    def __init__(self):
        self.p = self.v = None
        self.frame = None
        self.lock = threading.Lock()

    def update(self, frame: int, pos) -> None:
        z = np.asarray(pos, float)
        with self.lock:
            if self.p is None or frame - self.frame > 0.1 * RATE or frame <= self.frame:
                self.p, self.v = z.copy(), np.zeros(3)
            else:
                dt = (frame - self.frame) / RATE
                pred = self.p + self.v * dt
                r = z - pred
                self.p = pred + AB_ALPHA * r
                self.v = self.v + (AB_BETA / dt) * r
            self.frame = frame

    def velocity(self) -> np.ndarray:
        with self.lock:
            return np.zeros(3) if self.v is None else self.v.copy()


class Edge:
    def __init__(self):
        self.prev = False

    def __call__(self, now: bool) -> bool:
        rising = now and not self.prev
        self.prev = now
        return rising


def quat_rotate(q, v) -> np.ndarray:
    """Rotate v by the unit quaternion q = (w, x, y, z)."""
    w, u = q[0], np.asarray(q[1:], float)
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class ObstacleSet:
    """Policy obstacles: mocap-tracked rigid bodies (sphere centre = pivot + body-frame offset) plus static spheres."""

    def __init__(self, tracked=(), static=()):
        self.tracked = [dict(id=int(o["id"]), radius=float(o["radius"]), offset=np.asarray(o.get("offset", (0, 0, 0)), float))
                        for o in tracked]
        self.static = [Obstacle(o["pos"], float(o["radius"])) for o in static]
        self.est = {o["id"]: AlphaBeta() for o in self.tracked}
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


OBST_Z = (0.5, 2.0)        # training obstacle centres: z in the sample cylinder's range
OBST_R = 1.5               # ... and within ~1.5 m of the arena centre


def obstacle_ood(pos, slack=0.0) -> str | None:
    """Obstacle centre outside where training ever put one (slack < 0 shrinks the region, for hysteresis)."""
    r = math.hypot(pos[0], pos[1])
    if not OBST_Z[0] - slack <= pos[2] <= OBST_Z[1] + slack:
        return f"obstacle centre z {pos[2]:.2f} m outside the trained {OBST_Z[0]}-{OBST_Z[1]} m"
    if r > OBST_R + slack:
        return f"obstacle centre {r:.2f} m from the arena centre (trained <= {OBST_R} m)"
    return None


def load_obstacles(args) -> ObstacleSet:
    cfg = {}
    if args.obstacles and Path(args.obstacles).is_file():
        import yaml
        cfg = yaml.safe_load(Path(args.obstacles).read_text()) or {}
    tracked = cfg.get("obstacles") or []
    static = (cfg.get("static") or []) + [{"pos": o[:3], "radius": o[3]} for o in (args.obstacle or [])]
    if any(int(o["id"]) == args.body for o in tracked):
        raise ValueError(f"obstacle id {args.body} is the drone's own rigid body")
    if not tracked and not static:
        raise ValueError("no obstacles configured: the policy needs at least one. Track one (obstacles: - id) or add a static "
                         f"sphere at z {OBST_Z[0]}-{OBST_Z[1]} m within {OBST_R} m of the centre in {args.obstacles}")
    for o in tracked + static:
        if not 0.08 <= float(o["radius"]) <= 0.12:
            print(f"warning: obstacle radius {o['radius']} m is outside the trained 0.08-0.12 m")
    for o in static:
        why = obstacle_ood(o["pos"])
        if why:
            print(f"warning: static {why}; the policy hovers off target with such inputs")
    return ObstacleSet(tracked, static)


def sticks(pad, args) -> np.ndarray:
    """(forward, right, up, yaw_cw) in [-1, 1] with deadband; Linux js axes are + right / + down."""
    raw = [-(pad.axis(args.ax_ry) or 0.0), pad.axis(args.ax_rx) or 0.0, -(pad.axis(args.ax_ly) or 0.0), pad.axis(args.ax_lx) or 0.0]
    out = np.array([0.0 if abs(x) < args.deadband else (x - math.copysign(args.deadband, x)) / (1 - args.deadband) for x in raw])
    return np.clip(out, -1.0, 1.0)


def stick_velocity(s, yaw_rad, args) -> tuple:
    """Sticks -> world velocity (m/s) and CCW yaw rate (deg/s)."""
    fwd, left = s[0] * args.max_xy, -s[1] * args.max_xy
    if args.room_frame:
        vx, vy = fwd, left
    else:
        c, si = math.cos(yaw_rad), math.sin(yaw_rad)
        vx, vy = c * fwd - si * left, si * fwd + c * left
    return np.array([vx, vy, s[2] * args.max_z]), -s[3] * args.max_yaw_rate


def keep_inside(v, p, args) -> np.ndarray:
    """Manual mode: drop velocity components that push further past the soft arena boundary."""
    v = v.copy()
    r = math.hypot(p[0], p[1])
    if r > ARENA_RADIUS - args.soft_wall and r > 1e-6:
        out = np.array([p[0] / r, p[1] / r])
        radial = v[:2] @ out
        if radial > 0:
            v[:2] -= radial * out
    if (p[2] > args.soft_top and v[2] > 0) or (p[2] < args.soft_bottom and v[2] < 0):
        v[2] = 0.0
    return v


def outside_soft(p, args) -> str | None:
    if math.hypot(p[0], p[1]) > ARENA_RADIUS - args.soft_wall:
        return "near the arena wall"
    if p[2] > args.soft_top:
        return "near the ceiling"
    if p[2] < args.soft_bottom:
        return "too low"
    return None


def engage_problem(p, v) -> str | None:
    if math.hypot(p[0], p[1]) > TARGET_RADIUS:
        return f"outside the trained target region (r {math.hypot(p[0], p[1]):.2f} > {TARGET_RADIUS} m)"
    if not TARGET_Z[0] <= p[2] <= TARGET_Z[1]:
        return f"height {p[2]:.2f} m outside the trained {TARGET_Z[0]}-{TARGET_Z[1]} m"
    if np.linalg.norm(v) > 0.3:
        return f"moving {np.linalg.norm(v):.2f} m/s (> 0.3); hold still first"
    return None


def braking_point(p, v, args) -> np.ndarray:
    """Where the drone stops if it brakes now: outward-radial and vertical speed, v*delay + v^2/(2*a)."""
    q = np.array(p, float)
    r = math.hypot(q[0], q[1])
    stop = lambda u: u * args.brake_delay + u * u / (2.0 * args.brake_acc)
    if r > 1e-6:
        v_out = max(0.0, (v[0] * q[0] + v[1] * q[1]) / r)
        q[:2] *= (r + stop(v_out)) / r
    q[2] += math.copysign(stop(abs(v[2])), v[2])
    return q


def inside_soft(p, args, margin: float = 0.05) -> tuple:
    """Nearest point at least `margin` inside the soft arena boundary: the hold point after a boundary exit."""
    q = np.array(p, float)
    r_max = ARENA_RADIUS - args.soft_wall - margin
    r = math.hypot(q[0], q[1])
    if r > r_max:
        q[:2] *= r_max / r
    q[2] = min(max(q[2], args.soft_bottom + margin), args.soft_top - margin)
    return tuple(q)


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
    """Policy target: the launch spot at takeoff height by default, moved live (clamped + checked) from RViz."""

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


def estop_reason(args, kb, pad, body, age, tel, z0, tilt_armed) -> str | None:
    if not pad.trigger_held(args.deadman_axis, args.deadman_hold):
        a = pad.axis(args.deadman_axis)
        return f"deadman (LT) released (reads {a if a is None else round(a, 2)})" if pad.alive else "gamepad disconnected"
    if kb.kill:
        return "SPACE"
    if age > 0.25:
        return f"mocap lost {age:.2f} s"
    if tel.get("mocapEst.ageMs", 65535) > 250:
        return f"drone has no mocap for {tel.get('mocapEst.ageMs', 65535)} ms"
    if tel.age("pos") > 0.5:
        return f"no telemetry for {tel.age('pos'):.1f} s"
    p = body.position
    if math.hypot(p[0], p[1]) > ARENA_RADIUS - 0.05 or p[2] > ARENA_HEIGHT - 0.1:
        return f"left the arena at ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})"
    if tilt_armed and tilt_deg(body.quat) > args.max_tilt:
        return f"tilt {tilt_deg(body.quat):.0f} deg"
    est = [tel.get(f"stateEstimate.{a}") for a in "xyz"]
    if math.dist(p, est) > 0.10:
        return f"onboard estimate {math.dist(p, est) * 100:.0f} cm from mocap"
    return None


def session(cf, args, mocap, est, tel, kb, pad, policy, obs, obstacle_set, p0, rows, target_param=None, events=None,
            shared=None) -> str:
    """100 Hz: launch -> manual <-> policy (RB, flies to target_param) -> land (A); every tick checks the E-STOP conditions."""
    x0, y0, z0 = p0
    mode, t_start = "launch", time.monotonic()
    t_climb = z_climb = hold = hold_yaw = target = t_land = z_land0 = ground_since = None
    target_param = target_param if target_param is not None else TargetParam()
    target_param.reset((x0, y0, z0 + args.height))
    events = events if events is not None else []
    shared = shared if shared is not None else {}
    ood_note = None
    last_cmd = np.zeros(3)
    rb, a_btn = Edge(), Edge()
    rb(pad.held(args.btn_rb))
    a_btn(pad.held(args.btn_a))
    period, next_t = 0.01, time.monotonic()

    def event(what):
        events.append((time.monotonic(), mode, what))
        print(f"  {what}")

    def to_manual(why, p, yaw, hold_at=None):
        nonlocal mode, hold, hold_yaw
        if mode == "policy":
            for k, val in (("kiX", args.ki_xy), ("kiY", args.ki_xy), ("kiZ", args.ki_z)):
                cf.param.set_value(f"velCtl.{k}", val)
        event(f"manual ({why})")
        mode, hold, hold_yaw = "manual", tuple(hold_at if hold_at is not None else p), yaw

    while True:
        try:
            now = time.monotonic()
            body, age = mocap.body(args.body)
            tilt_armed = t_climb is not None and now - t_climb > 0.5
            reason = estop_reason(args, kb, pad, body, age, tel, z0, tilt_armed)
            if reason:
                cf.commander.send_stop_setpoint()
                cf.supervisor.send_emergency_stop()
                return f"E-STOP: {reason}"
            p, v = np.array(body.position), est.velocity()
            yaw = yaw_deg(body.quat)
            s = sticks(pad, args)
            rb_hit, a_hit = rb(pad.held(args.btn_rb)), a_btn(pad.held(args.btn_a))
            send_vel, vcmd, yawrate, action, o = False, np.zeros(3), 0.0, None, None

            if mode == "launch":
                if p[2] > z0 + args.launch_height and tilt_deg(body.quat) < 20.0:
                    event("lift-off: climbing")
                    mode, t_climb, z_climb = "climb", now, p[2]
                elif now - t_start > args.launch_timeout:
                    cf.commander.send_stop_setpoint()
                    return f"no lift-off within {args.launch_timeout:.1f} s: motors stopped"
                else:
                    thrust = int(args.launch_thrust * 65535)
                    cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
                    rows.append((now, "launch", *p, *v, *([math.nan] * 9), 0.0, *s, *([math.nan] * 3)))
                    next_t += period
                    time.sleep(max(0.0, next_t - time.monotonic()))
                    continue
            if mode == "climb":
                frac = min(1.0, (now - t_climb) / args.ramp_time)
                hold, hold_yaw = (x0, y0, z_climb + (z0 + args.height - z_climb) * frac), yaw if hold_yaw is None else hold_yaw
                if frac >= 1.0:
                    to_manual("takeoff complete", hold, hold_yaw)

            if a_hit and mode in ("manual", "policy"):
                if mode == "policy":
                    to_manual("A pressed", p, yaw)
                event("landing (A)")
                mode, t_land, z_land0, hold, hold_yaw = "land", now, p[2], (p[0], p[1]), yaw
            obstacles, obs_age, obs_id = obstacle_set.snapshot(now)
            if obstacles:
                why = obstacle_ood(SafeCFObs.closest(p, obstacles).pos, slack=0.0 if ood_note is None else -0.05)
                if (why is None) != (ood_note is None):
                    ood_note = why
                    event(f"warning: closest {why}" if why else "closest obstacle back inside the trained positions")
            if mode == "policy" and obs_age > args.obstacle_timeout:
                to_manual(f"obstacle {obs_id} not tracked for {obs_age:.2f} s", p, yaw)
            if mode == "manual" and rb_hit:
                target = target_param.get()
                problem = (engage_problem(p, v) or target_problem(target, obstacles)
                           or (f"obstacle {obs_id} not tracked for {obs_age:.2f} s" if obs_age > args.obstacle_fresh else None))
                if problem:
                    event(f"policy refused: {problem}")
                else:
                    for k in ("kiX", "kiY", "kiZ"):
                        cf.param.set_value(f"velCtl.{k}", 0.0)
                    obs.reset(np.clip(last_cmd / V_MAX, -1.0, 1.0))
                    event(f"POLICY: target ({target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}), integrators frozen")
                    mode = "policy"
            elif mode == "policy" and not a_hit:
                soft = outside_soft(braking_point(p, v, args), args)
                if rb_hit:
                    to_manual("RB", p, yaw)
                elif np.abs(s).max() > args.override:
                    to_manual("stick override", p, yaw)
                elif soft:
                    to_manual(f"would stop {soft} at {np.linalg.norm(v):.2f} m/s", p, yaw, inside_soft(p, args))

            if mode == "policy":
                target = target_param.get()
                seen = [Obstacle(ob.pos, ob.radius, (ob.vel[0], ob.vel[1], 0.0)) for ob in obstacles] if args.zero_obstacle_vz \
                    else obstacles
                o = obs.build(p, v, target, seen)
                action = policy(o)
                if not np.all(np.isfinite(action)):
                    to_manual("non-finite policy output", p, yaw)
                else:
                    obs.push(np.clip(action, -args.history_clip, args.history_clip) if args.history_clip else action)
                    vcmd, send_vel = np.clip(action, -1.0, 1.0) * V_MAX, True
            if mode == "manual":
                if np.abs(s).max() > 0.0:
                    vcmd, yawrate = stick_velocity(s, math.radians(yaw), args)
                    vcmd, send_vel, hold = keep_inside(vcmd, p, args), True, None
                elif hold is None:
                    hold, hold_yaw = inside_soft(p, args), yaw
            if mode == "land":
                hold = (hold[0], hold[1], max(z0 - 0.1, z_land0 - args.land_speed * (now - t_land)))
                ground_since = (ground_since or now) if p[2] < z0 + 0.05 else None
                if ground_since and now - ground_since > 0.5:
                    return "landed"
                if now - t_land > (z_land0 - z0) / args.land_speed + 3.0:
                    return "landed (timeout)"

            if send_vel:
                cf.commander.send_velocity_world_setpoint(*vcmd, yawrate)
                last_cmd = vcmd
                sp = (math.nan, math.nan, math.nan)
            else:
                cf.commander.send_position_setpoint(*hold, hold_yaw if hold_yaw is not None else yaw)
                last_cmd = np.zeros(3)
                sp = hold
            shared.update(mode=mode, cmd=list(vcmd) if send_vel else None, hold=list(hold) if not send_vel else None)
            tgt = target if mode == "policy" else (math.nan,) * 3
            act = action if action is not None else (math.nan,) * 3
            rows.append((now, mode, *p, *v, *tgt, *sp, *(vcmd if send_vel else (math.nan,) * 3), yawrate, *s, *act)
                        + (tuple(o) if o is not None else ()))
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
        except KeyboardInterrupt:
            if mode != "land":
                print("  Ctrl+C: landing (Ctrl+C again = E-STOP)")
                mode, t_land, z_land0, hold = "land", time.monotonic(), mocap.body(args.body)[0].position[2], tuple(p[:2])
            else:
                kb.kill = True


def configure_history(args, policy) -> str | None:
    """Training conventions from the policy file: action-history clip; refuses guards that would contradict training."""
    if args.zero_obstacle_vz and policy.obstacle_vertical_motion:
        return "--zero-obstacle-vz: this policy was trained with vertical obstacle motion, so zeroing vz would mislead it"
    if policy.clip_actions:
        args.history_clip = policy.clip_actions
        print(f"action history clipped to +-{policy.clip_actions:g}, as in training (clip_actions in the policy file)")
    elif args.clip_action_history:
        args.history_clip = 1.0
        print("action history clipped to +-1 as a guard (this policy was trained on raw actions)")
    else:
        args.history_clip = None
    return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--policy", required=True, help="exported .npz (policy/export_policy.py)")
    ap.add_argument("--obstacles", default=str(HERE / "config/obstacles.yaml"), help="tracked + static obstacle config")
    ap.add_argument("--obstacle", nargs=4, type=float, action="append", metavar=("X", "Y", "Z", "R"),
                    help="extra virtual static obstacle (world, m)")
    ap.add_argument("--obstacle-fresh", type=float, default=0.1, help="[s] RB engages only if every obstacle is this fresh")
    ap.add_argument("--obstacle-timeout", type=float, default=0.2, help="[s] stale obstacle while the policy flies -> manual")
    ap.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7")
    ap.add_argument("--server", default="192.168.1.2")
    ap.add_argument("--body", type=int, default=23)
    ap.add_argument("--mass", type=float, default=0.044)
    ap.add_argument("--height", type=float, default=0.6, help="[m] takeoff height above the floor point")
    ap.add_argument("--ramp-time", type=float, default=2.5)
    ap.add_argument("--launch-thrust", type=float, default=1.0)
    ap.add_argument("--launch-height", type=float, default=0.05)
    ap.add_argument("--launch-timeout", type=float, default=1.0)
    ap.add_argument("--ki-xy", type=float, default=1.0, help="manual-mode integrators (frozen at 0 while the policy flies)")
    ap.add_argument("--ki-z", type=float, default=2.0)
    ap.add_argument("--yaw-trim", type=float, default=0.0)
    ap.add_argument("--min-vbat", type=float, default=3.85, help="[V] resting; at 3.78 V the burst lifted only 4-6 cm in 1 s")
    ap.add_argument("--max-tilt", type=float, default=60.0)
    ap.add_argument("--land-speed", type=float, default=0.2)
    ap.add_argument("--max-xy", type=float, default=0.5, help="[m/s] manual stick full scale")
    ap.add_argument("--max-z", type=float, default=0.3)
    ap.add_argument("--max-yaw-rate", type=float, default=60.0, help="[deg/s]")
    ap.add_argument("--room-frame", action="store_true", help="right stick in mocap x/y instead of heading-relative")
    ap.add_argument("--deadband", type=float, default=0.1)
    ap.add_argument("--override", type=float, default=0.5, help="stick deflection that takes control back from the policy")
    ap.add_argument("--soft-wall", type=float, default=0.15, help="[m] inside the arena wall: policy -> manual")
    ap.add_argument("--brake-delay", type=float, default=0.06, help="[s] policy -> manual when the braking point crosses a soft limit")
    ap.add_argument("--brake-acc", type=float, default=3.0, help="[m/s^2] assumed braking (fitted tilt cap 3.5)")
    ap.add_argument("--zero-obstacle-vz", action="store_true", help="guard: feed the policy horizontal obstacle velocity only")
    ap.add_argument("--clip-action-history", action="store_true",
                    help="guard for policies trained on raw actions; automatic for policies exported with clip_actions")
    ap.add_argument("--viz", action="store_true", help="stream state to crazyflie/viz/rviz_bridge.py; accept its target moves")
    ap.add_argument("--viz-port", type=int, default=47800, help="state -> bridge (localhost UDP)")
    ap.add_argument("--viz-cmd-port", type=int, default=47801, help="bridge -> target requests (localhost UDP)")
    ap.add_argument("--soft-top", type=float, default=2.6)
    ap.add_argument("--soft-bottom", type=float, default=0.25)
    ap.add_argument("--param", action="append", default=None, metavar="GROUP.NAME=VALUE",
                    help="firmware params (RAM); default: attitude kp 9 as in the sysID")
    ap.add_argument("--js", default="/dev/input/js0")
    ap.add_argument("--deadman-axis", type=int, default=2)
    ap.add_argument("--deadman-threshold", type=float, default=0.5)
    ap.add_argument("--deadman-hold", type=float, default=0.0)
    ap.add_argument("--btn-rb", type=int, default=5)
    ap.add_argument("--btn-a", type=int, default=0)
    ap.add_argument("--btn-y", type=int, default=3, help="--loop: fly again")
    ap.add_argument("--btn-b", type=int, default=1, help="--loop: quit")
    ap.add_argument("--loop", action="store_true", help="after each run, Y = fly again (restarts a locked drone), B = quit")
    ap.add_argument("--ax-lx", type=int, default=0)
    ap.add_argument("--ax-ly", type=int, default=1)
    ap.add_argument("--ax-rx", type=int, default=3)
    ap.add_argument("--ax-ry", type=int, default=4)
    ap.add_argument("--dry-run", type=float, nargs="?", const=15.0, default=None, metavar="SECONDS")
    ap.add_argument("--log-dir", default=str(HERE / "logs"))
    args = ap.parse_args(argv)
    args.param = args.param or ["pid_attitude.roll_kp=9", "pid_attitude.pitch_kp=9"]
    args.history_clip = 1.0 if args.clip_action_history else None
    return args


SUP_LOCKED = 1 << 6   # supervisor.info: locked after an emergency stop, cleared only by a restart
SUP_CRASHED = 1 << 7  # supervisor.info: tumbled/crashed; arming refused until recovered

CONTROL_HEADER = ["t", "mode", "x", "y", "z", "vx_est", "vy_est", "vz_est", "tx", "ty", "tz", "sp_x", "sp_y", "sp_z",
                  "cmd_vx", "cmd_vy", "cmd_vz", "yawrate", "s_fwd", "s_right", "s_up", "s_yaw", "a0", "a1", "a2"]


class DroneLink:
    """Radio link + configured firmware params + telemetry; reopened after a restart over the radio."""

    def __init__(self, args, cf_holder):
        self.args, self.cf_holder = args, cf_holder
        self.scf = self.cf = self.tel = None
        self.params = {}

    def open(self) -> bool:
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        self.scf = SyncCrazyflie(self.args.uri, cf=Crazyflie(rw_cache=str(Path(self.args.log_dir) / "cache")))
        err = connect(self.scf, timeout=20.0)
        if err:
            print(f"no radio link to {self.args.uri} ({err}); is the drone on?", file=sys.stderr)
            self.scf = None
            return False
        self.cf = self.scf.cf
        try:
            self.params = configure(self.cf, self.args)
        except RuntimeError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            self.close()
            return False
        self.tel = Telemetry(self.cf)
        self.tel.start()
        self.cf_holder["cf"] = self.cf
        time.sleep(1.5)
        return True

    def ok(self) -> bool:
        return self.tel is not None and self.tel.age("pos") < 1.0

    def needs_restart(self) -> bool:
        return self.tel is not None and bool(int(self.tel.get("supervisor.info", 0)) & (SUP_LOCKED | SUP_CRASHED))

    def close(self) -> None:
        self.cf_holder["cf"] = None
        if self.tel:
            self.tel.stop()
            self.tel = None
        if self.scf:
            self.scf.close_link()
            self.scf = None

    def reboot(self) -> bool:
        from cflib.utils.power_switch import PowerSwitch
        self.close()
        try:
            PowerSwitch(self.args.uri).stm_power_cycle()
        except Exception as exc:
            print(f"  radio restart failed ({exc})", file=sys.stderr)
        time.sleep(4.0)
        return self.open()


class VizLink:
    """--viz: state JSON to the RViz bridge at 30 Hz over localhost UDP; target requests come back the same way."""

    def __init__(self, args, mocap, obstacle_set, target_param, shared, tel_fn):
        self.args, self.mocap, self.obstacle_set = args, mocap, obstacle_set
        self.target_param, self.shared, self.tel_fn = target_param, shared, tel_fn
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.inp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.inp.bind(("127.0.0.1", args.viz_cmd_port))
        self.inp.settimeout(0.1)
        self.stop_evt = threading.Event()

    def start(self) -> "VizLink":
        threading.Thread(target=self._publish, daemon=True).start()
        threading.Thread(target=self._receive, daemon=True).start()
        return self

    def close(self) -> None:
        self.stop_evt.set()

    def state(self) -> dict:
        now = time.monotonic()
        body, age = self.mocap.body(self.args.body)
        obstacles, _, _ = self.obstacle_set.snapshot(now)
        tel = self.tel_fn()
        target = self.target_param.get()
        a = self.args
        return {
            "mode": self.shared.get("mode", "ground"), "cmd": self.shared.get("cmd"), "hold": self.shared.get("hold"),
            "drone": None if body is None else {"pos": list(body.position), "quat": list(body.quat), "age": age},
            "target": None if target is None else list(target), "msg": self.target_param.msg,
            "obstacles": [{"pos": list(o.pos), "radius": o.radius, "vel": list(o.vel)} for o in obstacles],
            "vbat": None if tel is None else tel.get("pm.vbat"),
            "arena": {"radius": ARENA_RADIUS, "height": ARENA_HEIGHT, "target_radius": TARGET_RADIUS, "target_z": list(TARGET_Z),
                      "soft_wall": a.soft_wall, "soft_top": a.soft_top, "soft_bottom": a.soft_bottom,
                      "unsafe_extra": DRONE_RADIUS + UNSAFE_MARGIN, "keep_out_extra": DRONE_RADIUS + UNSAFE_MARGIN + TARGET_KEEP_OUT_EXTRA},
        }

    def _publish(self) -> None:
        dst = ("127.0.0.1", self.args.viz_port)
        while not self.stop_evt.is_set():
            try:
                self.out.sendto(json.dumps(self.state()).encode(), dst)
            except OSError:
                pass
            time.sleep(1 / 30)

    def _receive(self) -> None:
        while not self.stop_evt.is_set():
            try:
                data, _ = self.inp.recvfrom(65535)
                msg = json.loads(data)
            except (socket.timeout, ValueError):
                continue
            except OSError:
                break
            if "target" in msg and len(msg["target"]) == 3:
                obstacles, _, _ = self.obstacle_set.snapshot(time.monotonic())
                print(f"  RViz: {self.target_param.request(msg['target'], obstacles)}")


def wait_for_restart(args, pad) -> bool:
    """After a run: Y = fly again (True), B = quit (False); other buttons are printed to help with the mapping."""
    print(f"\nY = fly again, B = quit (buttons {args.btn_y} / {args.btn_b}; Ctrl+C also quits)")
    prev = dict(pad.buttons)
    while pad.alive:
        now = dict(pad.buttons)
        for b, down in now.items():
            if down and not prev.get(b):
                if b == args.btn_y:
                    return True
                if b == args.btn_b:
                    return False
                print(f"  button {b} pressed (Y is {args.btn_y}, B is {args.btn_b})")
        prev = now
        time.sleep(0.02)
    print("gamepad disconnected", file=sys.stderr)
    return False


def run_loop(args, link, fly, wait) -> int:
    """fly() once; with --loop keep offering Y = fly again, restarting the drone first if it is locked or unreachable."""
    code = fly()
    while args.loop and args.dry_run is None:
        if not wait():
            break
        if link.needs_restart() or not link.ok():
            print("  restarting the drone over the radio (E-STOP lock, crash flag or lost link) ...")
            if not link.reboot():
                print("  could not reconnect: check the drone, then Y to retry")
                continue
        code = fly()
    return code


def fly_once(args, link, mocap, est, pad, policy, obs, obstacle_set, sent, mocap_rows, target_param=None, shared=None) -> int:
    """Preflight -> deadman arm -> session -> disarm; logs only this run's slice of the shared streams."""
    run_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S_policy")
    cf, tel = link.cf, link.tel
    start = (len(mocap_rows), {k: len(v) for k, v in tel.rows.items()}, len(obstacle_set.rows))
    rows, events, outcome, kb = [], [], "aborted before arming", None
    meta = {"args": vars(args), "policy_meta": {k: str(v) for k, v in policy.meta.items()}, "params": link.params}
    try:
        p0, _ = preflight(args, mocap, tel, sent)
        if args.dry_run is not None:
            o = obs.build(np.array(p0) + [0, 0, args.height], est.velocity(), np.array(p0) + [0, 0, args.height],
                          obstacle_set.snapshot(time.monotonic())[0])
            print(f"Dry run: est |v| {np.linalg.norm(est.velocity()):.3f} m/s; policy at the takeoff target -> "
                  f"{np.round(np.clip(policy(o), -1, 1) * V_MAX, 3)} m/s; streaming {args.dry_run:.0f} s, never arming")
            time.sleep(args.dry_run)
            outcome = "dry run complete"
            return 0
        kb = Keyboard() if sys.stdin.isatty() else NoKeyboard()
        cancelled = arm_with_deadman(cf, args, pad, kb, tel, "policy session")
        if cancelled:
            outcome = cancelled
            return 1
        print("  RB = policy on/off, A = land, release LT = motors off")
        outcome = session(cf, args, mocap, est, tel, kb, pad, policy, obs, obstacle_set, p0, rows, target_param, events, shared)
        print(f"\n{outcome}")
        return 0 if outcome.startswith("landed") else 2
    except RuntimeError as exc:
        outcome = f"ABORT: {exc}"
        print(outcome, file=sys.stderr)
        return 1
    finally:
        cf.commander.send_stop_setpoint()
        cf.commander.send_notify_setpoint_stop()
        cf.supervisor.send_arming_request(False)
        if kb:
            kb.close()
        meta.update(outcome=outcome, extpose_sent=sent[0])
        tel_slice = type("TelSlice", (), {"rows": {k: v[start[1][k]:] for k, v in tel.rows.items()}})
        save_logs(run_dir, mocap_rows[start[0]:], tel_slice, [], meta)
        with open(run_dir / "control.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CONTROL_HEADER + [f"obs{i}" for i in range(obs.dim)])
            w.writerows(rows)
        with open(run_dir / "obstacles.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "frame", "id", "x", "y", "z", "vx", "vy", "vz"])
            w.writerows(obstacle_set.rows[start[2]:])
        with open(run_dir / "events.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "mode_before", "event"])
            w.writerows(events + [(time.monotonic(), "", outcome)])
        if shared is not None:
            shared.update(mode="ground", cmd=None, hold=None)
        print(f"logs: {run_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)
    import cflib.crtp

    policy = MlpPolicy(args.policy)
    problem = configure_history(args, policy)
    if problem:
        print(f"ABORT: {problem}", file=sys.stderr)
        return 1
    obs = SafeCFObs(moving=policy.obs_dim == 26)
    if obs.dim != policy.obs_dim:
        print(f"ABORT: policy expects {policy.obs_dim} obs, builder makes {obs.dim}", file=sys.stderr)
        return 1
    try:
        obstacle_set = load_obstacles(args)
    except ValueError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    print(f"policy {Path(args.policy).name} ({policy.obs_dim} obs); tracked obstacles "
          f"{[(o['id'], o['radius']) for o in obstacle_set.tracked] or 'none'}; static "
          + (", ".join(f"({o.pos[0]:+.2f}, {o.pos[1]:+.2f}, {o.pos[2]:+.2f}) r {o.radius}" for o in obstacle_set.static) or "none"))
    pad = None
    if args.dry_run is None:
        try:
            pad = Gamepad(args.js)
        except OSError as exc:
            print(f"ABORT: no gamepad at {args.js} ({exc})", file=sys.stderr)
            return 1

    cf_holder, sent, mocap_rows, est = {"cf": None}, [0], [], AlphaBeta()

    def on_frame(frame):
        obstacle_set.update(frame)
        b = frame.bodies.get(args.body)
        if b is None or not b.valid:
            return
        est.update(frame.number, b.position)
        mocap_rows.append((frame.arrival, frame.number, *b.position, *b.quat))
        cf = cf_holder["cf"]
        if cf is not None:
            w, qx, qy, qz = b.quat
            cf.extpos.send_extpose(x=b.position[0], y=b.position[1], z=b.position[2], qx=qx, qy=qy, qz=qz, qw=w)
            sent[0] += 1

    mocap = NatNetClient(args.server, on_frame=on_frame).start()
    link = DroneLink(args, cf_holder)
    target_param, shared, viz = TargetParam(), {"mode": "ground"}, None
    try:
        t0 = time.monotonic()
        while mocap.body(args.body)[0] is None:
            if time.monotonic() - t0 > 3.0:
                print(f"ABORT: rigid body {args.body} not tracked", file=sys.stderr)
                return 1
            time.sleep(0.02)
        time.sleep(0.5)
        found, age, missing = obstacle_set.snapshot(time.monotonic())
        if age > 0.2:
            streamed = sorted(mocap.last.bodies) if mocap.last else []
            print(f"ABORT: obstacle id {missing} not tracked (bodies in the stream: {streamed})", file=sys.stderr)
            return 1
        for o in found[len(obstacle_set.static):]:
            print(f"  tracked obstacle at ({o.pos[0]:+.2f}, {o.pos[1]:+.2f}, {o.pos[2]:+.2f}) r {o.radius}")
        p_now = mocap.body(args.body)[0].position
        target_param.reset((p_now[0], p_now[1], p_now[2] + args.height))
        if args.viz:
            viz = VizLink(args, mocap, obstacle_set, target_param, shared, lambda: link.tel).start()
            print(f"  viz: state -> udp://127.0.0.1:{args.viz_port}, targets <- :{args.viz_cmd_port} "
                  f"(run crazyflie/viz/run_rviz.sh)")

        cflib.crtp.init_drivers()
        if not link.open():
            return 1
        return run_loop(args, link,
                        lambda: fly_once(args, link, mocap, est, pad, policy, obs, obstacle_set, sent, mocap_rows,
                                         target_param, shared),
                        lambda: wait_for_restart(args, pad))
    except KeyboardInterrupt:
        print("\nquit", file=sys.stderr)
        return 1
    finally:
        if viz:
            viz.close()
        link.close()
        if pad:
            pad.close()
        mocap.stop()


if __name__ == "__main__":
    sys.exit(main())
