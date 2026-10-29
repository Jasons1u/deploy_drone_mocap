#!/usr/bin/env python3
"""Policy flights: manual gamepad flight, RB hands control to a trained safe_cf policy (LT deadman, A land, --loop, --viz)."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import add_override_args, apply_overrides, load, policy_path, set_firmware_params  # noqa: E402
from cfdeploy.estimation import AlphaBeta  # noqa: E402
from cfdeploy.flight import Landing, Takeoff, arm_with_deadman, preflight  # noqa: E402
from cfdeploy.gamepad import Edge, Gamepad, keyboard, stick_velocity, sticks  # noqa: E402
from cfdeploy.geometry import yaw_deg  # noqa: E402
from cfdeploy.logs import run_dir, save_flight_logs, save_link_logs, write_csv  # noqa: E402
from cfdeploy.mlp_policy import load_policy  # noqa: E402
from cfdeploy.natnet import NatNetClient  # noqa: E402
from cfdeploy.obstacles import load_obstacles, obstacle_ood  # noqa: E402
from cfdeploy.radio import DroneLink, MocapRelay, stop_and_disarm, wait_for_body  # noqa: E402
from cfdeploy.safe_cf_obs import TARGET_RADIUS, TARGET_Z, V_MAX, SafeCFObs  # noqa: E402
from cfdeploy.safety import Hysteresis, arena_limit, braking_point, estop_reason, inside_soft, keep_inside, outside_soft  # noqa: E402
from cfdeploy.targets import TargetParam, target_problem  # noqa: E402
from cfdeploy.viz_link import VizLink, arena_state, launch_rviz, obstacle_state, stop_rviz  # noqa: E402

CONTROL_HEADER = ["t", "mode", "x", "y", "z", "vx_est", "vy_est", "vz_est", "tx", "ty", "tz", "sp_x", "sp_y", "sp_z",
                  "cmd_vx", "cmd_vy", "cmd_vz", "yawrate", "s_fwd", "s_right", "s_up", "s_yaw", "a0", "a1", "a2"]
NAN3 = (math.nan,) * 3


# #### // Engage rule: the policy only starts where and how training started it // ####
def engage_problem(p, v, max_speed) -> str | None:
    if math.hypot(p[0], p[1]) > TARGET_RADIUS:
        return f"outside the trained target region (r {math.hypot(p[0], p[1]):.2f} > {TARGET_RADIUS} m)"
    if not TARGET_Z[0] <= p[2] <= TARGET_Z[1]:
        return f"height {p[2]:.2f} m outside the trained {TARGET_Z[0]}-{TARGET_Z[1]} m"
    if np.linalg.norm(v) > max_speed:
        return f"moving {np.linalg.norm(v):.2f} m/s (> {max_speed}); hold still first"
    return None


# #### // Flight session: launch -> climb -> manual <-> policy (RB) -> land (A); E-STOP checks every tick // ####
class PolicySession:
    def __init__(self, cf, cfg, mocap, est, tel, kb, pad, policy, obs, obstacle_set, p0, rows,
                 target_param=None, events=None, shared=None, clock=time, mocap_at=None):
        self.cf, self.cfg, self.mocap, self.est, self.tel, self.kb, self.pad = cf, cfg, mocap, est, tel, kb, pad
        self.mocap_at, self.radio_late, self.ood = mocap_at, Hysteresis(), Hysteresis()
        self.policy, self.obs, self.obstacle_set, self.rows, self.clock = policy, obs, obstacle_set, rows, clock
        self.p0, self.p = p0, np.array(p0, float)
        self.mode = "launch"
        self.takeoff = Takeoff(cfg.launch, p0, p0[2] + cfg.policy_flight.height, clock.monotonic())
        self.limit = arena_limit(cfg.arena_limits)
        self.hold = self.hold_yaw = self.landing = None
        self.target_param = target_param if target_param is not None else TargetParam()
        self.target_param.reset((p0[0], p0[1], p0[2] + cfg.policy_flight.height))
        self.events = events if events is not None else []
        self.shared = shared if shared is not None else {}
        self.last_cmd = np.zeros(3)
        self.rb, self.a_btn = Edge(), Edge()
        self.rb(pad.held(cfg.gamepad.buttons.rb))
        self.a_btn(pad.held(cfg.gamepad.buttons.a))

    def run(self) -> str:
        period, next_t = self.cfg.training.policy_dt, self.clock.monotonic()
        while True:
            try:
                outcome = self.tick()
                if outcome:
                    return outcome
                next_t += period
                self.clock.sleep(max(0.0, next_t - self.clock.monotonic()))
            except KeyboardInterrupt:
                self.on_ctrl_c()

    def event(self, what) -> None:
        self.events.append((self.clock.monotonic(), self.mode, what))
        print(f"  {what}")

    def to_manual(self, why, p, yaw, hold_at=None) -> None:
        if self.mode == "policy":
            d = self.cfg.drone
            for k, val in (("kiX", d.ki_xy), ("kiY", d.ki_xy), ("kiZ", d.ki_z)):
                self.cf.param.set_value(f"velCtl.{k}", val)
        self.event(f"manual ({why})")
        self.mode, self.hold, self.hold_yaw = "manual", tuple(hold_at if hold_at is not None else p), yaw

    def check_radio(self) -> None:
        """Warn (once, with hysteresis) while telemetry arrives late: a downlink backlog, the link is struggling."""
        tc, delay = self.cfg.telemetry, self.tel.latency("pos")
        if self.radio_late.update(delay > tc.delay_warn, delay < tc.delay_clear):
            self.event(f"radio: telemetry {delay * 1000:.0f} ms late (downlink backlog)" if self.radio_late.on
                       else "radio: telemetry back on time")

    def on_ctrl_c(self) -> None:
        if self.mode != "land":
            print("  Ctrl+C: landing (Ctrl+C again = E-STOP)")
            now, z = self.clock.monotonic(), self.mocap.body(self.cfg.mocap.drone_body)[0].position[2]
            self.mode, self.landing, self.hold = "land", Landing(self.cfg.landing, now, z, self.p0[2]), tuple(self.p[:2])
        else:
            self.kb.kill = True

    def tick(self) -> str | None:
        """One control step: E-STOP checks, then each phase in order; returns the outcome when the session ends."""
        now = self.clock.monotonic()
        body, age = self.mocap.body(self.cfg.mocap.drone_body)
        reason = estop_reason(self.cfg, self.kb, self.pad, body, age, self.tel, self.limit, self.takeoff.tilt_guard_armed(now),
                              self.mocap_at)
        if reason:
            self.cf.commander.send_stop_setpoint()
            self.cf.supervisor.send_emergency_stop()
            return f"E-STOP: {reason}"
        self.check_radio()
        t = self.read_inputs(now, body)
        if self.mode == "launch":
            outcome = self.launch(t)
            if self.mode == "launch":
                return outcome
        if self.mode == "climb":
            self.climb(t)
        self.on_a_press(t)
        self.watch_obstacles(t)
        self.engage_or_exit(t)
        outcome = self.command(t)
        if outcome:
            return outcome
        self.send_and_log(t)
        return None

    def read_inputs(self, now, body) -> "Tick":
        cfg = self.cfg
        p, v = np.array(body.position), self.est.velocity()
        self.p = p
        yaw = yaw_deg(body.quat)
        s = sticks(self.pad, cfg.gamepad.axes, cfg.sticks.deadband)
        rb_hit, a_hit = self.rb(self.pad.held(cfg.gamepad.buttons.rb)), self.a_btn(self.pad.held(cfg.gamepad.buttons.a))
        return Tick(now, p, v, yaw, s, rb_hit, a_hit)

    # #### // Launch burst and climb // ####
    def launch(self, t) -> str | None:
        """Thrust burst until lift-off (mode -> climb); motors stop and the session ends if it never lifts."""
        if self.takeoff.lifted_off(t.now, t.p):
            self.event("lift-off: climbing")
            self.mode = "climb"
        elif self.takeoff.timed_out(t.now):
            self.cf.commander.send_stop_setpoint()
            return f"no lift-off within {self.cfg.launch.timeout:.1f} s: motors stopped"
        else:
            self.cf.commander.send_setpoint(0.0, 0.0, 0.0, self.takeoff.thrust)
            self.rows.append((t.now, "launch", *t.p, *t.v, *([math.nan] * 9), 0.0, *t.s, *NAN3))
        return None

    def climb(self, t) -> None:
        hold, done = self.takeoff.climb(t.now)
        self.hold, self.hold_yaw = hold, t.yaw if self.hold_yaw is None else self.hold_yaw
        if done:
            self.to_manual("takeoff complete", self.hold, self.hold_yaw)

    # #### // Mode changes: A lands; obstacle checks; RB engages / exits the policy // ####
    def on_a_press(self, t) -> None:
        if t.a_hit and self.mode in ("manual", "policy"):
            if self.mode == "policy":
                self.to_manual("A pressed", t.p, t.yaw)
            self.event("landing (A)")
            self.mode, self.landing = "land", Landing(self.cfg.landing, t.now, t.p[2], self.p0[2])
            self.hold, self.hold_yaw = (t.p[0], t.p[1]), t.yaw

    def watch_obstacles(self, t) -> None:
        """Snapshot the obstacles, warn when the closest is somewhere training never put one, drop the policy if one is stale."""
        pf = self.cfg.policy_flight
        t.obstacles, t.obs_age, t.obs_id = self.obstacle_set.snapshot(t.now)
        if t.obstacles:
            pos = SafeCFObs.closest(t.p, t.obstacles).pos
            if self.ood.update(obstacle_ood(pos) is not None, obstacle_ood(pos, slack=-pf.ood_hysteresis) is None):
                self.event(f"warning: closest {obstacle_ood(pos)}" if self.ood.on else "closest obstacle back inside the trained positions")
        if self.mode == "policy" and t.obs_age > pf.obstacle_timeout:
            self.to_manual(f"obstacle {t.obs_id} not tracked for {t.obs_age:.2f} s", t.p, t.yaw)

    def engage_or_exit(self, t) -> None:
        """RB in manual engages the policy (if training covered this state); RB, a stick push or the braking point exit it."""
        pf, limits = self.cfg.policy_flight, self.cfg.arena_limits
        if self.mode == "manual" and t.rb_hit:
            target = self.target_param.get()
            problem = (engage_problem(t.p, t.v, pf.max_engage_speed) or target_problem(target, t.obstacles)
                       or (f"obstacle {t.obs_id} not tracked for {t.obs_age:.2f} s" if t.obs_age > pf.obstacle_fresh else None))
            if problem:
                self.event(f"policy refused: {problem}")
            else:
                for k in ("kiX", "kiY", "kiZ"):
                    self.cf.param.set_value(f"velCtl.{k}", 0.0)
                self.obs.reset(np.clip(self.last_cmd / V_MAX, -1.0, 1.0))
                self.event(f"POLICY: target ({target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}), integrators frozen")
                self.mode = "policy"
        elif self.mode == "policy" and not t.a_hit:
            soft = outside_soft(braking_point(t.p, t.v, limits), limits)
            if t.rb_hit:
                self.to_manual("RB", t.p, t.yaw)
            elif np.abs(t.s).max() > self.cfg.sticks.override:
                self.to_manual("stick override", t.p, t.yaw)
            elif soft:
                self.to_manual(f"would stop {soft} at {np.linalg.norm(t.v):.2f} m/s", t.p, t.yaw, inside_soft(t.p, limits))

    # #### // Command for the current mode, then send + log // ####
    def command(self, t) -> str | None:
        """Policy velocity, stick velocity or hold, or the landing descent; returns the outcome once landed."""
        limits = self.cfg.arena_limits
        if self.mode == "policy":
            t.target = self.target_param.get()
            t.o = self.obs.build(t.p, t.v, t.target, t.obstacles)
            t.action = self.policy(t.o)
            if not np.all(np.isfinite(t.action)):
                self.to_manual("non-finite policy output", t.p, t.yaw)
            else:
                self.obs.push(t.action)
                t.vcmd, t.send_vel = np.clip(t.action, -1.0, 1.0) * V_MAX, True
        if self.mode == "manual":
            if np.abs(t.s).max() > 0.0:
                t.vcmd, t.yawrate = stick_velocity(t.s, math.radians(t.yaw), self.cfg.sticks)
                t.vcmd, t.send_vel, self.hold = keep_inside(t.vcmd, t.p, limits), True, None
            elif self.hold is None:
                self.hold, self.hold_yaw = inside_soft(t.p, limits), t.yaw
        if self.mode == "land":
            self.hold = (self.hold[0], self.hold[1], self.landing.z(t.now))
            return self.landing.outcome(t.now, t.p)
        return None

    def send_and_log(self, t) -> None:
        if t.send_vel:
            self.cf.commander.send_velocity_world_setpoint(*t.vcmd, t.yawrate)
            self.last_cmd, sp = t.vcmd, NAN3
        else:
            self.cf.commander.send_position_setpoint(*self.hold, self.hold_yaw if self.hold_yaw is not None else t.yaw)
            self.last_cmd, sp = np.zeros(3), self.hold
        self.shared.update(mode=self.mode, cmd=list(t.vcmd) if t.send_vel else None, hold=list(self.hold) if not t.send_vel else None)
        tgt = t.target if self.mode == "policy" else NAN3
        act = t.action if t.action is not None else NAN3
        self.rows.append((t.now, self.mode, *t.p, *t.v, *tgt, *sp, *(t.vcmd if t.send_vel else NAN3), t.yawrate, *t.s, *act)
                         + (tuple(t.o) if t.o is not None else ()))


@dataclass
class Tick:
    """One control step's inputs and the command built from them, passed between the session's phases."""
    now: float
    p: np.ndarray
    v: np.ndarray
    yaw: float
    s: np.ndarray
    rb_hit: bool
    a_hit: bool
    obstacles: list = None
    obs_age: float = 0.0
    obs_id: int | None = None
    send_vel: bool = False
    vcmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yawrate: float = 0.0
    action: np.ndarray | None = None
    o: np.ndarray | None = None
    target: np.ndarray | None = None


# #### // One flight: preflight, arm, session, this run's logs // ####
def fly_once(cfg, opts, link, mocap, est, pad, policy, obs, obstacle_set, relay, target_param=None, shared=None) -> int:
    out = run_dir(cfg, "policy")
    cf, tel = link.cf, link.tel
    start = (len(relay.rows), {k: len(v) for k, v in tel.rows.items()}, len(obstacle_set.rows), len(link.stats), len(relay.sends))
    rows, events, outcome, kb = [], [], "aborted before arming", None
    meta = {"config": dict(cfg), "policy_meta": {k: str(v) for k, v in policy.meta.items()}, "params": link.params}
    try:
        p0, _ = preflight(cfg, mocap, tel, relay.sent)
        if opts.dry_run is not None:
            above = np.array(p0) + [0, 0, cfg.policy_flight.height]
            o = obs.build(above, est.velocity(), above, obstacle_set.snapshot(time.monotonic())[0])
            print(f"Dry run: est |v| {np.linalg.norm(est.velocity()):.3f} m/s; policy at the takeoff target -> "
                  f"{np.round(np.clip(policy(o), -1, 1) * V_MAX, 3)} m/s; streaming {opts.dry_run:.0f} s, never arming")
            time.sleep(opts.dry_run)
            outcome = "dry run complete"
            return 0
        kb = keyboard()
        cancelled = arm_with_deadman(cf, cfg, pad, kb, tel, "policy session")
        if cancelled:
            outcome = cancelled
            return 1
        print("  RB = policy on/off, A = land, release LT = motors off")
        session = PolicySession(cf, cfg, mocap, est, tel, kb, pad, policy, obs, obstacle_set, p0, rows, target_param, events, shared,
                                mocap_at=relay.position_at)
        outcome = session.run()
        print(f"\n{outcome}")
        return 0 if outcome.startswith("landed") else 2
    except RuntimeError as exc:
        outcome = f"ABORT: {exc}"
        print(outcome, file=sys.stderr)
        return 1
    finally:
        stop_and_disarm(cf)
        if kb:
            kb.close()
        meta.update(outcome=outcome, extpose_sent=relay.sent, extpose_skipped=relay.skipped, extpose_errors=relay.errors)
        save_flight_logs(out, relay.rows[start[0]:], {k: v[start[1][k]:] for k, v in tel.rows.items()}, [], meta)
        save_link_logs(out, link.stats[start[3]:], relay.sends[start[4]:])
        write_csv(out / "control.csv", CONTROL_HEADER + [f"obs{i}" for i in range(obs.dim)], rows)
        write_csv(out / "obstacles.csv", ["t", "frame", "id", "x", "y", "z", "vx", "vy", "vz"], obstacle_set.rows[start[2]:])
        write_csv(out / "events.csv", ["t", "mode_before", "event"], events + [(time.monotonic(), "", outcome)])
        if shared is not None:
            shared.update(mode="ground", cmd=None, hold=None)
        print(f"logs: {out}")


# #### // --loop: Y flies again (restarting a locked drone first), B quits // ####
def wait_for_restart(cfg, pad) -> bool:
    """After a run: Y = fly again (True), B = quit (False); other buttons are printed to help with the mapping."""
    y, b_quit = cfg.gamepad.buttons.y, cfg.gamepad.buttons.b
    print(f"\nY = fly again, B = quit (buttons {y} / {b_quit}; Ctrl+C also quits)")
    prev = dict(pad.buttons)
    while pad.alive:
        now = dict(pad.buttons)
        for b, down in now.items():
            if down and not prev.get(b):
                if b == y:
                    return True
                if b == b_quit:
                    return False
                print(f"  button {b} pressed (Y is {y}, B is {b_quit})")
        prev = now
        time.sleep(0.02)
    print("gamepad disconnected", file=sys.stderr)
    return False


def run_loop(opts, link, fly, wait) -> int:
    """fly() once; with --loop keep offering Y = fly again, restarting the drone first if it is locked or unreachable."""
    code = fly()
    while opts.loop and opts.dry_run is None:
        if not wait():
            break
        if link.needs_restart() or not link.ok():
            print("  restarting the drone over the radio (E-STOP lock, crash flag or lost link) ...")
            if not link.reboot():
                print("  could not reconnect: check the drone, then Y to retry")
                continue
        code = fly()
    return code


# #### // RViz link: drone / obstacles / target out, target drags in // ####
def flight_viz(cfg, mocap, obstacle_set, target_param, shared, tel_fn) -> VizLink:
    def state() -> dict:
        body, age = mocap.body(cfg.mocap.drone_body)
        obstacles, _, _ = obstacle_set.snapshot(time.monotonic())
        tel, target = tel_fn(), target_param.get()
        return {"mode": shared.get("mode", "ground"), "cmd": shared.get("cmd"), "hold": shared.get("hold"),
                "drone": None if body is None else {"pos": list(body.position), "quat": list(body.quat), "age": age},
                "target": None if target is None else list(target), "msg": target_param.msg,
                "obstacles": obstacle_state(obstacles), "vbat": None if tel is None else tel.get("pm.vbat"),
                "arena": arena_state(cfg.arena_limits)}

    def on_message(msg) -> None:
        if "target" in msg and len(msg["target"]) == 3:
            obstacles, _, _ = obstacle_set.snapshot(time.monotonic())
            print(f"  RViz: {target_param.request(msg['target'], obstacles)}")

    return VizLink(cfg.viz, state, on_message)


# #### // CLI // ####
def parse_args(argv=None) -> tuple:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", help="exported .npz (default policy_flight.policy)")
    ap.add_argument("--height", type=float, help="[m] takeoff height above the floor point (default policy_flight.height)")
    ap.add_argument("--loop", action="store_true", help="after each run: Y = fly again (restarts a locked drone), B = quit")
    ap.add_argument("--viz", action="store_true", help="start RViz (crazyflie/viz/run_rviz.sh), stream to it and take its target drags")
    ap.add_argument("--dry-run", type=float, nargs="?", const=15.0, default=None, metavar="SECONDS",
                    help="stream + preflight + one policy evaluation for SECONDS (default 15), never arm")
    ap.add_argument("--param", action="append", default=[], metavar="GROUP.NAME=VALUE", help="extra firmware parameter (RAM)")
    add_override_args(ap)
    opts = ap.parse_args(argv)
    try:
        cfg = set_firmware_params(apply_overrides(load(), opts.set), opts.param)
    except ValueError as exc:
        ap.error(str(exc))
    cfg.policy_flight.policy = policy_path(cfg, opts.policy)
    if opts.height is not None:
        cfg.policy_flight.height = opts.height
    return cfg, opts


def main(argv=None) -> int:
    cfg, opts = parse_args(argv)
    pf = cfg.policy_flight
    import cflib.crtp

    try:
        policy, obs, note = load_policy(pf.policy)
        obstacle_set, warnings = load_obstacles(cfg)
    except ValueError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    for w in ([note] if note else []) + warnings:
        print(w)
    n_real = len(obstacle_set.tracked) + len(obstacle_set.static)
    if n_real < obs.n_obstacles:
        print(f"policy sees {obs.n_obstacles} obstacles, {n_real} configured: the other slot reads empty (as in its 1-obstacle training)")
    print(f"policy {Path(pf.policy).name} ({policy.obs_dim} obs); tracked obstacles "
          f"{[(o['id'], o['radius']) for o in obstacle_set.tracked] or 'none'}; static "
          + (", ".join(f"({o.pos[0]:+.2f}, {o.pos[1]:+.2f}, {o.pos[2]:+.2f}) r {o.radius}" for o in obstacle_set.static) or "none"))
    pad = None
    if opts.dry_run is None:
        try:
            pad = Gamepad(cfg.gamepad.device)
        except OSError as exc:
            print(f"ABORT: no gamepad at {cfg.gamepad.device} ({exc})", file=sys.stderr)
            return 1

    est = AlphaBeta(cfg.mocap)
    relay = MocapRelay(cfg.mocap.drone_body, est=est, obstacles=obstacle_set)
    mocap = NatNetClient(cfg.mocap.server, on_frame=relay).start()
    link = DroneLink(cfg, relay, cfg.telemetry.flight)
    target_param, shared, viz = TargetParam(), {"mode": "ground"}, None
    rviz = launch_rviz(cfg) if opts.viz else None   # started first so its window is up by takeoff
    try:
        if not wait_for_body(mocap, cfg.mocap.drone_body, pf.start_timeout):
            return 1
        time.sleep(0.5)
        found, age, missing = obstacle_set.snapshot(time.monotonic())
        if age > pf.max_obstacle_age:
            streamed = sorted(mocap.last.bodies) if mocap.last else []
            print(f"ABORT: obstacle id {missing} not tracked (bodies in the stream: {streamed})", file=sys.stderr)
            return 1
        for o in found[len(obstacle_set.static):]:
            print(f"  tracked obstacle at ({o.pos[0]:+.2f}, {o.pos[1]:+.2f}, {o.pos[2]:+.2f}) r {o.radius}")
        p_now = mocap.body(cfg.mocap.drone_body)[0].position
        target_param.reset((p_now[0], p_now[1], p_now[2] + pf.height))
        if opts.viz:
            viz = flight_viz(cfg, mocap, obstacle_set, target_param, shared, lambda: link.tel).start()
            print(f"  viz: state -> udp://127.0.0.1:{cfg.viz.state_port}, targets <- :{cfg.viz.cmd_port}")

        cflib.crtp.init_drivers()
        if not link.open():
            return 1
        return run_loop(opts, link,
                        lambda: fly_once(cfg, opts, link, mocap, est, pad, policy, obs, obstacle_set, relay, target_param, shared),
                        lambda: wait_for_restart(cfg, pad))
    except KeyboardInterrupt:
        print("\nquit", file=sys.stderr)
        return 1
    finally:
        if viz:
            viz.close()
        stop_rviz(rviz)
        link.close()
        if pad:
            pad.close()
        mocap.stop()


if __name__ == "__main__":
    sys.exit(main())
