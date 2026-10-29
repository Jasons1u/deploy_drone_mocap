#!/usr/bin/env python3
"""Hover check and velocity-step sysID flights on the mocap_velctl firmware (LT deadman); logs feed fit_vsteps.py."""

from __future__ import annotations

import argparse
import math
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import add_override_args, apply_overrides, load, set_firmware_params  # noqa: E402
from cfdeploy.flight import Landing, Takeoff, arm_with_deadman, preflight  # noqa: E402
from cfdeploy.gamepad import Gamepad, keyboard  # noqa: E402
from cfdeploy.logs import run_dir, save_flight_logs, save_link_logs  # noqa: E402
from cfdeploy.natnet import NatNetClient  # noqa: E402
from cfdeploy.radio import DroneLink, MocapRelay, stop_and_disarm, wait_for_body  # noqa: E402
from cfdeploy.safety import box_limit, estop_reason  # noqa: E402


# #### // Plans: hold at the hover height, or velocity pulses per axis // ####
def vstep_schedule(axes: str, p0, z_mid: float) -> list:
    """Hold to learn the trim, freeze the integrators, then velocity pulses back and forth per axis with holds between."""
    x0, y0, z0 = p0
    centre = (x0, y0, z0 + z_mid)
    segs = [("hold", 4.0, centre), ("freeze", 0.0, None)]
    for ax in axes:
        if ax in "xy":
            i = "xy".index(ax)
            for amp, t_on in ((0.3, 2.0), (0.6, 1.5), (1.0, 0.95)):
                start = list(centre)
                start[i] -= 0.6
                v = [0.0, 0.0, 0.0]
                v[i] = amp
                segs += [("hold", 2.0, tuple(start)), ("vel", t_on, tuple(v)), ("vel", 1.2, (0.0, 0.0, 0.0)),
                         ("hold_here", 0.8, None), ("vel", t_on, tuple(-c for c in v)), ("vel", 1.2, (0.0, 0.0, 0.0))]
        else:
            for amp, t_on in ((0.3, 1.5), (0.5, 1.0), (0.7, 0.7)):
                segs += [("hold", 2.0, (x0, y0, z0 + 0.45)), ("vel", t_on, (0.0, 0.0, amp)), ("vel", 1.2, (0.0, 0.0, 0.0)),
                         ("hold_here", 0.8, None), ("vel", t_on, (0.0, 0.0, -amp)), ("vel", 1.2, (0.0, 0.0, 0.0))]
        segs.append(("hold", 2.0, centre))
    return segs


def plan(cfg, name: str, p0) -> tuple:
    """(segments, E-STOP ceiling, E-STOP box half-width) of the hover or vsteps plan."""
    c = cfg.sysid_flight
    if name == "vsteps":
        return vstep_schedule(c.axes, p0, c.vsteps_height), p0[2] + c.z_max + c.vsteps_ceiling_margin, c.box_vsteps
    return [("hold", c.hover_time, (p0[0], p0[1], p0[2] + c.height))], p0[2] + c.height + c.hover_ceiling_margin, c.box_hover


def pulse_past_soft_limits(c, p, v, p0) -> bool:
    """Where the drone stops if the pulse ends now (stopping distance ~ v * stop_time) is past the soft limits."""
    q = [p[i] + c.stop_time * v[i] - p0[i] for i in range(3)]
    return abs(q[0]) > c.soft or abs(q[1]) > c.soft or not c.z_min < q[2] < c.z_max


# #### // Flight loop: launch -> climb -> plan segments -> land; every tick checks the E-STOP conditions // ####
def fly(cf, cfg, mocap, tel, kb, pad, p0, yaw0, sp_rows, schedule, ceiling, box, clock=time, mocap_at=None) -> str:
    return PlanFlight(cf, cfg, mocap, tel, kb, pad, p0, yaw0, sp_rows, schedule, ceiling, box, clock, mocap_at).run()


class PlanFlight:
    def __init__(self, cf, cfg, mocap, tel, kb, pad, p0, yaw0, sp_rows, schedule, ceiling, box, clock=time, mocap_at=None):
        self.cf, self.cfg, self.mocap, self.tel, self.kb, self.pad = cf, cfg, mocap, tel, kb, pad
        self.p0, self.yaw0, self.sp_rows, self.schedule, self.clock, self.mocap_at = p0, yaw0, sp_rows, schedule, clock, mocap_at
        self.takeoff = Takeoff(cfg.launch, p0, schedule[0][2][2], clock.monotonic())
        self.limit = box_limit(p0, box, ceiling)
        self.stage, self.seg_i, self.seg_t0, self.hold, self.landing = "launch", 0, None, None, None

    def run(self) -> str:
        period, next_t = self.cfg.radio.setpoint_period, self.clock.monotonic()
        while True:
            try:
                outcome = self.tick()
                if outcome:
                    return outcome
                next_t += period
                self.clock.sleep(max(0.0, next_t - self.clock.monotonic()))
            except KeyboardInterrupt:
                if self.stage != "land":
                    self.kb.land = True
                    print("  Ctrl+C: landing (Ctrl+C again = E-STOP)")
                else:
                    self.kb.kill = True

    def tick(self) -> str | None:
        """One setpoint: E-STOP checks, then each stage in order; returns the outcome when the flight ends."""
        now = self.clock.monotonic()
        body, age = self.mocap.body(self.cfg.mocap.drone_body)
        reason = self.estop(now, body, age)
        if reason:
            self.cf.commander.send_stop_setpoint()
            self.cf.supervisor.send_emergency_stop()
            return f"E-STOP: {reason}"
        p = body.position
        sp = types.SimpleNamespace(mode="pos", target=None, vcmd=(0.0, 0.0, 0.0), yawrate=0.0, kind=self.stage, row_seg=-1)
        if self.stage == "launch":
            outcome = self.launch(now, p, sp)
            if outcome:
                return outcome
        if self.stage == "takeoff":
            self.climb(now, sp)
        if self.stage == "plan" and (self.kb.land or self.seg_i >= len(self.schedule)):
            self.start_landing(now, p)
        if self.stage == "plan":
            self.plan_segment(now, p, sp)
        if self.stage == "land":
            outcome = self.descend(now, p, sp)
            if outcome:
                return outcome
        self.send(now, sp)
        return None

    def estop(self, now, body, age) -> str | None:
        return estop_reason(self.cfg, self.kb, self.pad, body, age, self.tel, self.limit, self.takeoff.tilt_guard_armed(now), self.mocap_at)

    def launch(self, now, p, sp) -> str | None:
        if self.takeoff.lifted_off(now, p):
            self.stage = "takeoff"
        elif self.takeoff.timed_out(now):
            self.cf.commander.send_stop_setpoint()
            return f"no lift-off within {self.cfg.launch.timeout:.1f} s (z {p[2] - self.p0[2]:+.3f} m): motors stopped"
        else:
            sp.mode = "thrust"
        return None

    def climb(self, now, sp) -> None:
        sp.target, done = self.takeoff.climb(now)
        if done:
            self.stage = "plan"

    def start_landing(self, now, p) -> None:
        self.stage, self.landing, self.hold = "land", Landing(self.cfg.landing, now, p[2], self.p0[2]), (p[0], p[1])
        print("  landing")

    def plan_segment(self, now, p, sp) -> None:
        """Current segment: hold a point, freeze the integrators, or a velocity pulse cut short at the soft limits."""
        kind, dur, val = self.schedule[self.seg_i]
        sp.kind, sp.row_seg = kind, self.seg_i
        if self.seg_t0 is None:
            self.seg_t0 = now
            if kind == "freeze":
                for k in ("kiX", "kiY", "kiZ"):
                    self.cf.param.set_value(f"velCtl.{k}", 0.0)
                print("  integrators frozen (ki = 0)")
            elif kind == "hold":
                self.hold = val
            elif kind == "hold_here":
                self.hold = tuple(p)
        elapsed = now - self.seg_t0
        if kind == "vel":
            sp.mode, sp.vcmd = "vel", val
            v_est = [self.tel.get(f"stateEstimate.v{a}", 0.0) for a in "xyz"]
            if any(c != 0.0 for c in val) and pulse_past_soft_limits(self.cfg.sysid_flight, p, v_est, self.p0):
                print(f"  soft limit: segment {self.seg_i} ended after {elapsed:.2f} s")
                elapsed = dur
        else:
            sp.target = self.hold
        if elapsed >= dur:
            self.seg_i, self.seg_t0 = self.seg_i + 1, None

    def descend(self, now, p, sp) -> str | None:
        sp.target = (self.hold[0], self.hold[1], self.landing.z(now))
        return self.landing.outcome(now, p)

    def send(self, now, sp) -> None:
        if sp.mode == "thrust":
            thrust = self.takeoff.thrust
            self.cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
            self.sp_rows.append((now, sp.kind, sp.row_seg, sp.mode, math.nan, math.nan, math.nan, math.nan, math.nan, thrust))
        elif sp.mode == "vel":
            self.cf.commander.send_velocity_world_setpoint(*sp.vcmd, sp.yawrate)
            self.sp_rows.append((now, sp.kind, sp.row_seg, sp.mode, math.nan, math.nan, math.nan, *sp.vcmd))
        else:
            self.cf.commander.send_position_setpoint(*sp.target, self.yaw0)
            self.sp_rows.append((now, sp.kind, sp.row_seg, sp.mode, *sp.target, math.nan, math.nan, math.nan))


# #### // One flight: preflight, arm, plan, logs // ####
def flight(cfg, opts, link, mocap, relay, pad) -> int:
    out = run_dir(cfg)
    cf, tel = link.cf, link.tel
    sp_rows, outcome, kb = [], "aborted before arming", None
    meta = {"plan": opts.plan, "config": dict(cfg), "params": link.params}
    try:
        p0, yaw0 = preflight(cfg, mocap, tel, relay.sent)
        if opts.dry_run is not None:
            print(f"Dry run: streaming {opts.dry_run:.0f} s (move the drone by hand if you like), never arming")
            time.sleep(opts.dry_run)
            outcome = "dry run complete"
            return 0
        kb = keyboard()
        cancelled = arm_with_deadman(cf, cfg, pad, kb, tel, opts.plan)
        if cancelled:
            outcome = cancelled
            return 1
        schedule, ceiling, box = plan(cfg, opts.plan, p0)
        outcome = fly(cf, cfg, mocap, tel, kb, pad, p0, yaw0, sp_rows, schedule, ceiling, box, mocap_at=relay.position_at)
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
        rows = relay.rows
        rate = len(rows) / max(1e-3, rows[-1][0] - rows[0][0]) if len(rows) > 1 else 0
        meta.update(outcome=outcome, extpose_sent=relay.sent, extpose_skipped=relay.skipped, mocap_rate_hz=round(rate, 1),
                    onboard_poses=tel.get("mocapEst.poses"), onboard_drops=tel.get("mocapEst.drops"))
        save_flight_logs(out, rows, tel.rows, sp_rows, meta)
        save_link_logs(out, link.stats, relay.sends)
        print(f"extpose sent {relay.sent}, onboard poses {tel.get('mocapEst.poses')}, drops {tel.get('mocapEst.drops')}; logs: {out}")


# #### // CLI // ####
def parse_args(argv=None) -> tuple:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", choices=("hover", "vsteps"), default="hover", help="vsteps = velocity-step sysID (+-0.9 m xy, 0.3-1.2 m z)")
    ap.add_argument("--axes", help="vsteps: axes to excite, in order (default sysid_flight.axes)")
    ap.add_argument("--height", type=float, help="[m] hover height above takeoff (default sysid_flight.height)")
    ap.add_argument("--dry-run", type=float, nargs="?", const=15.0, default=None, metavar="SECONDS",
                    help="stream + preflight checks for SECONDS (default 15), never arm")
    ap.add_argument("--param", action="append", default=[], metavar="GROUP.NAME=VALUE", help="extra firmware parameter (RAM)")
    add_override_args(ap)
    opts = ap.parse_args(argv)
    try:
        cfg = set_firmware_params(apply_overrides(load(), opts.set), opts.param)
    except ValueError as exc:
        ap.error(str(exc))
    for key in ("axes", "height"):
        if getattr(opts, key) is not None:
            cfg.sysid_flight[key] = getattr(opts, key)
    return cfg, opts


def main(argv=None) -> int:
    cfg, opts = parse_args(argv)
    import cflib.crtp

    pad = None
    if opts.dry_run is None:
        try:
            pad = Gamepad(cfg.gamepad.device)
        except OSError as exc:
            print(f"ABORT: no gamepad for the deadman at {cfg.gamepad.device} ({exc})", file=sys.stderr)
            return 1
    relay = MocapRelay(cfg.mocap.drone_body)
    mocap = NatNetClient(cfg.mocap.server, on_frame=relay).start()
    print(f"Motive {'.'.join(map(str, mocap.app_version))}, NatNet {'.'.join(map(str, mocap.natnet_version[:2]))}, "
          f"local {mocap.local_ip}")
    link = DroneLink(cfg, relay, cfg.telemetry.sysid)
    try:
        if not wait_for_body(mocap, cfg.mocap.drone_body, 3.0):
            return 1
        cflib.crtp.init_drivers()
        if not link.open():
            return 1
        return flight(cfg, opts, link, mocap, relay, pad)
    finally:
        link.close()
        if pad:
            pad.close()
        mocap.stop()


if __name__ == "__main__":
    sys.exit(main())
