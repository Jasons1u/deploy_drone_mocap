#!/usr/bin/env python3
"""Hover check and velocity-step sysID flights on the mocap_velctl firmware (LT deadman); logs feed fit_vsteps.py."""

from __future__ import annotations

import argparse
import math
import sys
import time
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
    x0, y0, z0 = p0
    takeoff = Takeoff(cfg.launch, p0, schedule[0][2][2], clock.monotonic())
    limit = box_limit(p0, box, ceiling)
    stage, seg_i, seg_t0, hold, landing = "launch", 0, None, None, None
    period, next_t = cfg.radio.setpoint_period, clock.monotonic()
    while True:
        try:
            now = clock.monotonic()
            body, age = mocap.body(cfg.mocap.drone_body)
            reason = estop_reason(cfg, kb, pad, body, age, tel, limit, takeoff.tilt_guard_armed(now), mocap_at)
            if reason:
                cf.commander.send_stop_setpoint()
                cf.supervisor.send_emergency_stop()
                return f"E-STOP: {reason}"
            p = body.position
            mode, target, vcmd, kind, row_seg = "pos", None, (0.0, 0.0, 0.0), stage, -1

            if stage == "launch":
                if takeoff.lifted_off(now, p):
                    stage = "takeoff"
                elif takeoff.timed_out(now):
                    cf.commander.send_stop_setpoint()
                    return f"no lift-off within {cfg.launch.timeout:.1f} s (z {p[2] - z0:+.3f} m): motors stopped"
                else:
                    mode = "thrust"
            if stage == "takeoff":
                target, done = takeoff.climb(now)
                if done:
                    stage = "plan"
            if stage == "plan" and (kb.land or seg_i >= len(schedule)):
                stage, landing, hold = "land", Landing(cfg.landing, now, p[2], z0), (p[0], p[1])
                print("  landing")
            if stage == "plan":
                kind, dur, val = schedule[seg_i]
                row_seg = seg_i
                if seg_t0 is None:
                    seg_t0 = now
                    if kind == "freeze":
                        for k in ("kiX", "kiY", "kiZ"):
                            cf.param.set_value(f"velCtl.{k}", 0.0)
                        print("  integrators frozen (ki = 0)")
                    elif kind == "hold":
                        hold = val
                    elif kind == "hold_here":
                        hold = tuple(p)
                elapsed = now - seg_t0
                if kind == "vel":
                    mode, vcmd = "vel", val
                    v_est = [tel.get(f"stateEstimate.v{a}", 0.0) for a in "xyz"]
                    if any(c != 0.0 for c in val) and pulse_past_soft_limits(cfg.sysid_flight, p, v_est, p0):
                        print(f"  soft limit: segment {seg_i} ended after {elapsed:.2f} s")
                        elapsed = dur
                else:
                    target = hold
                if elapsed >= dur:
                    seg_i, seg_t0 = seg_i + 1, None
            if stage == "land":
                target = (hold[0], hold[1], landing.z(now))
                done = landing.outcome(now, p)
                if done:
                    return done

            if mode == "thrust":
                thrust = takeoff.thrust
                cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
                sp_rows.append((now, kind, row_seg, mode, math.nan, math.nan, math.nan, math.nan, math.nan, thrust))
            elif mode == "vel":
                cf.commander.send_velocity_world_setpoint(*vcmd, 0.0)
                sp_rows.append((now, kind, row_seg, mode, math.nan, math.nan, math.nan, *vcmd))
            else:
                cf.commander.send_position_setpoint(*target, yaw0)
                sp_rows.append((now, kind, row_seg, mode, *target, math.nan, math.nan, math.nan))
            next_t += period
            clock.sleep(max(0.0, next_t - clock.monotonic()))
        except KeyboardInterrupt:
            if stage != "land":
                kb.land = True
                print("  Ctrl+C: landing (Ctrl+C again = E-STOP)")
            else:
                kb.kill = True


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
