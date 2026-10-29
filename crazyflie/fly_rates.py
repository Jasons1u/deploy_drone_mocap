#!/usr/bin/env python3
"""Rate-mode flights on mocap_velctl (LT deadman): props-off sign bench, body-rate/thrust doublet sysID, and mj-nlp flip tapes on X."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy import radio  # noqa: E402
from cfdeploy.config import add_override_args, apply_overrides, load, set_firmware_params  # noqa: E402
from cfdeploy.flight import arm_with_deadman, preflight  # noqa: E402
from cfdeploy.gamepad import Edge, Gamepad, keyboard, stick_velocity, sticks  # noqa: E402
from cfdeploy.geometry import yaw_deg  # noqa: E402
from cfdeploy.logs import run_dir, save_flight_logs, save_link_logs, write_csv  # noqa: E402
from cfdeploy.natnet import NatNetClient  # noqa: E402
from cfdeploy.radio import DroneLink, MocapRelay, stop_and_disarm, wait_for_body  # noqa: E402
from cfdeploy.rate_loop import compensate  # noqa: E402
from cfdeploy.rate_tape import PERIOD, doublet, flip_tape, setpoint  # noqa: E402
from cfdeploy.safety import arena_limit, estop_reason, inside_soft, keep_inside, mocap_reason, outside_soft  # noqa: E402
from sysid_flight import PlanFlight  # noqa: E402

radio.LOG_BLOCKS["rate"] = [("gyro.x", "float"), ("gyro.y", "float"), ("gyro.z", "float"),
                            ("velCtl.rateP", "float"), ("velCtl.rateQ", "float"), ("velCtl.rateR", "float")]
RATES_HEADER = ["t", "seg", "row", "T", "p", "q", "r", "roll", "pitch", "yawrate", "thrust"]
COLS = "Tpqr"                 # doublet columns: collective, then body rates
RATE, ANGLE = 0, 1            # flightmode.stabModeRoll/Pitch


def set_stab_mode(cf, mode: int) -> None:
    for axis in ("Roll", "Pitch"):
        cf.param.set_value(f"flightmode.stabMode{axis}", mode)


def check_firmware(cf) -> None:
    if "rateP" not in cf.log.toc.toc.get("velCtl", {}) or "gyroGate" not in cf.param.toc.toc.get("mocapEst", {}):
        raise RuntimeError("firmware predates the rate pass-through; rebuild + flash crazyflie/firmware")


# #### // Plans: doublets per column between trimmed holds (sysID), or the sticks with X splicing in a flip // ####
def schedule(cfg, plan: str, p0) -> list:
    c = cfg.rate_flight
    if plan == "flip":   # climb to flip_height, then the sticks until A
        return [("manual", math.inf, (p0[0], p0[1], p0[2] + c.flip_height))]
    centre, body = (p0[0], p0[1], p0[2] + c.height), []
    for col in c.axes:
        for amp, dur in c.doublets[col]:
            rows = doublet(COLS.index(col), amp, dur)
            body += [("rate", len(rows) * PERIOD, (rows, False)), ("hold", c.between, centre)] * c.repeats
    return ([("hold", c.settle, centre), ("trim", c.trim_time, None), ("rate_mode", 0.3, RATE)] + body
            + [("rate_mode", 0.3, ANGLE), ("hold", 1.0, centre)])


def flip_block(cfg, tape, here) -> list:
    """X pressed at `here`: hold, trim, the tape in rate mode, recover at `here`, back to the sticks."""
    c = cfg.rate_flight
    return [("hold", c.flip_settle, here), ("trim", c.trim_time, None), ("rate_mode", 0.3, RATE), ("rate", len(tape) * PERIOD, (tape, True)),
            ("hold", c.recover, here), ("rate_mode", 0.3, ANGLE), ("manual", math.inf, here)]


def flip_problem(cfg, p, path) -> str | None:
    """Why X can't flip at p: the tape's path (offsets from its start) plus flip_margin leaves the soft arena limits."""
    m = cfg.rate_flight.flip_margin
    r = math.hypot(p[0], p[1]) + np.linalg.norm(path[:, :2], axis=1).max() + m
    for z in (p[2] + path[:, 2].min() - m, p[2] + path[:, 2].max() + m):
        problem = outside_soft((r, 0.0, z), cfg.arena_limits)
        if problem:
            return f"the flip would come {problem} (flip_margin {m} m)"
    return None


class RateFlight(PlanFlight):
    """PlanFlight + 'trim' (hover thrust from velCtl.thrustN), 'rate_mode' (flightmode params), 'rate' (rows at PERIOD), 'manual' (sticks, X, A)."""

    def __init__(self, *args, rate_log=None, tape=None, path=None, **kw):
        super().__init__(*args, **kw)
        self.rate_log, self.hover_T, self.trim, self.guard_off, self.blind_t0 = rate_log, None, [], -math.inf, None
        self.tape, self.path, self.rng = tape, path, np.random.default_rng()
        armed = self.takeoff.tilt_guard_armed
        self.takeoff.tilt_guard_armed = lambda now: armed(now) and now > self.guard_off   # off through a flip tape
        if tape is not None:   # flown anywhere on the sticks: the arena limits, not the takeoff box
            self.limit = arena_limit(self.cfg.arena_limits)
        b = self.cfg.gamepad.buttons
        self.x_btn, self.a_btn = Edge(), Edge()
        self.x_btn(self.pad.held(b.x))
        self.a_btn(self.pad.held(b.a))

    def estop(self, now, body, age) -> str | None:
        """No mocap E-STOP through a flip tape (the rate loop runs on the gyro); still lost after it: level hold blind_hold, then E-STOP."""
        reason = estop_reason(self.cfg, self.kb, self.pad, body, age, self.tel, self.limit, self.takeoff.tilt_guard_armed(now),
                              mocap_armed=False)
        if reason or now < self.guard_off:
            return reason
        lost = mocap_reason(self.cfg, body, age, self.tel, self.mocap_at)
        if lost is None or self.guard_off == -math.inf:   # no flip flown: mocap loss is an E-STOP as on every flight
            if self.blind_t0 is not None:
                print(f"  mocap back after {now - self.blind_t0:.2f} s of level hold")
            self.blind_t0 = None
            return lost
        hold = self.cfg.rate_flight.blind_hold
        if self.blind_t0 is None:
            self.blind_t0 = now
            set_stab_mode(self.cf, ANGLE)
            print(f"  after the flip, {lost}: level hold at hover thrust for up to {hold:.1f} s")
        return f"{lost}; still lost after {hold:.1f} s of level hold" if now - self.blind_t0 > hold else None

    def in_rate_segment(self) -> bool:
        return self.seg_i < len(self.schedule) and self.schedule[self.seg_i][0] == "rate" and self.seg_t0 is not None

    def start_landing(self, now, p) -> None:
        if not self.in_rate_segment():   # never hand an upside-down drone to the position loop: land after the tape
            super().start_landing(now, p)

    def dither_logs(self) -> None:
        """Restart the rate and motor log blocks at a random phase: 10 ms samples locked to 10 ms setpoints bias the inertia fit."""
        self.clock.sleep(self.rng.uniform(0.0, PERIOD))
        for lc in getattr(self.tel, "configs", []):
            if lc.name in ("rate", "mot"):
                lc.stop()
                lc.start()

    def manual(self, now, p, sp, start) -> None:
        """fly_policy's manual mode from a hold at `start`: the sticks fly it, released it holds; X splices in a flip centred here."""
        cfg, limits = self.cfg, self.cfg.arena_limits
        sp.kind, sp.row_seg = "manual", self.seg_i
        if self.seg_t0 is None:
            self.seg_t0, self.hold = now, inside_soft(start, limits)
            print("  sticks: X = flip here, A = land, release LT = motors off")
        yaw = yaw_deg(self.mocap.body(cfg.mocap.drone_body)[0].quat)
        if self.x_btn(self.pad.held(cfg.gamepad.buttons.x)):
            problem = flip_problem(cfg, p, self.path)
            print(f"  X: no flip here, {problem}" if problem else f"  X: flip at ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})")
            if problem is None:
                self.schedule[self.seg_i + 1:] = flip_block(cfg, self.tape, tuple(p))
                self.seg_i, self.seg_t0, self.hold, self.yaw0 = self.seg_i + 1, None, tuple(p), yaw
                sp.target = self.hold
                return
        s = sticks(self.pad, cfg.gamepad.axes, cfg.sticks.deadband)
        if np.abs(s).max() > 0.0:
            v, sp.yawrate = stick_velocity(s, math.radians(yaw), cfg.sticks)
            sp.mode, sp.vcmd, self.hold = "vel", tuple(keep_inside(v, p, limits)), None
            return
        if self.hold is None:
            self.hold, self.yaw0 = inside_soft(p, limits), yaw   # position setpoints face yaw0: the yaw stick re-aims it
        sp.target = self.hold

    def plan_segment(self, now, p, sp) -> None:
        if self.a_btn(self.pad.held(self.cfg.gamepad.buttons.a)):
            self.kb.land = True
        kind, dur, val = self.schedule[self.seg_i]
        nxt = self.schedule[self.seg_i + 1] if self.seg_i + 1 < len(self.schedule) else None
        if self.seg_t0 is None and nxt and nxt[0] == "rate" and not nxt[2][1]:
            self.dither_logs()
        if kind == "manual":
            return self.manual(now, p, sp, val)
        if kind not in ("trim", "rate_mode", "rate"):
            return super().plan_segment(now, p, sp)
        c = self.cfg.rate_flight
        sp.kind, sp.row_seg, first = kind, self.seg_i, self.seg_t0 is None
        if first:
            self.seg_t0 = now
        elapsed = now - self.seg_t0
        sp.target = self.hold
        if kind == "trim":
            self.trim.append(self.tel.get("velCtl.thrustN"))
            if elapsed >= dur:
                self.hover_T = float(np.median(self.trim))
                ratio = self.hover_T / (self.cfg.drone.mass * 9.81)
                print(f"  hover thrust {self.hover_T:.3f} N ({ratio:.3f} x mass g)")
                if not c.trim_range[0] < ratio < c.trim_range[1]:
                    print(f"  hover thrust outside {c.trim_range} x mass g: landing", file=sys.stderr)
                    self.kb.land = True
        elif kind == "rate_mode":
            if first:
                set_stab_mode(self.cf, val)
            if elapsed >= dur and int(self.cf.param.get_value("flightmode.stabModePitch")) != val:
                print(f"  flightmode.stabModePitch did not read back {val}: landing", file=sys.stderr)
                self.kb.land, self.seg_i = True, len(self.schedule)
                return
        else:
            rows, flip = val
            if first and flip:
                self.guard_off = now + dur + c.guard_extra
            sp.mode, sp.row = "rate", min(int(round(elapsed / PERIOD)), len(rows) - 1)   # rounded: tick jitter never skips a row
            sp.rate = rows[sp.row]
            if sp.row == len(rows) - 1:
                self.seg_i, self.seg_t0 = self.seg_i + 1, None
            return
        if elapsed >= dur:
            self.seg_i, self.seg_t0 = self.seg_i + 1, None

    def send(self, now, sp) -> None:
        if self.blind_t0 is not None:   # angle mode, level, trimmed hover thrust: needs only the IMU
            thrust = setpoint(self.hover_T, np.zeros(3))[3]
            self.cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
            self.sp_rows.append((now, "blind", sp.row_seg, "level", *[math.nan] * 5, thrust))
            return
        if sp.mode != "rate":
            return super().send(now, sp)
        T = float(sp.rate[0]) * self.hover_T
        args = setpoint(T, sp.rate[1:])
        self.cf.commander.send_setpoint(*args)
        self.sp_rows.append((now, sp.kind, sp.row_seg, sp.mode, *[math.nan] * 6))
        self.rate_log.append((now, sp.row_seg, sp.row, T, *sp.rate[1:], *args))


# #### // Props-off bench: each axis and sign in turn; the logged rate setpoint, thrustN and the motor pattern must follow // ####
BENCH_PATTERN = {1: (-1, -1, 1, 1), 2: (-1, 1, 1, -1), 3: (-1, 1, -1, 1)}   # motors that speed up for +p, +q, +r (ALLOC rows)


def bench(cf, cfg, tel, kb, pad, rate_log) -> str:
    """Props OFF. Rate setpoints at bench thrust, one (axis, sign) at a time; prints a pass/fail table and returns the outcome."""
    c, dz = cfg.rate_flight, cfg.gamepad.deadman
    set_stab_mode(cf, RATE)
    time.sleep(0.3)
    cases, samples = [(col, s) for col in (1, 2, 3) for s in (1, -1)], []
    for i, (col, s) in enumerate(cases):
        w = np.zeros(3)
        w[col - 1] = s * c.bench_rate
        for k in range(int(0.9 / PERIOD)):
            if kb.kill or not pad.trigger_held(dz.axis, dz.hold_threshold):
                return "stopped: deadman released or SPACE"
            args = setpoint(c.bench_thrust, w) if k < int(0.6 / PERIOD) else (0.0, 0.0, 0.0, 0)   # thrust 0 resets the PIDs
            cf.commander.send_setpoint(*args)
            now = time.monotonic()
            rate_log.append((now, i, k, c.bench_thrust if args[3] else 0.0, *w, *args))
            if int(0.3 / PERIOD) <= k < int(0.6 / PERIOD):
                samples.append((i, *(tel.get(f"velCtl.rate{a}") for a in "PQR"), tel.get("velCtl.thrustN"),
                                *(tel.get(f"motor.m{j}") for j in range(1, 5))))
            time.sleep(max(0.0, now + PERIOD - time.monotonic()))
    S, fails = np.array(samples, float), 0
    print("\n case    rate setpoint p q r [deg/s]   thrustN    motor pattern p q r     verdict")
    for i, (col, s) in enumerate(cases):
        m = S[S[:, 0] == i].mean(axis=0)
        want = np.zeros(3)
        want[col - 1] = s * math.degrees(c.bench_rate)
        pat = {a: float(np.dot(BENCH_PATTERN[a], m[5:9])) / 4 for a in (1, 2, 3)}
        ok = (np.abs(m[1:4] - want).max() < 1.0 and abs(m[4] - c.bench_thrust) < 0.02 * c.bench_thrust
              and np.sign(pat[col]) == s and abs(pat[col]) > 2 * max(abs(pat[a]) for a in (1, 2, 3) if a != col))
        fails += not ok
        print(f" {'+' if s > 0 else '-'}{'pqr'[col - 1]}   {m[1]:+7.1f} {m[2]:+7.1f} {m[3]:+7.1f}        {m[4]:.3f} N   "
              f"{pat[1]:+7.0f} {pat[2]:+7.0f} {pat[3]:+7.0f}   {'PASS' if ok else 'FAIL'}")
    return "bench PASS" if not fails else f"bench FAIL ({fails} of {len(cases)} cases)"


# #### // One flight: preflight, arm, plan, logs // ####
def flight(cfg, opts, link, mocap, relay, pad) -> int:
    out = run_dir(cfg, opts.plan)
    cf, tel = link.cf, link.tel
    sp_rows, rate_log, outcome, kb = [], [], "aborted before arming", None
    meta = {"plan": opts.plan, "tape": opts.tape, "config": dict(cfg), "params": link.params}
    try:
        check_firmware(cf)
        set_stab_mode(cf, ANGLE)
        tape = path = None
        if opts.plan == "flip":
            tape, tape_hover, plan = flip_tape(opts.tape)
            if abs(tape_hover / (cfg.drone.mass * 9.81) - 1.0) > 0.01:
                raise RuntimeError(f"tape hover {tape_hover:.4f} N was solved for another mass than drone.mass {cfg.drone.mass}")
            tape, path = compensate(tape, plan, tape_hover), plan["x"][:, :3] - plan["x"][0, :3]
            print(f"tape {Path(opts.tape).name}: {len(tape) * PERIOD:.2f} s, compensated for the firmware rate loop "
                  f"(J {np.array2string(plan['J'], precision=3)} kg m^2, motor lag {plan['tau_m'] * 1e3:.0f} ms)")
        if mocap:   # the bench runs without mocap
            p0, yaw0 = preflight(cfg, mocap, tel, relay.sent)
        if opts.dry_run is not None:
            print(f"Dry run: streaming {opts.dry_run:.0f} s, never arming")
            time.sleep(opts.dry_run)
            outcome = "dry run complete"
            return 0
        kb = keyboard()
        if opts.plan == "bench":
            print("BENCH: PROPS OFF. Motors spin at bench thrust.")
        cancelled = arm_with_deadman(cf, cfg, pad, kb, tel, opts.plan)
        if cancelled:
            outcome = cancelled
            return 1
        if opts.plan == "bench":
            outcome = bench(cf, cfg, tel, kb, pad, rate_log)
            print(f"\n{outcome}")
            return 0 if outcome == "bench PASS" else 2
        c = cfg.rate_flight
        segs = schedule(cfg, opts.plan, p0)
        ceiling = segs[0][2][2] + c.ceiling_margin
        outcome = RateFlight(cf, cfg, mocap, tel, kb, pad, p0, yaw0, sp_rows, segs, ceiling, c.box,
                             mocap_at=relay.position_at, rate_log=rate_log, tape=tape, path=path).run()
        print(f"\n{outcome}")
        return 0 if outcome.startswith("landed") else 2
    except (RuntimeError, ValueError) as exc:
        outcome = f"ABORT: {exc}"
        print(outcome, file=sys.stderr)
        return 1
    finally:
        stop_and_disarm(cf)
        try:
            set_stab_mode(cf, ANGLE)
        except Exception as exc:   # RAM params: a restart resets them too
            print(f"could not restore flightmode angle mode ({exc})", file=sys.stderr)
        if kb:
            kb.close()
        meta.update(outcome=outcome, extpose_sent=relay.sent, onboard_poses=tel.get("mocapEst.poses"),
                    onboard_drops=tel.get("mocapEst.drops"))
        save_flight_logs(out, relay.rows, tel.rows, sp_rows, meta)
        save_link_logs(out, link.stats, relay.sends)
        write_csv(out / "rates.csv", RATES_HEADER, rate_log)
        print(f"logs: {out}")


# #### // CLI // ####
def parse_args(argv=None) -> tuple:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", choices=("bench", "rsteps", "flip"), required=True,
                    help="bench = props OFF sign check; rsteps = rate/thrust doublet sysID; flip = --tape")
    ap.add_argument("--tape", help="mj-nlp examples/crazyflie_flip/crazyflie_flip_*.npz (flip)")
    ap.add_argument("--axes", help="rsteps: doublet columns in order, from 'pqrT' (default rate_flight.axes)")
    ap.add_argument("--dry-run", type=float, nargs="?", const=15.0, default=None, metavar="SECONDS",
                    help="stream + preflight checks for SECONDS (default 15), never arm")
    ap.add_argument("--param", action="append", default=[], metavar="GROUP.NAME=VALUE", help="extra firmware parameter (RAM)")
    add_override_args(ap)
    opts = ap.parse_args(argv)
    if opts.plan == "flip" and not opts.tape:
        ap.error("--plan flip needs --tape")
    try:
        cfg = set_firmware_params(apply_overrides(load(), opts.set), opts.param)
    except ValueError as exc:
        ap.error(str(exc))
    if opts.axes:
        cfg.rate_flight.axes = opts.axes
    if not set(cfg.rate_flight.axes) <= set(COLS):
        ap.error(f"axes {cfg.rate_flight.axes!r}: use letters from {COLS!r}")
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
    mocap = None if opts.plan == "bench" else NatNetClient(cfg.mocap.server, on_frame=relay).start()
    link = DroneLink(cfg, relay, {**cfg.telemetry.flight, "mot": 10, "rate": 10})
    try:
        if mocap and not wait_for_body(mocap, cfg.mocap.drone_body, 3.0):
            return 1
        cflib.crtp.init_drivers()
        if not link.open():
            return 1
        return flight(cfg, opts, link, mocap, relay, pad)
    finally:
        link.close()
        if pad:
            pad.close()
        if mocap:
            mocap.stop()


if __name__ == "__main__":
    sys.exit(main())
