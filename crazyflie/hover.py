#!/usr/bin/env python3
"""Hover on the mocap_velctl firmware: every mocap frame -> extpose, onboard velocity controller flies position setpoints.

    python3 crazyflie/hover.py --dry-run          # full pipeline + onboard-vs-mocap checks, never arms
    python3 crazyflie/hover.py --mass 0.044       # fly: hold LT (deadman) to take off, release = E-STOP
    python3 crazyflie/hover.py --plan vsteps --axes x   # velocity-step sysID (needs +-0.9 m xy, 0.3-1.2 m z)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import select
import struct
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap"))
from natnet import NatNetClient  # noqa: E402

CONTROLLER_VELCTL = 6
SUP_CAN_BE_ARMED, SUP_ARMED = 1 << 0, 1 << 1
LOG_BLOCKS = {
    "pos": (10, [("stateEstimate.x", "float"), ("stateEstimate.y", "float"), ("stateEstimate.z", "float"),
                 ("velCtl.thrustN", "float"), ("mocapEst.ageMs", "uint16_t"), ("supervisor.info", "uint16_t")]),
    "vel": (10, [("stateEstimate.vx", "float"), ("stateEstimate.vy", "float"), ("stateEstimate.vz", "float"),
                 ("stabilizer.roll", "float"), ("stabilizer.pitch", "float")]),
    "ref": (20, [("velCtl.vRefX", "float"), ("velCtl.vRefY", "float"), ("velCtl.vRefZ", "float"),
                 ("velCtl.rollDes", "float"), ("velCtl.pitchDes", "float"), ("velCtl.yawCmd", "float")]),
    "acc": (20, [("velCtl.aCmdX", "float"), ("velCtl.aCmdY", "float"), ("velCtl.aCmdZ", "float"),
                 ("stabilizer.yaw", "float"), ("pm.vbat", "float")]),
    "mot": (10, [("motor.m1", "uint16_t"), ("motor.m2", "uint16_t"), ("motor.m3", "uint16_t"), ("motor.m4", "uint16_t")]),
    "est": (100, [("mocapEst.poses", "uint32_t"), ("mocapEst.drops", "uint32_t"), ("mocapEst.yawOffset", "float")]),
}


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def yaw_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tilt_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


class Telemetry:
    """Latest value + full history of every log block."""

    def __init__(self, cf):
        self.latest, self.rows, self.stamp = {}, {k: [] for k in LOG_BLOCKS}, {}
        self.lock = threading.Lock()
        self.configs = []
        for name, (period, variables) in LOG_BLOCKS.items():
            lc = LogConfig(name=name, period_in_ms=period)
            for v, t in variables:
                lc.add_variable(v, t)
            lc.data_received_cb.add_callback(lambda ts, d, c, name=name: self._on(name, d))
            cf.log.add_config(lc)
            self.configs.append(lc)

    def _on(self, name, d):
        now = time.monotonic()
        with self.lock:
            self.latest.update(d)
            self.stamp[name] = now
            self.rows[name].append((now, *d.values()))

    def start(self):
        for lc in self.configs:
            lc.start()

    def stop(self):
        for lc in self.configs:
            lc.stop()

    def get(self, key, default=None):
        with self.lock:
            return self.latest.get(key, default)

    def age(self, block="pos") -> float:
        with self.lock:
            return time.monotonic() - self.stamp.get(block, -1e9)


class Gamepad:
    """Linux joystick API reader; a read error (controller gone) counts as every input released."""

    EVENT = struct.Struct("<IhBB")

    def __init__(self, device: str):
        self.fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        self.buttons, self.axes, self.alive = {}, {}, True
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set() and self.alive:
            try:
                if not select.select([self.fd], [], [], 0.05)[0]:
                    continue
                data = os.read(self.fd, self.EVENT.size * 64)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if not data:
                self.alive = False
                break
            for i in range(0, len(data) - self.EVENT.size + 1, self.EVENT.size):
                _, value, typ, num = self.EVENT.unpack_from(data, i)
                if typ & 0x01:
                    self.buttons[num] = bool(value)
                elif typ & 0x02:
                    self.axes[num] = value / 32767.0

    def held(self, button: int) -> bool:
        return self.alive and self.buttons.get(button, False)

    def axis(self, num: int) -> float | None:
        """[-1, 1], None if never reported or the controller is gone."""
        return self.axes.get(num) if self.alive else None

    def trigger_held(self, num: int, threshold: float) -> bool:
        a = self.axis(num)
        return a is not None and a > threshold

    def close(self):
        self._stop.set()
        os.close(self.fd)


class NoKeyboard:
    kill = land = False

    def close(self):
        pass


class Keyboard:
    """SPACE -> kill, L -> land; needs a TTY on stdin."""

    def __init__(self):
        self.kill = self.land = False
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                c = os.read(self.fd, 1)
                if c == b" ":
                    self.kill = True
                elif c in (b"l", b"L"):
                    self.land = True

    def close(self):
        self._stop.set()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def save_logs(run_dir: Path, mocap_rows, tel: Telemetry, sp_rows, meta: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "mocap.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "frame", "x", "y", "z", "qw", "qx", "qy", "qz"])
        w.writerows(mocap_rows)
    for name, (_, variables) in LOG_BLOCKS.items():
        with open(run_dir / f"cf_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [v for v, _ in variables])
            w.writerows(tel.rows[name])
    with open(run_dir / "setpoints.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "kind", "seg", "mode", "x", "y", "z", "vx", "vy", "vz"])
        w.writerows(sp_rows)
    (run_dir / "meta.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7")
    ap.add_argument("--server", default="192.168.1.2", help="Motive PC")
    ap.add_argument("--body", type=int, default=23, help="Motive rigid body streaming ID")
    ap.add_argument("--mass", type=float, default=0.044, help="[kg] drone incl. markers + battery")
    ap.add_argument("--height", type=float, default=0.4, help="[m] above takeoff")
    ap.add_argument("--hover-time", type=float, default=8.0)
    ap.add_argument("--ramp-time", type=float, default=2.5, help="[s] climb ramp after launch")
    ap.add_argument("--launch-thrust", type=float, default=1.0, help="fraction of full thrust, level attitude, until launch height")
    ap.add_argument("--launch-height", type=float, default=0.05, help="[m] above takeoff, then the controller takes over")
    ap.add_argument("--launch-timeout", type=float, default=1.0, help="[s] motors stop if not at launch height by then")
    ap.add_argument("--ki-xy", type=float, default=1.0, help="velCtl.kiX/kiY for this flight")
    ap.add_argument("--ki-z", type=float, default=2.0, help="velCtl.kiZ for this flight")
    ap.add_argument("--yaw-trim", type=float, default=0.0, help="velCtl.yawTrim: this drone's yaw imbalance (unproven, keep 0)")
    ap.add_argument("--min-vbat", type=float, default=3.7)
    ap.add_argument("--max-tilt", type=float, default=60.0, help="[deg] airborne flip guard (off during launch); 180 = off")
    ap.add_argument("--plan", choices=("hover", "vsteps"), default="hover", help="vsteps = velocity-step sysID")
    ap.add_argument("--axes", default="xyz", help="vsteps: axes to excite, in order")
    ap.add_argument("--box", type=float, default=None, help="[m] E-STOP |x|,|y| from takeoff (hover 0.5, vsteps 1.0)")
    ap.add_argument("--soft", type=float, default=0.85, help="[m] vsteps: a pulse ends early past this |x|,|y|")
    ap.add_argument("--z-min", type=float, default=0.3, help="[m] vsteps: pulse soft limits above takeoff")
    ap.add_argument("--z-max", type=float, default=1.2)
    ap.add_argument("--land-speed", type=float, default=0.2, help="[m/s]")
    ap.add_argument("--dry-run", type=float, nargs="?", const=15.0, default=None, metavar="SECONDS",
                    help="stream + check for SECONDS (default 15), never arm")
    ap.add_argument("--param", action="append", default=[], metavar="GROUP.NAME=VALUE",
                    help="set a firmware parameter for this flight (RAM only), e.g. pid_attitude.roll_kp=9")
    ap.add_argument("--js", default="/dev/input/js0", help="gamepad device for the deadman")
    ap.add_argument("--deadman-axis", type=int, default=2, help="LT on the 8BitDo Ultimate 2C (-1 released, +1 pulled)")
    ap.add_argument("--deadman-threshold", type=float, default=0.5, help="LT must pass this to arm")
    ap.add_argument("--deadman-hold", type=float, default=0.0, help="in flight, LT below this = E-STOP (hysteresis)")
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    args = ap.parse_args(argv)
    if args.box is None:
        args.box = 1.0 if args.plan == "vsteps" else 0.5

    pad = None
    if args.dry_run is None:
        try:
            pad = Gamepad(args.js)
        except OSError as exc:
            print(f"ABORT: no gamepad for the deadman at {args.js} ({exc})", file=sys.stderr)
            return 1

    cf_holder, sent, mocap_rows = {"cf": None}, [0], []

    def on_frame(frame):
        b = frame.bodies.get(args.body)
        if b is None or not b.valid:
            return
        mocap_rows.append((frame.arrival, frame.number, *b.position, *b.quat))
        cf = cf_holder["cf"]
        if cf is not None:
            w, qx, qy, qz = b.quat
            cf.extpos.send_extpose(x=b.position[0], y=b.position[1], z=b.position[2], qx=qx, qy=qy, qz=qz, qw=w)
            sent[0] += 1

    mocap = NatNetClient(args.server, on_frame=on_frame).start()
    print(f"Motive {'.'.join(map(str, mocap.app_version))}, NatNet {'.'.join(map(str, mocap.natnet_version[:2]))}, "
          f"local {mocap.local_ip}")
    t0 = time.monotonic()
    while mocap.body(args.body)[0] is None:
        if time.monotonic() - t0 > 3.0:
            print(f"ABORT: rigid body {args.body} not tracked ({mocap.frames} frames received)", file=sys.stderr)
            mocap.stop()
            return 1
        time.sleep(0.02)

    cflib.crtp.init_drivers()
    run_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S")
    sp_rows, outcome, kb = [], "aborted before arming", None
    meta = {"args": vars(args)}
    scf = SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=str(Path(args.log_dir) / "cache")))
    link_error = connect(scf, timeout=20.0)
    if link_error:
        print(f"ABORT: no radio link to {args.uri} ({link_error}); is the drone on?", file=sys.stderr)
        mocap.stop()
        return 1
    try:
        cf = scf.cf
        tel = None
        try:
            meta["params"] = configure(cf, args)
            tel = Telemetry(cf)
            tel.start()
            cf_holder["cf"] = cf
            time.sleep(1.5)

            p0, yaw0 = preflight(args, mocap, tel, sent)
            if args.dry_run is not None:
                print(f"Dry run: streaming {args.dry_run:.0f} s (move the drone by hand if you like), never arming")
                time.sleep(args.dry_run)
                outcome = "dry run complete"
                return 0

            kb = Keyboard() if sys.stdin.isatty() else NoKeyboard()
            cancelled = arm_with_deadman(cf, args, pad, kb, tel, args.plan)
            if cancelled:
                outcome = cancelled
                return 1
            if args.plan == "vsteps":
                schedule, ceiling = vstep_schedule(args.axes, p0, 0.6), p0[2] + args.z_max + 0.1
            else:
                schedule, ceiling = [("hold", args.hover_time, (p0[0], p0[1], p0[2] + args.height))], p0[2] + args.height + 0.4
            outcome = fly(cf, args, mocap, tel, kb, pad, p0, yaw0, sp_rows, schedule, ceiling)
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
            cf_holder["cf"] = None
            if kb:
                kb.close()
            if pad:
                pad.close()
            if tel:
                tel.stop()
                rate = len(mocap_rows) / max(1e-3, mocap_rows[-1][0] - mocap_rows[0][0]) if len(mocap_rows) > 1 else 0
                meta.update(outcome=outcome, extpose_sent=sent[0], mocap_rate_hz=round(rate, 1),
                            onboard_poses=tel.get("mocapEst.poses"), onboard_drops=tel.get("mocapEst.drops"))
                save_logs(run_dir, mocap_rows, tel, sp_rows, meta)
                print(f"extpose sent {sent[0]}, onboard poses {tel.get('mocapEst.poses')}, "
                      f"drops {tel.get('mocapEst.drops')}; logs: {run_dir}")
            mocap.stop()
    finally:
        scf.close_link()


def configure(cf, args) -> dict:
    """Check the firmware, select the velocity controller, set mass / integrators / --param overrides; returns all params."""
    toc = cf.param.toc.toc
    if "velCtl" not in toc or "alpha" not in toc.get("mocapEst", {}):
        raise RuntimeError("drone is not running the mocap_velctl firmware (crazyflie/firmware/flash.py)")
    if int(cf.param.get_value("system.selftestPassed")) != 1:
        raise RuntimeError("drone failed its power-on self-test; power-cycle it lying flat")
    cf.param.set_value("stabilizer.controller", CONTROLLER_VELCTL)
    cf.param.set_value("velCtl.mass", args.mass)
    if "yawTrim" not in toc["velCtl"]:
        raise RuntimeError("firmware predates velCtl.yawTrim; rebuild + flash crazyflie/firmware")
    for k, v in (("kiX", args.ki_xy), ("kiY", args.ki_xy), ("kiZ", args.ki_z), ("yawTrim", args.yaw_trim)):
        cf.param.set_value(f"velCtl.{k}", v)
    for kv in args.param:
        name, value = kv.split("=", 1)
        group, key = name.split(".", 1)
        if key not in toc.get(group, {}):
            raise RuntimeError(f"unknown parameter {name}")
        cf.param.set_value(name, float(value))
    time.sleep(0.3)
    params = {f"{g}.{k}": cf.param.get_value(f"{g}.{k}")
              for g in ("velCtl", "mocapEst", "pid_attitude", "pid_rate") for k in toc[g]}
    for kv in args.param:
        name, value = kv.split("=", 1)
        if abs(float(params[name]) - float(value)) > 1e-4:
            raise RuntimeError(f"{name} reads back {params[name]}, not {value}")
    if args.param:
        print("params: " + ", ".join(f"{kv.split('=')[0]}={params[kv.split('=')[0]]}" for kv in args.param))
    if int(cf.param.get_value("stabilizer.controller")) != CONTROLLER_VELCTL:
        raise RuntimeError("could not select the velocity controller")
    return params


def preflight(args, mocap, tel, sent) -> tuple:
    """Onboard estimate vs mocap, battery, level, supervisor; returns the takeoff pose (p0, yaw0)."""
    body, age = mocap.body(args.body)
    p0 = body.position
    yaw0 = yaw_deg(body.quat)
    est = [tel.get(f"stateEstimate.{a}") for a in "xyz"]
    vel = [tel.get(f"stateEstimate.v{a}") for a in "xyz"]
    onboard_age = tel.get("mocapEst.ageMs")
    vbat = tel.get("pm.vbat")
    sup = int(tel.get("supervisor.info", 0))
    err = math.dist(p0, est)
    yaw_err = wrap180(tel.get("stabilizer.yaw") - yaw0)
    print(f"mocap   ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f}) yaw {yaw0:+.1f}  tilt {tilt_deg(body.quat):.1f} deg")
    print(f"onboard ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) yaw err {yaw_err:+.1f} deg, "
          f"|v| {math.hypot(*vel):.3f} m/s, mocap age {onboard_age} ms, IMU roll/pitch "
          f"{tel.get('stabilizer.roll'):+.1f}/{-tel.get('stabilizer.pitch'):+.1f} deg")
    print(f"battery {vbat:.2f} V, supervisor.info 0x{sup:04x}, extpose sent {sent[0]}")
    problems = []
    if age > 0.1:
        problems.append(f"mocap stale ({age:.2f} s)")
    if onboard_age is None or onboard_age > 100:
        problems.append(f"drone not receiving mocap (ageMs {onboard_age})")
    if err > 0.02:
        problems.append(f"onboard position {err * 100:.1f} cm from mocap")
    if abs(yaw_err) > 5:
        problems.append(f"onboard yaw {yaw_err:+.1f} deg from mocap")
    if math.hypot(*vel) > 0.05:
        problems.append(f"onboard velocity {math.hypot(*vel):.3f} m/s at rest")
    if tilt_deg(body.quat) > 10:
        problems.append("not level")
    if vbat < args.min_vbat:
        problems.append(f"battery {vbat:.2f} V < {args.min_vbat}")
    if not sup & SUP_CAN_BE_ARMED:
        problems.append(f"supervisor cannot arm (0x{sup:04x})")
    if problems:
        raise RuntimeError("preflight: " + "; ".join(problems))
    print("Preflight OK")
    return p0, yaw0


def arm_with_deadman(cf, args, pad, kb, tel, label: str) -> str | None:
    """Wait for LT released then held 1.5 s, arm, confirm; returns None when armed, else why it was cancelled."""
    print(f"\nRelease LT, then hold it to take off ({label}) (120 s). "
          f"Releasing LT at any time = motors off.")
    seen_release, held_since, deadline = False, None, time.monotonic() + 120.0
    next_report = time.monotonic() + 5.0
    while held_since is None or time.monotonic() - held_since < 1.5:
        if kb.kill or kb.land or time.monotonic() > deadline:
            a = pad.axis(args.deadman_axis)
            outcome = (f"cancelled before arming (LT {a if a is None else round(a, 2)}, "
                       f"release seen {seen_release}, gamepad alive {pad.alive})")
            print(outcome, file=sys.stderr)
            return outcome
        if time.monotonic() > next_report:
            a = pad.axis(args.deadman_axis)
            hint = ("LT is already pulled: release it, then pull again" if not seen_release
                    else f"need > {args.deadman_threshold} for 1.5 s")
            print(f"  waiting for LT: reads {a if a is None else round(a, 2)} ({hint})")
            next_report += 5.0
        a = pad.axis(args.deadman_axis)
        seen_release |= a is not None and a < -args.deadman_threshold
        held = seen_release and pad.trigger_held(args.deadman_axis, args.deadman_threshold)
        held_since = (held_since or time.monotonic()) if held else None
        time.sleep(0.02)
    print("  LT held: arming")
    cf.supervisor.send_arming_request(True)
    time.sleep(0.2)
    cf.commander.send_setpoint(0, 0, 0, 0)
    deadline = time.monotonic() + 1.0
    while not int(tel.get("supervisor.info", 0)) & SUP_ARMED:
        if time.monotonic() > deadline:
            raise RuntimeError(f"drone did not arm (supervisor.info 0x{int(tel.get('supervisor.info', 0)):04x})")
        time.sleep(0.02)
    return None


def connect(scf, timeout: float) -> str | None:
    """open_link + parameter download with a watchdog (a half-open link otherwise blocks forever)."""
    result = {}

    def run():
        try:
            scf.open_link()
            scf.wait_for_params()
            result["ok"] = True
        except Exception as exc:
            result["error"] = str(exc)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if result.get("ok"):
        return None
    if th.is_alive():
        scf.cf.close_link()
        return f"connect timed out after {timeout:.0f} s"
    return result.get("error", "unknown")


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


def outside_soft(args, p, v, p0) -> bool:
    """Where the drone stops if the pulse ends now (stopping distance ~ v * 0.35 s) is past the soft limits."""
    q = [p[i] + 0.35 * v[i] - p0[i] for i in range(3)]
    return abs(q[0]) > args.soft or abs(q[1]) > args.soft or not args.z_min < q[2] < args.z_max


def fly(cf, args, mocap, tel, kb, pad, p0, yaw0, sp_rows, schedule, ceiling) -> str:
    """100 Hz setpoints: takeoff ramp, schedule segments, landing; every tick checks the E-STOP conditions."""
    x0, y0, z0 = p0
    top = schedule[0][2][2]
    stage, seg_i, seg_t0, hold, t_land, z_land0, ground_since = "launch", 0, None, None, None, None, None
    t_start = time.monotonic()
    t_climb, z_climb = None, None
    period, next_t = 0.01, time.monotonic()
    while True:
        try:
            now = time.monotonic()
            body, age = mocap.body(args.body)
            tilt_armed = stage != "launch" and t_climb is not None and now - t_climb > 0.5
            reason = estop_reason(args, kb, pad, body, age, tel, x0, y0, z0, ceiling, tilt_armed)
            if reason:
                cf.commander.send_stop_setpoint()
                cf.supervisor.send_emergency_stop()
                return f"E-STOP: {reason}"
            p = body.position
            mode, target, vcmd, kind, row_seg = "pos", None, (0.0, 0.0, 0.0), stage, -1

            if stage == "launch":
                if p[2] > z0 + args.launch_height and tilt_deg(body.quat) < 20.0:
                    stage, t_climb, z_climb = "takeoff", now, p[2]
                elif now - t_start > args.launch_timeout:
                    cf.commander.send_stop_setpoint()
                    return f"no lift-off within {args.launch_timeout:.1f} s (z {p[2] - z0:+.3f} m): motors stopped"
                else:
                    mode = "thrust"
            if stage == "takeoff":
                frac = min(1.0, (now - t_climb) / args.ramp_time)
                target = (x0, y0, z_climb + (top - z_climb) * frac)
                if frac >= 1.0:
                    stage = "plan"
            if stage == "plan" and (kb.land or seg_i >= len(schedule)):
                stage, t_land, z_land0, hold = "land", now, p[2], (p[0], p[1])
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
                    if any(c != 0.0 for c in val) and outside_soft(args, p, v_est, p0):
                        print(f"  soft limit: segment {seg_i} ended after {elapsed:.2f} s")
                        elapsed = dur
                else:
                    target = hold
                if elapsed >= dur:
                    seg_i, seg_t0 = seg_i + 1, None
            if stage == "land":
                z = max(z0 - 0.1, z_land0 - args.land_speed * (now - t_land))
                target = (hold[0], hold[1], z)
                ground_since = (ground_since or now) if p[2] < z0 + 0.05 else None
                if ground_since and now - ground_since > 0.5:
                    return "landed"
                if now - t_land > (z_land0 - z0) / args.land_speed + 3.0:
                    return "landed (timeout)"

            if mode == "thrust":
                thrust = int(args.launch_thrust * 65535)
                cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
                sp_rows.append((now, kind, row_seg, mode, math.nan, math.nan, math.nan, math.nan, math.nan, thrust))
            elif mode == "vel":
                cf.commander.send_velocity_world_setpoint(*vcmd, 0.0)
                sp_rows.append((now, kind, row_seg, mode, math.nan, math.nan, math.nan, *vcmd))
            else:
                cf.commander.send_position_setpoint(*target, yaw0)
                sp_rows.append((now, kind, row_seg, mode, *target, math.nan, math.nan, math.nan))
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
        except KeyboardInterrupt:
            if stage != "land":
                kb.land = True
                print("  Ctrl+C: landing (Ctrl+C again = E-STOP)")
            else:
                kb.kill = True


def estop_reason(args, kb, pad, body, age, tel, x0, y0, z0, ceiling, tilt_armed=True) -> str | None:
    if not pad.trigger_held(args.deadman_axis, args.deadman_hold):
        a = pad.axis(args.deadman_axis)
        return f"deadman (LT) released (reads {a if a is None else round(a, 2)})" if pad.alive else "gamepad disconnected"
    if kb.kill:
        return "SPACE"
    if age > 0.25:
        return f"mocap lost {age:.2f} s"
    onboard_age = tel.get("mocapEst.ageMs", 65535)
    if onboard_age > 250:
        return f"drone has no mocap for {onboard_age} ms"
    if tel.age("pos") > 0.5:
        return f"no telemetry for {tel.age('pos'):.1f} s"
    p = body.position
    if abs(p[0] - x0) > args.box or abs(p[1] - y0) > args.box or p[2] > ceiling:
        return f"left the box at ({p[0] - x0:+.2f}, {p[1] - y0:+.2f}, {p[2] - z0:+.2f})"
    if tilt_armed and tilt_deg(body.quat) > args.max_tilt:
        return f"tilt {tilt_deg(body.quat):.0f} deg"
    est = [tel.get(f"stateEstimate.{a}") for a in "xyz"]
    if math.dist(p, est) > 0.10:
        return f"onboard estimate {math.dist(p, est) * 100:.0f} cm from mocap"
    return None


if __name__ == "__main__":
    sys.exit(main())
