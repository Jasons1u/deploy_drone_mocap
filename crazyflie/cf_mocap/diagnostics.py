"""Bench checks that need neither mocap nor the gamepad, and never touch the motors.

    cf-hover --scan          which Crazyflies the radio can see
    cf-hover --check-link    connect, then stream attitude + battery live: tilt the drone by hand
                             and watch roll/pitch follow. Read-only (no params written, no setpoints).
                             Exit 0 = ready to fly, 1 = no link / failed self-test, 2 = link OK but
                             battery not ready.
    cf-hover --filter-test   after the normal startup, log the onboard estimate vs mocap while you
                             carry and tilt the drone by hand; print the filter error. Never arms.
    cf-hover --calibrate-tilt  drone sitting still: measure the fixed tilt between the Motive rigid
                             body axes and the IMU (gravity as the accelerometer sees it) and write
                             mocap.body_tilt_offset_deg into the config. Never arms.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from cflib.crazyflie.log import LogConfig

from mocap_natnet.frames import (quat_conj, quat_from_rpy, quat_mul, rotate, rpy_from_quat,
                                 shortest_arc, yaw_from_quat)

from .config import Config
from .flight_log import FlightLog
from .link import Telemetry, self_test_passed

NO_BATTERY_V = 3.0          # below this the drone is running from USB power (no or dead battery)
BATTERY_JITTER_V = 0.3      # min..max spread over the check beyond this = unstable connection

FILTER_TEST_S = 30.0
# What a healthy mocap-fed onboard estimate tracks to (handheld, 50 Hz sampling). Worse = look closer.
FILTER_LIMITS = {"pos_cm": 3.0, "vel_cm_s": 20.0, "roll_deg": 3.0, "pitch_deg": 3.0,
                 "yaw_deg": 5.0}
STILL_SPEED = 0.02          # [m/s] mocap speed below which the drone counts as held still
STILL_VEL_LIMIT_CM_S = 3.0  # onboard speed allowed while still


def scan() -> int:
    cflib.crtp.init_drivers()
    found = cflib.crtp.scan_interfaces()
    if not found:
        print("No Crazyflie found. Is the Crazyradio plugged in (lsusb: 1915:7777) and the drone on?")
        return 1
    for uri, desc in found:
        print(f"{uri}  {desc}")
    return 0


def check_link(cfg: Config, duration: float = 10.0) -> int:
    logging.getLogger("cflib").setLevel(logging.ERROR)
    cflib.crtp.init_drivers()
    cache = cfg.resolve(cfg.radio.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    uri = cfg.radio.uri
    print(f"Connecting to {uri} ...")
    t0 = time.monotonic()
    try:
        scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache=str(cache)))
        scf.open_link()
    except Exception as exc:  # cflib raises bare Exception on connection failure
        print(f"FAILED to connect: {exc}\n"
              f"  - drone powered on? radio plugged in? `cf-hover --scan` lists what answers\n"
              f"  - only one program can use the radio: close cfclient")
        return 1
    try:
        cf = scf.cf
        if not self_test_passed(cf):
            print("FAILED: connected, but the Crazyflie failed its power-on self-test: its sensors, "
                  "stabilizer and motors never started (every logged value is frozen). Power-cycle "
                  "it lying flat and still; if it persists, read cfclient's Console right after "
                  "power-on to see which part failed.")
            return 1
        print(f"Connected in {time.monotonic() - t0:.1f} s  |  CRTP protocol "
              f"{cf.platform.get_protocol_version()}  |  estimator "
              f"{cf.param.get_value('stabilizer.estimator')}  controller "
              f"{cf.param.get_value('stabilizer.controller')}")
        telemetry = Telemetry(cf).start()
        print(f"Streaming for {duration:.0f} s — tilt the drone by hand, roll/pitch should follow "
              f"(Ctrl+C to stop):")
        volts = []
        end = time.monotonic() + duration
        try:
            while time.monotonic() < end:
                s = telemetry.latest
                if s.rpy_deg is not None and s.battery_v is not None:
                    volts.append(s.battery_v)
                    r, p, y = s.rpy_deg
                    print(f"\r  roll {r:+6.1f}  pitch {p:+6.1f}  yaw {y:+7.1f} deg   "
                          f"battery {s.battery_v:4.2f} V ", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            print()
            telemetry.stop()

        if not volts:
            print("FAILED: connected but no telemetry arrived — radio link is poor, move closer")
            return 1
        return _battery_verdict(min(volts), max(volts), cfg.safety.min_battery_v)
    finally:
        scf.close_link()


def _battery_verdict(lo: float, hi: float, min_v: float) -> int:
    if hi - lo > BATTERY_JITTER_V:
        print(f"Link OK, but the battery reading JUMPED between {lo:.2f} and {hi:.2f} V. Unplug "
              f"USB and re-run; if it still jumps, the battery connector is loose. Fix before flying.")
        return 2
    if hi < NO_BATTERY_V:
        print(f"Link OK. Battery reads {hi:.2f} V: running on USB power, insert a charged battery "
              f"before flying.")
        return 2
    if lo < min_v:
        print(f"Link OK. Battery {lo:.2f} V is below {min_v} V: charge it before flying.")
        return 2
    print(f"Link OK. Battery {lo:.2f} V, steady.")
    return 0


# ---------------------------------------------------------------------------
# Filter test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterError:
    pos_cm: float
    vel_cm_s: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


def filter_error(pose, tel) -> FilterError | None:
    """Onboard estimate minus mocap for one sample, or None if either side is missing."""
    if pose is None or tel.position is None or tel.velocity is None or tel.rpy_deg is None:
        return None
    mocap_rpy = [math.degrees(a) for a in rpy_from_quat(pose.quat)]
    d_ang = [(e - m + 180.0) % 360.0 - 180.0 for e, m in zip(tel.rpy_deg, mocap_rpy)]
    return FilterError(
        pos_cm=math.dist(tel.position, pose.position) * 100.0,
        vel_cm_s=math.dist(tel.velocity, pose.velocity) * 100.0,
        roll_deg=d_ang[0], pitch_deg=d_ang[1], yaw_deg=d_ang[2])


def summarize(errors: list[FilterError]) -> dict[str, tuple[float, float]]:
    """{field: (rms, max |error|)}"""
    out = {}
    for name in FILTER_LIMITS:
        vals = [abs(getattr(e, name)) for e in errors]
        out[name] = (math.sqrt(sum(v * v for v in vals) / len(vals)), max(vals))
    return out


def filter_test(cfg: Config, mocap, telemetry: Telemetry, flog: FlightLog,
                duration: float = FILTER_TEST_S) -> int:
    body = cfg.mocap.rigid_body_id
    if cfg.flight.state_source == "kalman":
        print(f"Kalman settings: ext_pos_std_dev {cfg.mocap.ext_pos_std_dev} m, ext_quat_std_dev "
              f"{cfg.mocap.ext_quat_std_dev} rad, extpose {cfg.mocap.extpose_hz:.0f} Hz")
        print("Note: with motors off the Kalman filter runs its NOT-flying model (full accelerometer "
              "+ pull toward level), which differs from the in-flight one. Trust the still-on-the-"
              "table numbers; judge the in-flight estimate from a flight log.")
    else:
        print(f"Mocap passthrough estimator, extpose {cfg.mocap.extpose_hz:.0f} Hz: position should "
              f"match mocap to ~1 mm; roll/pitch come from the IMU, yaw from mocap.")
    print(f"Filter test for {duration:.0f} s (Ctrl+C to stop early). Motors stay OFF.\n"
          f"Pick the drone up and move it: translate ~0.5 m each way, then tilt ~20-30 deg in roll "
          f"and pitch, then turn it 90 deg in yaw.")
    period = 1.0 / cfg.flight.setpoint_hz
    errors = []
    still_speeds = []   # onboard speed [cm/s] while mocap says the drone is not moving
    end = time.monotonic() + duration
    next_t = time.monotonic()
    try:
        while time.monotonic() < end:
            now = time.monotonic()
            pose, age, tel = mocap.pose(body), mocap.age(body), telemetry.latest
            flog.write(now, "handheld", None, pose, age, tel)
            err = filter_error(pose, tel) if age < cfg.safety.mocap_timeout else None
            if err is not None:
                errors.append(err)
                if math.hypot(*pose.velocity) < STILL_SPEED:
                    still_speeds.append(math.hypot(*tel.velocity) * 100.0)
                print(f"\r  est - mocap:  pos {err.pos_cm:5.1f} cm  vel {err.vel_cm_s:5.1f} cm/s  "
                      f"roll {err.roll_deg:+5.1f}  pitch {err.pitch_deg:+5.1f}  "
                      f"yaw {err.yaw_deg:+6.1f} deg ", end="", flush=True)
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"\nLog: {flog.path}")
    if not errors:
        print("FAILED: no samples with both mocap and onboard-estimate data")
        return 1

    print(f"\n{len(errors)} samples      rms      max    limit")
    bad = []
    for name, (rms, peak) in summarize(errors).items():
        limit = FILTER_LIMITS[name]
        flag = "" if peak <= limit else "   <-- high"
        if flag:
            bad.append(name)
        print(f"  {name:10s} {rms:7.2f}  {peak:7.2f}  {limit:6.1f}{flag}")
    if still_speeds:
        still_speeds.sort()
        median = still_speeds[len(still_speeds) // 2]
        flag = "" if median <= STILL_VEL_LIMIT_CM_S else "   <-- high"
        print(f"  onboard speed while held still ({len(still_speeds)} samples): median {median:.1f} "
              f"cm/s, limit {STILL_VEL_LIMIT_CM_S}{flag}")
        if flag:
            bad.append("still_velocity")
            print("  A steady velocity while still = gravity leaking into the estimate: the mocap "
                  "attitude and the accelerometer disagree. Try a larger mocap.ext_quat_std_dev.")
    if bad:
        print(f"Filter error high in {', '.join(bad)}. Yaw/roll/pitch errors that grow when you "
              f"tilt point at the rigid body axes not matching the drone (recreate it level, "
              f"nose along +X).")
        return 2
    print("Filter tracks mocap well.")
    return 0


# ---------------------------------------------------------------------------
# Tilt calibration
# ---------------------------------------------------------------------------

CALIBRATE_S = 5.0
MAX_ACC_STD_G = 0.01     # accelerometer noise allowed while "still" [G]
MAX_OFFSET_DEG = 10.0    # anything bigger is a setup problem (wrong body, up axis), not a trim


def tilt_offset(acc_body, mocap_quat, current_offset_deg) -> tuple[float, float]:
    """New [roll, pitch] body_tilt_offset_deg so mocap's "up" matches the accelerometer's.

    acc_body: mean accelerometer at rest (body frame; points up). mocap_quat: mean mocap orientation
    with current_offset_deg already applied. At rest the accelerometer measures gravity in the IMU
    frame, which is what the onboard filter uses, so that defines the frame mocap must agree with.
    """
    up_mocap = rotate(quat_conj(mocap_quat), (0.0, 0.0, 1.0))   # world up, in mocap's body frame
    residual = shortest_arc(acc_body, up_mocap)
    old = quat_from_rpy(*(math.radians(a) for a in current_offset_deg), 0.0)
    roll, pitch, _ = rpy_from_quat(quat_mul(old, residual))
    return math.degrees(roll), math.degrees(pitch)


def _write_config_value(config_path: Path, key: str, value: str) -> bool:
    """Replace `key: <value>` in the YAML text in place, keeping the line's comment."""
    text = config_path.read_text()
    new, n = re.subn(rf"(^\s*{re.escape(key)}:\s*)(\[[^\]]*\]|[^\s#]+)", lambda m: m.group(1) + value,
                     text, count=1, flags=re.M)
    if n:
        config_path.write_text(new)
    return bool(n)


def _write_offset(config_path: Path, roll: float, pitch: float) -> bool:
    return _write_config_value(config_path, "body_tilt_offset_deg", f"[{roll:.3f}, {pitch:.3f}]")


def calibrate_tilt(cfg: Config, cf, mocap, duration: float = CALIBRATE_S) -> int:
    body = cfg.mocap.rigid_body_id
    acc, quats = [], []
    lock = threading.Lock()

    def on_acc(_ts, data, _conf):
        with lock:
            acc.append((data["acc.x"], data["acc.y"], data["acc.z"]))

    log_cfg = LogConfig(name="calib_acc", period_in_ms=10)
    for v in ("acc.x", "acc.y", "acc.z"):
        log_cfg.add_variable(v, "float")
    cf.log.add_config(log_cfg)
    log_cfg.data_received_cb.add_callback(on_acc)
    print(f"Tilt calibration: keep the drone still on the floor/table for {duration:.0f} s ...")
    log_cfg.start()
    try:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if mocap.age(body) < cfg.safety.mocap_timeout:
                quats.append(mocap.pose(body).quat)
            time.sleep(0.01)
    finally:
        log_cfg.stop()
        log_cfg.delete()

    with lock:
        samples = list(acc)
    if len(samples) < 50 or len(quats) < 50:
        print(f"FAILED: too few samples (accelerometer {len(samples)}, mocap {len(quats)})")
        return 1
    mean = [sum(s[i] for s in samples) / len(samples) for i in range(3)]
    std = max(math.sqrt(sum((s[i] - mean[i]) ** 2 for s in samples) / len(samples)) for i in range(3))
    if std > MAX_ACC_STD_G:
        print(f"FAILED: the drone moved (accelerometer noise {std:.3f} G > {MAX_ACC_STD_G}); retry")
        return 1
    q = [sum(qq[i] for qq in quats) for i in range(4)]
    n = math.sqrt(sum(a * a for a in q))
    mocap_quat = tuple(a / n for a in q)

    old = cfg.mocap.body_tilt_offset_deg
    roll, pitch = tilt_offset(mean, mocap_quat, old)
    m_roll, m_pitch, _ = (math.degrees(a) for a in rpy_from_quat(mocap_quat))
    a_roll = math.degrees(math.atan2(mean[1], mean[2]))
    a_pitch = math.degrees(math.atan2(-mean[0], math.hypot(mean[1], mean[2])))
    print(f"  mocap says roll {m_roll:+.2f} pitch {m_pitch:+.2f} deg; accelerometer says roll "
          f"{a_roll:+.2f} pitch {a_pitch:+.2f} deg  (|acc| {math.dist(mean, (0, 0, 0)):.3f} G)")
    print(f"  body_tilt_offset_deg: {list(old)} -> [{roll:.3f}, {pitch:.3f}]")
    if max(abs(roll), abs(pitch)) > MAX_OFFSET_DEG:
        print(f"FAILED: offset over {MAX_OFFSET_DEG} deg is not a trim — check the rigid body ID, "
              f"up axis, and that the body was created level with the nose along +X")
        return 1
    if _write_offset(cfg.source, roll, pitch):
        print(f"  written to {cfg.source}")
    else:
        print(f"  could not find body_tilt_offset_deg in {cfg.source}; add it under mocap: or use "
              f"--set mocap.body_tilt_offset_deg=[{roll:.3f},{pitch:.3f}]")
    print("Now run `cf-hover --filter-test` with the drone still: onboard speed should be ~1 cm/s.")
    return 0


# ---------------------------------------------------------------------------
# Yaw calibration
# ---------------------------------------------------------------------------

YAW_CALIBRATE_S = 2.0
MAX_YAW_JITTER_DEG = 1.0


def yaw_calibration(mocap_quat, old_yaw_deg: float, old_tilt_deg) -> tuple[float, list[float]]:
    """New (body_yaw_offset_deg, body_tilt_offset_deg) so the drone, sitting nose along +X, reads
    yaw 0. mocap_quat has the old offsets already applied. The body rotation is
    yaw_offset * tilt_offset; turning the yaw part by `fix` re-expresses the tilt part as
    fix^-1 * tilt * fix, so the physical tilt correction is unchanged."""
    measured = math.degrees(yaw_from_quat(mocap_quat))
    fix = quat_from_rpy(0.0, 0.0, -math.radians(measured))
    tilt = quat_from_rpy(*(math.radians(a) for a in old_tilt_deg), 0.0)
    roll, pitch, _ = rpy_from_quat(quat_mul(quat_mul(quat_conj(fix), tilt), fix))
    new_yaw = (old_yaw_deg - measured + 180.0) % 360.0 - 180.0
    return new_yaw, [math.degrees(roll), math.degrees(pitch)]


def calibrate_yaw(cfg: Config, mocap, duration: float = YAW_CALIBRATE_S) -> int:
    body = cfg.mocap.rigid_body_id
    print(f"Yaw calibration: the drone must sit level with its NOSE POINTING ALONG MOCAP +X. "
          f"Measuring for {duration:.0f} s ...")
    quats = []
    end = time.monotonic() + duration
    while time.monotonic() < end:
        if mocap.age(body) < cfg.safety.mocap_timeout:
            quats.append(mocap.pose(body).quat)
        time.sleep(0.01)
    if len(quats) < 50:
        print(f"FAILED: too few mocap samples ({len(quats)})")
        return 1
    yaws = [math.degrees(yaw_from_quat(q)) for q in quats]
    mean_yaw = math.degrees(math.atan2(sum(math.sin(math.radians(y)) for y in yaws),
                                       sum(math.cos(math.radians(y)) for y in yaws)))
    jitter = max(abs((y - mean_yaw + 180) % 360 - 180) for y in yaws)
    if jitter > MAX_YAW_JITTER_DEG:
        print(f"FAILED: the drone moved (yaw varied by {jitter:.1f} deg); keep it still and retry")
        return 1
    if max(abs(a) for a in rpy_from_quat(quats[-1])[:2]) > math.radians(10):
        print("FAILED: the drone is not level")
        return 1
    q = [sum(qq[i] for qq in quats) for i in range(4)]
    n = math.sqrt(sum(a * a for a in q))
    new_yaw, new_tilt = yaw_calibration(tuple(a / n for a in q), cfg.mocap.body_yaw_offset_deg,
                                        cfg.mocap.body_tilt_offset_deg)
    print(f"  mocap yaw with the nose along +X: {mean_yaw:+.1f} deg")
    print(f"  body_yaw_offset_deg: {cfg.mocap.body_yaw_offset_deg} -> {new_yaw:.2f}")
    print(f"  body_tilt_offset_deg (same correction, new frame): {list(cfg.mocap.body_tilt_offset_deg)} "
          f"-> [{new_tilt[0]:.3f}, {new_tilt[1]:.3f}]")
    ok = (_write_config_value(cfg.source, "body_yaw_offset_deg", f"{new_yaw:.2f}")
          and _write_offset(cfg.source, *new_tilt))
    if not ok:
        print(f"  could not write both keys into {cfg.source}; set them by hand")
        return 1
    print(f"  written to {cfg.source}. Check: python3 -m mocap_natnet shows yaw ~0 now? "
          f"(it prints raw Motive yaw; cf-hover applies the offset)")
    return 0
