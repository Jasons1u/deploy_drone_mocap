"""Crazyflie hover with OptiTrack.

    cf-hover --scan                          # find the radio + drone URI
    cf-hover --check-link                    # connect + live attitude/battery (no mocap, no motors)
    cf-hover --gamepad-test                  # find the deadman axis / land button
    cf-hover --preflight-only                # connect, run every check, exit (no motors)
    cf-hover --calibrate-yaw                 # nose along mocap +X: measure rigid body yaw offset
    cf-hover --calibrate-tilt                # drone still: measure rigid body vs IMU tilt, save it
    cf-hover                                 # fly, mocap as the state (default)
    cf-hover --kalman                        # fly, onboard Kalman (fed by mocap) as the state
    cf-hover --kalman --filter-test          # Kalman vs mocap while you carry the drone (never arms)
    cf-hover --set flight.hover_height=0.6   # override any config key

State source:
  default   OptiTrack position/velocity/heading drive a host-side position controller
            (mocap_control.py) that sends roll/pitch/yaw-rate/thrust. The drone only stabilises
            attitude from its IMU (complementary estimator); no extpose, no Kalman.
  --kalman  mocap -> extpose -> onboard Kalman filter -> onboard position controller, which the
            host feeds position setpoints.

Sequence:
  1. Connect to mocap and the drone; with --kalman, reset the Kalman filter on mocap.
  2. Preflight: mocap fresh, inside the geofence, level, battery OK, and the drone agrees with mocap
     (Kalman position with --kalman; IMU roll/pitch otherwise).
  3. Hold the deadman: re-check, capture the takeoff pose, arm, take off / hover / land.

Ending a flight:
  land button / hover_duration elapsed / Ctrl+C  -> controlled landing
  deadman released / mocap lost / geofence / tilt beyond max_flight_tilt_deg / Kalman divergence
  (--kalman) / Ctrl+C while landing -> E-STOP (motors cut: at < 1 m a drop is the safer failure)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from mocap_natnet import NatNetClient
from mocap_natnet.frames import yaw_from_quat

from . import diagnostics, safety
from .config import DEFAULT_CONFIG, Config, load_config
from .flight_log import FlightLog
from .flight_plan import HoverPlan, Phase
from .gamepad import Gamepad, print_inputs
from .link import (ExtPoseFeeder, Telemetry, arm, configure, has_mocap_firmware, reset_estimator,
                   self_test_passed, send_attitude, stop_motors)
from .mocap_control import MocapPositionController

log = logging.getLogger("cf_hover")

TELEMETRY_SETTLE_S = 0.5   # after connecting / the Kalman reset, let the log blocks deliver


class Abort(Exception):
    """A check failed before the motors were ever armed."""


def _preflight(cfg: Config, mocap: NatNetClient, telemetry: Telemetry):
    body = cfg.mocap.rigid_body_id
    pose = mocap.pose(body)
    tel = telemetry.latest
    reason = safety.preflight(cfg.safety, mocap.age(body),
                              pose.position if pose else (math.nan,) * 3,
                              pose.quat if pose else (1.0, 0.0, 0.0, 0.0), tel, cfg.onboard_control)
    if reason:
        raise Abort(f"preflight failed: {reason}")
    return pose, tel


ARM_TIMEOUT_S = 1.0


def _wait_armed(telemetry: Telemetry) -> None:
    """The arming request is fire-and-forget; confirm it took before sending any thrust."""
    deadline = time.monotonic() + ARM_TIMEOUT_S
    while True:
        info = telemetry.latest.supervisor_info
        reason = safety.supervisor_problem(info, safety.SUP_ARMED)
        if reason is None:
            return
        if time.monotonic() > deadline:
            raise Abort(f"the drone did not arm: {reason}")
        time.sleep(0.02)


def _fly(cf, cfg: Config, mocap: NatNetClient, telemetry: Telemetry, gamepad: Gamepad,
         flog: FlightLog) -> str:
    body = cfg.mocap.rigid_body_id
    print(f"\nHold the deadman to take off to {cfg.flight.hover_height:.2f} m "
          f"(Ctrl+C to abort). Release it at any time to cut the motors.")
    while not gamepad.held:
        time.sleep(0.02)

    pose, _ = _preflight(cfg, mocap, telemetry)   # the drone may have been moved meanwhile
    yaw_deg = math.degrees(yaw_from_quat(pose.quat))
    plan = HoverPlan(pose.position, yaw_deg, cfg.flight.hover_height, cfg.flight.takeoff_time,
                     cfg.flight.land_time, cfg.flight.hover_duration)
    print(f"Takeoff from ({pose.position[0]:+.2f}, {pose.position[1]:+.2f}, "
          f"{pose.position[2]:+.2f}) yaw {yaw_deg:+.0f} deg")
    # mocap / kalman: the onboard controller flies position setpoints; host: the host controller
    controller = (None if cfg.onboard_control
                  else MocapPositionController(cfg.host_control, ground_z=pose.position[2]))

    arm(cf)
    try:
        _wait_armed(telemetry)
    except Abort:
        stop_motors(cf)
        raise
    period = 1.0 / cfg.flight.setpoint_hz
    next_t = time.monotonic()
    plan.start(next_t)
    last_phase = None
    try:
        while True:
            try:
                now = time.monotonic()
                if not gamepad.held:
                    return "E-STOP: deadman released"
                pose, age, tel = mocap.pose(body), mocap.age(body), telemetry.latest
                reason = safety.in_flight(cfg.safety, age, pose.position if pose else None,
                                          tel.position, pose.quat if pose else None, tel.age_s, tel)
                if reason:
                    return f"E-STOP: {reason}"
                if gamepad.take_land_request():
                    plan.land(now)

                phase, sp = plan.step(now)
                if phase is not last_phase:
                    print(f"  {phase.value}")
                    last_phase = phase
                if phase is Phase.DONE:
                    return "landed"
                cmd = None
                if controller is None:
                    cf.commander.send_position_setpoint(sp.x, sp.y, sp.z, sp.yaw_deg)
                else:
                    cmd = controller.step(now, sp, pose)
                    send_attitude(cf, cmd)
                flog.write(now, phase.value, sp, pose, age, tel, cmd)

                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
            except KeyboardInterrupt:
                if plan.landing:
                    return "E-STOP: Ctrl+C while landing"
                print("Ctrl+C — landing (Ctrl+C again cuts the motors)")
                plan.land(time.monotonic())
    finally:
        stop_motors(cf)


MODES = ("fly", "preflight", "filter-test", "calibrate-tilt", "calibrate-yaw")
ONBOARD_MOCAP_WAIT_S = 3.0


def _check_firmware(cf, cfg: Config) -> None:
    """The state source must match what the drone runs (the mocap firmware forces its estimator)."""
    mocap_fw = has_mocap_firmware(cf)
    if cfg.flight.state_source == "mocap" and not mocap_fw:
        raise Abort("state_source mocap needs the mocap-estimator firmware on the drone: "
                    "crazyflie/firmware/build.sh, then flash.py (or fly --kalman on stock firmware)")
    if cfg.flight.state_source == "kalman" and mocap_fw:
        raise Abort("the drone runs the mocap-estimator firmware, which always uses its mocap "
                    "passthrough; --kalman needs stock firmware (crazyflie/firmware/flash.py --stock)")


def _wait_onboard_mocap(cfg: Config, telemetry: Telemetry) -> None:
    """Confirm the extpose stream actually reaches the drone before trusting its state."""
    deadline = time.monotonic() + ONBOARD_MOCAP_WAIT_S
    while time.monotonic() < deadline:
        age = telemetry.latest.onboard_mocap_age_s
        if age is not None and age < cfg.safety.mocap_timeout:
            return
        time.sleep(0.05)
    raise Abort("the drone is not receiving mocap (mocapEst.ageMs never became fresh) — "
                "check the radio link and that the rigid body is tracked")


def run(cfg: Config, mode: str = "fly") -> int:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not cfg.mocap.server_ip:
        raise Abort(f"mocap.server_ip is not set (in {cfg.source} or via --set)")
    if mode == "filter-test" and not cfg.onboard_control:
        raise Abort("--filter-test checks the drone's onboard estimate; not available with host control")

    gamepad = Gamepad(cfg.gamepad).start() if mode == "fly" else None
    if gamepad:
        print(f"Gamepad: {gamepad.name}")
    mocap = NatNetClient(cfg.mocap.server_ip, local_ip=cfg.mocap.local_ip,
                         up_axis=cfg.mocap.up_axis, natnet_version=cfg.mocap.natnet_version,
                         body_rotation=cfg.mocap.body_rotation())
    try:
        mocap.start()
        mocap.wait_for_body(cfg.mocap.rigid_body_id, timeout=3.0)
        print(f"Mocap: NatNet {'.'.join(map(str, mocap.natnet_version))}, "
              f"rigid body {cfg.mocap.rigid_body_id} tracked")
        if mode == "calibrate-yaw":
            return diagnostics.calibrate_yaw(cfg, mocap)

        cflib.crtp.init_drivers()
        cache = cfg.resolve(cfg.radio.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Connecting to {cfg.radio.uri} ...")
        with SyncCrazyflie(cfg.radio.uri, cf=Crazyflie(rw_cache=str(cache))) as scf:
            cf = scf.cf
            if not self_test_passed(cf):
                raise Abort("the Crazyflie failed its power-on self-test, so its sensors and motors "
                            "never started. Power-cycle it lying flat and still; if it persists, "
                            "connect cfclient right after power-on and read the Console tab.")
            if mode == "calibrate-tilt":
                return diagnostics.calibrate_tilt(cfg, cf, mocap)
            _check_firmware(cf, cfg)
            configure(cf, cfg)
            feeder = None
            if cfg.onboard_control:
                feeder = ExtPoseFeeder(cf, mocap, cfg.mocap.rigid_body_id, cfg.mocap.extpose_hz,
                                       cfg.safety.mocap_timeout).start()
            telemetry = Telemetry(cf, estimate=cfg.onboard_control,
                                  onboard_mocap=cfg.flight.state_source == "mocap").start()
            try:
                source = cfg.flight.state_source
                if source == "mocap":
                    print(f"State: mocap passthrough onboard (mocap-estimator firmware); "
                          f"{cfg.flight.controller.upper()} flies it.")
                    _wait_onboard_mocap(cfg, telemetry)
                elif source == "kalman":
                    print(f"State: onboard Kalman filter fed by mocap; "
                          f"{cfg.flight.controller.upper()} flies it. Resetting the filter ...")
                    reset_estimator(cf, cfg.safety.estimator_timeout)
                else:
                    print("State: mocap on the host (host controller); onboard only stabilises attitude.")
                time.sleep(TELEMETRY_SETTLE_S)
                if mode == "filter-test":
                    with FlightLog(cfg.resolve(cfg.log_dir), "filter", cfg) as flog:
                        return diagnostics.filter_test(cfg, mocap, telemetry, flog)
                pose, tel = _preflight(cfg, mocap, telemetry)
                if cfg.onboard_control:
                    err_cm = math.dist(pose.position, tel.position) * 100
                    agree = f"onboard estimate within {err_cm:.1f} cm of mocap"
                else:
                    agree = "onboard attitude agrees with mocap"
                print(f"Preflight OK: {agree}, battery {tel.battery_v:.2f} V")
                if mode == "preflight":
                    return 0
                with FlightLog(cfg.resolve(cfg.log_dir), "hover", cfg) as flog:
                    outcome = _fly(cf, cfg, mocap, telemetry, gamepad, flog)
                print(f"\n{outcome}. Log: {flog.path}")
                return 0 if outcome == "landed" else 2
            finally:
                telemetry.stop()
                if feeder is not None:
                    feeder.stop()
    finally:
        mocap.stop()
        if gamepad:
            gamepad.stop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set flight.hover_height=0.6")
    ap.add_argument("--kalman", action="store_true",
                    help="stock firmware: mocap -> onboard Kalman filter -> onboard controller, instead "
                         "of the mocap-estimator firmware's passthrough (default)")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true", help="run all checks, do not fly")
    modes.add_argument("--filter-test", action="store_true",
                       help="onboard estimate vs mocap while you move the drone by hand (never arms)")
    modes.add_argument("--calibrate-yaw", action="store_true",
                       help="drone still, nose along mocap +X: measure the rigid body's yaw offset")
    modes.add_argument("--calibrate-tilt", action="store_true",
                       help="measure rigid body vs IMU tilt (drone still) and save it to the config")
    modes.add_argument("--scan", action="store_true", help="list Crazyflies the radio can see")
    modes.add_argument("--check-link", action="store_true",
                       help="connect and stream attitude + battery (no mocap, no motors)")
    modes.add_argument("--gamepad-test", action="store_true", help="print gamepad axes/buttons")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.scan:
        return diagnostics.scan()
    if args.gamepad_test:
        print_inputs()
        return 0
    try:
        overrides = args.set + (["flight.state_source=kalman"] if args.kalman else [])
        cfg = load_config(args.config, overrides)
        if args.check_link:
            return diagnostics.check_link(cfg)
        mode = ("preflight" if args.preflight_only else "filter-test" if args.filter_test
                else "calibrate-tilt" if args.calibrate_tilt
                else "calibrate-yaw" if args.calibrate_yaw else "fly")
        return run(cfg, mode)
    except (Abort, TimeoutError, ValueError, RuntimeError) as exc:
        print(f"\nABORT: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nAborted before takeoff.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
