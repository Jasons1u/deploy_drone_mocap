#!/usr/bin/env python3
"""Yaw authority / spin-direction check: does the mixer's yaw differential twist the drone the way
the firmware expects?

    python3 crazyflie/scripts/spin_direction.py                  # base 24000, differential 6000

Set-up: drone ON THE FLOOR and free to turn (holding it hides the twist), props on, nothing within
reach of the props, hand near the power switch.

Why a differential: on the floor, friction beats the twist of one or two motors. All four at an
equal base lighten the drone (24000 on all four is ~67% of hover, ~36000, so it cannot lift) and
then the yaw command the controller would give is applied on top, exactly as in flight:
  1. all four at base                     -> drift at equal power (should be small)
  2. M1+M3 base+diff, M2+M4 base-diff     -> firmware expects CLOCKWISE twist
  3. M1+M3 base-diff, M2+M4 base+diff     -> firmware expects COUNTER-CLOCKWISE twist
(controller_pid.c:131 negates the yaw output; the mixer, power_distribution_quadrotor.c:89-92,
adds it to M1/M3 and subtracts it from M2/M4.)

Measured with the drone's gyro (gyro.z > 0 = counter-clockwise from above).
!! Uses the motorPowerSet override (cf_mocap/motor_bench.py): below every firmware safety net.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from cf_mocap.config import load_config
from cf_mocap.link import self_test_passed
from cf_mocap.motor_bench import MOTORS, MotorBench, stop_or_warn

RAMP_S, HOLD_S, SETTLE_S = 1.5, 1.5, 0.5     # the rate is averaged over the last HOLD_S - SETTLE_S
ABORT_RATE_DEG_S = 150.0
TELEMETRY_TIMEOUT_S = 0.3
MIN_AUTHORITY_DEG_S = 10.0   # |rate(CCW cmd) - rate(CW cmd)| below this: the differential does nothing
MAX_DRIFT_DEG_S = 30.0       # drift at equal power beyond this: the motors do not cancel
MAX_TOTAL_PWM = 4 * 30000    # keeps the total well below hover on all four (~4 x 36000)


@dataclass(frozen=True)
class Result:
    drift: float        # deg/s, all four equal
    cw_cmd: float       # deg/s, with M1+M3 up  (firmware expects < drift)
    ccw_cmd: float      # deg/s, with M2+M4 up  (firmware expects > drift)


def verdict(r: Result) -> tuple[str, str]:
    """('ok' | 'reversed' | 'no_authority', explanation)."""
    authority = r.ccw_cmd - r.cw_cmd           # firmware expects > 0
    drift_note = ""
    if abs(r.drift) > MAX_DRIFT_DEG_S:
        drift_note = (f" Also: at equal power it drifted {r.drift:+.0f} deg/s, far from zero, so the "
                      f"four motors' torques do not cancel.")
    if abs(authority) < MIN_AUTHORITY_DEG_S:
        return "no_authority", (
            f"The yaw differential barely changes the twist ({authority:+.0f} deg/s): the controller "
            f"has almost no yaw authority. Typical cause: the motors do not pair up as two "
            f"clockwise and two counter-clockwise (one or more spin the wrong way)." + drift_note)
    if authority < 0:
        return "reversed", (
            f"The differential twists the drone the OPPOSITE way to what the firmware expects "
            f"({authority:+.0f} deg/s): every yaw correction makes the spin worse. Reverse the spin "
            f"direction of all four motors in the ESC configuration (props stay on their motors), "
            f"or swap to the Bitcraze layout." + drift_note)
    return "ok", (f"Yaw differential works as the firmware expects ({authority:+.0f} deg/s)."
                  + drift_note)


class GyroLog:
    def __init__(self, cf):
        self._lock = threading.Lock()
        self.samples = []            # (monotonic t, gyro.z deg/s, (m1..m4))
        self._cfg = LogConfig(name="spin_dir", period_in_ms=10)
        self._cfg.add_variable("gyro.z", "float")
        for m in MOTORS:
            self._cfg.add_variable(f"motor.{m}", "uint16_t")
        cf.log.add_config(self._cfg)
        self._cfg.data_received_cb.add_callback(self._on_data)

    def start(self):
        self._cfg.start()
        return self

    def stop(self):
        self._cfg.stop()

    def _on_data(self, _ts, data, _conf):
        with self._lock:
            self.samples.append((time.monotonic(), data["gyro.z"],
                                 tuple(data[f"motor.{m}"] for m in MOTORS)))

    def since(self, t0):
        with self._lock:
            return [s for s in self.samples if s[0] >= t0]


class Aborted(Exception):
    pass


def hold(bench: MotorBench, log: GyroLog, target, seconds: float, ramp_from=None) -> float:
    """Drive all four motors to `target` (ramping from `ramp_from` if given) and hold.
    Returns the mean gyro.z over the settled part of the hold."""
    t0 = time.monotonic()
    while True:
        now = time.monotonic()
        frac = 1.0 if ramp_from is None else min((now - t0) / RAMP_S, 1.0)
        for i in range(4):
            start = target[i] if ramp_from is None else ramp_from[i]
            bench.set_requested(i, start + frac * (target[i] - start))
        bench.push(now)
        if not log.since(now - TELEMETRY_TIMEOUT_S):
            raise Aborted("telemetry stopped")
        recent = log.since(now - 0.05)
        if recent and abs(recent[-1][1]) > ABORT_RATE_DEG_S:
            raise Aborted(f"turning faster than {ABORT_RATE_DEG_S:.0f} deg/s")
        if now - t0 >= seconds:
            break
        time.sleep(0.02)
    settled = [r for _, r, _ in log.since(t0 + seconds - (HOLD_S - SETTLE_S))]
    applied = log.since(time.monotonic() - 0.1)
    if applied:
        print(f"    motors reported PWM {list(applied[-1][2])} (M1..M4)")
    return sum(settled) / len(settled) if settled else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", type=int, default=24000, help="PWM on all four motors (default 24000)")
    ap.add_argument("--diff", type=int, default=6000, help="yaw differential (default 6000)")
    ap.add_argument("--yes", action="store_true", help="skip the set-up confirmation")
    ap.add_argument("--uri", help="radio URI (default: radio.uri from config/hover.yaml)")
    args = ap.parse_args(argv)
    if not 0 < args.diff < args.base or 4 * args.base > MAX_TOTAL_PWM or args.base + args.diff > 40000:
        ap.error(f"need 0 < diff < base, 4 x base <= {MAX_TOTAL_PWM} and base + diff <= 40000")

    b, d = args.base, args.diff
    phases = (("all four equal", (b, b, b, b)),
              ("M1+M3 up, M2+M4 down (expect CW)", (b + d, b - d, b + d, b - d)),
              ("M1+M3 down, M2+M4 up (expect CCW)", (b - d, b + d, b - d, b + d)))

    print("Drone on the FLOOR, free to turn, props clear of everything, hand near the power switch.")
    if not args.yes and input("Ready? type 'yes': ").strip().lower() != "yes":
        return 1
    logging.getLogger("cflib").setLevel(logging.ERROR)
    cfg = load_config()
    cflib.crtp.init_drivers()
    with SyncCrazyflie(args.uri or cfg.radio.uri,
                       cf=Crazyflie(rw_cache=str(cfg.resolve(cfg.radio.cache_dir)))) as scf:
        cf = scf.cf
        if not self_test_passed(cf):
            print("ABORT: the Crazyflie failed its power-on self-test. Power-cycle it.")
            return 1
        bench = MotorBench(cf, b + d)
        log = GyroLog(cf).start()
        rates = []
        try:
            bench.stop()
            time.sleep(0.5)
            bench.enable()
            previous = (0, 0, 0, 0)
            for name, target in phases:
                print(f"  {name}: {list(target)}")
                ramp = previous if previous == (0, 0, 0, 0) else None
                rate = hold(bench, log, target, (RAMP_S if ramp else 0.0) + HOLD_S, ramp)
                rates.append(rate)
                print(f"    yaw rate {rate:+6.1f} deg/s ({'CCW' if rate > 0 else 'CW'})")
                previous = target
        except Aborted as exc:
            print(f"  stopped early: {exc}")
        finally:
            stopped = stop_or_warn(bench)
            log.stop()
    if not stopped:
        return 1
    if len(rates) < 3:
        print("\nIncomplete: no verdict. Lower --base if it turned too fast.")
        return 2
    kind, text = verdict(Result(*rates))
    print(f"\n{kind.upper()}: {text}")
    return 0 if kind == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
