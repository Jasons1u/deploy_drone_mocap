#!/usr/bin/env python3
"""Motor test bench: one slider per motor, live plot of what the drone actually applies.

    python3 crazyflie/scripts/motor_test.py                 # full range, 0..65535
    python3 crazyflie/scripts/motor_test.py --max-pwm 20000 # optional cap (lift-off is ~36000)

Controls: drag the M1..M4 sliders, click ENABLE to let the motors spin, STOP (or the space bar) to
cut them. Closing the window, Ctrl+C and any error also cut them.

Plots (all measured by the drone; nothing is modelled):
  motor PWM   motor.m1..m4 — the PWM the firmware writes to each motor (dashed: slider value)
  battery     pm.vbat      — sags with the current the motors draw
  yaw rate    gyro.z       — the reaction torque: with the drone free to turn, powering one
                             diagonal pair (M1+M3 or M2+M4) should turn it one way, the other pair
                             the other way
The Crazyflie has no thrust sensor; for grams, stand it on a kitchen scale.

!! motorPowerSet overrides the motors below every firmware safety net (disarm, commander timeout,
!! emergency stop). If the radio link drops, the motors KEEP the last values until you switch the
!! drone off. Props off, or the drone strapped down, and a hand on the power switch.
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
import threading
import time

import cflib.crtp
import matplotlib.pyplot as plt
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from cf_mocap.config import load_config
from cf_mocap.link import self_test_passed
from cf_mocap.motor_bench import MOTORS, PWM_MAX, MotorBench, stop_or_warn

WINDOW_S = 20.0         # plot history
LINK_TIMEOUT_S = 0.5


class Recorder:
    """50 Hz log block: applied PWM, battery, yaw rate."""

    def __init__(self, cf):
        self._lock = threading.Lock()
        self.t0 = time.monotonic()
        self.last_rx = None
        n = int(WINDOW_S * 50)
        self.t = collections.deque(maxlen=n)
        self.pwm = [collections.deque(maxlen=n) for _ in MOTORS]
        self.req = [collections.deque(maxlen=n) for _ in MOTORS]
        self.vbat = collections.deque(maxlen=n)
        self.gyro_z = collections.deque(maxlen=n)
        self.requested = [0, 0, 0, 0]
        self._log = LogConfig(name="motor_test", period_in_ms=20)          # 16 B
        for m in MOTORS:
            self._log.add_variable(f"motor.{m}", "uint16_t")
        self._log.add_variable("pm.vbat", "float")
        self._log.add_variable("gyro.z", "float")
        cf.log.add_config(self._log)
        self._log.data_received_cb.add_callback(self._on_data)

    def start(self):
        self._log.start()
        return self

    def stop(self):
        self._log.stop()

    def _on_data(self, _ts, data, _conf):
        now = time.monotonic()
        with self._lock:
            self.last_rx = now
            self.t.append(now - self.t0)
            for i, m in enumerate(MOTORS):
                self.pwm[i].append(data[f"motor.{m}"])
                self.req[i].append(self.requested[i])
            self.vbat.append(data["pm.vbat"])
            self.gyro_z.append(data["gyro.z"])

    def snapshot(self):
        with self._lock:
            return (list(self.t), [list(p) for p in self.pwm], [list(r) for r in self.req],
                    list(self.vbat), list(self.gyro_z), self.last_rx)


def run_gui(bench: MotorBench, rec: Recorder) -> None:
    fig = plt.figure(figsize=(13, 8))
    fig.canvas.manager.set_window_title("Crazyflie motor test")
    title = fig.suptitle("")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")

    ax_pwm = fig.add_axes((0.30, 0.66, 0.66, 0.27))
    ax_bat = fig.add_axes((0.30, 0.37, 0.66, 0.22), sharex=ax_pwm)
    ax_yaw = fig.add_axes((0.30, 0.08, 0.66, 0.22), sharex=ax_pwm)
    applied = [ax_pwm.plot([], [], color=c, lw=1.5, label=f"M{i + 1} applied")[0]
               for i, c in enumerate(colors)]
    wanted = [ax_pwm.plot([], [], color=c, lw=0.8, ls="--")[0] for c in colors]
    ax_pwm.set_ylabel("motor PWM (motor.mN)")
    ax_pwm.set_ylim(0, max(bench.cap * 1.1, 1000))
    ax_pwm.legend(loc="upper left", fontsize=8, ncol=4)
    bat_line, = ax_bat.plot([], [], "k", lw=1.2)
    ax_bat.set_ylabel("battery [V] (pm.vbat)")
    yaw_line, = ax_yaw.plot([], [], "tab:purple", lw=1.2)
    ax_yaw.axhline(0, color="gray", lw=0.6)
    ax_yaw.set_ylabel("yaw rate [deg/s] (gyro.z)\n> 0 = counter-clockwise")
    ax_yaw.set_xlabel("t [s]")
    for ax in (ax_pwm, ax_bat, ax_yaw):
        ax.grid(True)

    sliders = []
    for i, c in enumerate(colors):
        ax = fig.add_axes((0.04 + i * 0.055, 0.25, 0.03, 0.65))
        s = Slider(ax, f"M{i + 1}", 0, bench.cap, valinit=0, valstep=100, orientation="vertical",
                   color=c)
        s.on_changed(lambda v, i=i: set_motor(i, v))
        sliders.append(s)

    def set_motor(i, v):
        bench.set_requested(i, v)
        rec.requested[i] = bench.requested[i]

    def stop(_event=None):
        stop_or_warn(bench)
        for s in sliders:
            s.set_val(0)

    btn_enable = Button(fig.add_axes((0.04, 0.12, 0.2, 0.06)), "ENABLE motors", color="0.85")
    btn_stop = Button(fig.add_axes((0.04, 0.04, 0.2, 0.07)), "STOP  (space)", color="salmon",
                      hovercolor="red")
    btn_enable.on_clicked(lambda _e: bench.enable())
    btn_stop.on_clicked(stop)
    fig.canvas.mpl_connect("key_press_event", lambda e: stop() if e.key == " " else None)
    fig.canvas.mpl_connect("close_event", lambda _e: stop_or_warn(bench))

    def update(_frame):
        now = time.monotonic()
        bench.push(now)
        t, pwm, req, vbat, gyro_z, last_rx = rec.snapshot()
        link_ok = last_rx is not None and now - last_rx < LINK_TIMEOUT_S
        if not link_ok:
            title.set_text("NO TELEMETRY — link lost? Motors may still be running: switch the drone off")
            title.set_color("red")
        elif bench.enabled:
            title.set_text(f"MOTORS ENABLED (cap {bench.cap})   requested {bench.requested}")
            title.set_color("darkorange")
        else:
            title.set_text(f"motors disabled — set sliders, then ENABLE (cap {bench.cap})")
            title.set_color("green")
        if t:
            for i in range(4):
                applied[i].set_data(t, pwm[i])
                wanted[i].set_data(t, req[i])
            bat_line.set_data(t, vbat)
            yaw_line.set_data(t, gyro_z)
            ax_pwm.set_xlim(max(0.0, t[-1] - WINDOW_S), t[-1] + 0.5)
            ax_bat.set_ylim(min(vbat) - 0.05, max(vbat) + 0.05)
            span = max(20.0, max(abs(v) for v in gyro_z) * 1.2)
            ax_yaw.set_ylim(-span, span)
        return ()

    _anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)  # noqa: F841 (keep ref)
    plt.show()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--max-pwm", type=int, default=PWM_MAX,
                    help=f"optional slider cap, 1..{PWM_MAX} (default: full range; lift-off ~36000)")
    ap.add_argument("--uri", help="radio URI (default: radio.uri from config/hover.yaml)")
    args = ap.parse_args(argv)
    if not 0 < args.max_pwm <= PWM_MAX:
        ap.error(f"--max-pwm must be in 1..{PWM_MAX}")
    logging.getLogger("cflib").setLevel(logging.ERROR)
    cfg = load_config()
    uri = args.uri or cfg.radio.uri

    print("!! Motors spin below every firmware safety net: if the link drops they keep running.\n"
          "!! Props off or drone strapped down; keep a hand on the power switch.\n")
    cflib.crtp.init_drivers()
    print(f"Connecting to {uri} ...")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=str(cfg.resolve(cfg.radio.cache_dir)))) as scf:
        cf = scf.cf
        if not self_test_passed(cf):
            print("ABORT: the Crazyflie failed its power-on self-test, so its motors never started. "
                  "Power-cycle it lying flat and still.")
            return 1
        bench = MotorBench(cf, args.max_pwm)
        rec = Recorder(cf).start()
        try:
            bench.stop()                    # known state: override off, all zero
            run_gui(bench, rec)
        finally:
            stopped = stop_or_warn(bench)
            rec.stop()
            if stopped:
                print("Motors stopped, override disabled (confirmed by the drone).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
