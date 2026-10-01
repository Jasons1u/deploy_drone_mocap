"""Everything that talks to the Crazyflie over cflib, behind a small surface.

  configure(cf, cfg)        estimator + controller for flight.state_source
  has_mocap_firmware(cf)    True if the drone runs crazyflie/firmware (mocap passthrough estimator)
  ExtPoseFeeder             mocap pose -> cf.extpos at a fixed rate (fresh frames only)
  Telemetry                 latest Kalman position, attitude, battery (log blocks)
  reset_estimator(...)      reset the Kalman filter and wait for it to converge on mocap
  send_attitude(cf, cmd)    roll/pitch/yaw-rate/thrust setpoint (the mocap-state controller's output)
  self_test_passed(cf)      False if the drone failed its power-on self-test (flight system not started)
  arm / stop_motors         arming + the stop setpoint used for both landing and E-STOP
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from cflib.crazyflie.log import LogConfig

from mocap_natnet import NatNetClient

from .config import CONTROLLERS, Config

log = logging.getLogger(__name__)

ESTIMATOR_COMPLEMENTARY = 1   # attitude from the IMU only; ignores extpose
ESTIMATOR_KALMAN = 2
KALMAN_VAR_WINDOW = 10         # samples of position variance that must be flat
KALMAN_VAR_SPREAD = 0.001      # [m^2] max spread over the window (same test as cflib's helper)


def has_mocap_firmware(cf) -> bool:
    """The mocap-estimator build (crazyflie/firmware) exposes the mocapEst parameter group."""
    return cf.param.toc.get_element_by_complete_name("mocapEst.velAlpha") is not None


def configure(cf, cfg: Config) -> None:
    """mocap:  the mocap-estimator firmware forces its passthrough estimator; pick the controller.
    kalman: extpose -> Kalman + the chosen onboard controller.
    host:   the host sends attitude setpoints; the onboard PID only runs the attitude loops."""
    source = cfg.flight.state_source
    if source == "kalman":
        cf.param.set_value("stabilizer.estimator", ESTIMATOR_KALMAN)
        cf.param.set_value("locSrv.extPosStdDev", cfg.mocap.ext_pos_std_dev)
        cf.param.set_value("locSrv.extQuatStdDev", cfg.mocap.ext_quat_std_dev)
    elif source == "host":
        cf.param.set_value("stabilizer.estimator", ESTIMATOR_COMPLEMENTARY)
    controller = "pid" if source == "host" else cfg.flight.controller
    cf.param.set_value("stabilizer.controller", CONTROLLERS[controller])


class ExtPoseFeeder:
    """Forwards the latest mocap pose to the onboard Kalman filter.

    Stale poses are NOT resent: repeating an old position would tell the filter the drone is
    still there, with full confidence. Without fresh frames the filter coasts on the IMU and
    the in-flight safety check (mocap_timeout) ends the flight.
    """

    def __init__(self, cf, mocap: NatNetClient, body_id: int, rate_hz: float, max_age: float):
        self._cf = cf
        self._mocap = mocap
        self._body_id = body_id
        self._period = 1.0 / rate_hz
        self._max_age = max_age
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="extpose", daemon=True)
        self._last_frame = None
        self.sent = 0

    def start(self) -> "ExtPoseFeeder":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            pose = self._mocap.pose(self._body_id)
            if (pose is not None and pose.frame_number != self._last_frame
                    and time.monotonic() - pose.stamp < self._max_age):
                w, qx, qy, qz = pose.quat
                self._cf.extpos.send_extpose(*pose.position, qx, qy, qz, w)
                self._last_frame = pose.frame_number
                self.sent += 1
            next_t += self._period
            time.sleep(max(0.0, next_t - time.monotonic()))


@dataclass(frozen=True)
class TelemetrySample:
    position: tuple[float, float, float] | None = None   # Kalman estimate [m]
    velocity: tuple[float, float, float] | None = None   # Kalman estimate [m/s], world
    rpy_deg: tuple[float, float, float] | None = None    # right-handed ZYX, same as mocap
    motors: tuple[int, int, int, int] | None = None      # PWM m1..m4, 0..65535
    battery_v: float | None = None
    supervisor_info: int | None = None                   # supervisor.info (safety.SUP_*)
    age_s: float = float("inf")                          # since the last 50 Hz attitude packet
    onboard_mocap_age_s: float | None = None             # mocap firmware: drone's last extpose


class Telemetry:
    """Latest values from three log blocks (a CRTP log packet holds at most 26 bytes).

    The firmware logs pitch in the legacy CF2 convention (positive = nose up); it is negated here
    so roll/pitch/yaw are right-handed like mocap and can be compared directly.

    estimate=False (host control) skips the onboard position/velocity estimate, which the
    complementary estimator does not produce; those fields stay None. onboard_mocap=True (the
    mocap-estimator firmware) adds how long ago the drone itself last received mocap.
    """

    def __init__(self, cf, estimate: bool = True, onboard_mocap: bool = False):
        self._lock = threading.Lock()
        self._estimate = estimate
        self._position = self._velocity = self._roll_pitch = self._yaw = None
        self._motors = self._battery_v = self._supervisor_info = None
        self._onboard_mocap = onboard_mocap
        self._onboard_mocap_age = None
        self._last_rx = None

        state = LogConfig(name="state", period_in_ms=20)                  # <= 16 B
        state_vars = ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z") if estimate else ()
        for v in (*state_vars, "stabilizer.yaw"):
            state.add_variable(v, "float")
        attitude = LogConfig(name="attitude", period_in_ms=20)            # <= 22 B
        for v in ("stabilizer.roll", "stabilizer.pitch"):
            attitude.add_variable(v, "float")
        for v in ("motor.m1", "motor.m2", "motor.m3", "motor.m4"):
            attitude.add_variable(v, "uint16_t")
        if estimate:
            for v in ("stateEstimateZ.vx", "stateEstimateZ.vy", "stateEstimateZ.vz"):   # [mm/s]
                attitude.add_variable(v, "int16_t")
        status = LogConfig(name="status", period_in_ms=100)               # <= 8 B
        status.add_variable("pm.vbat", "float")
        status.add_variable("supervisor.info", "uint16_t")
        if onboard_mocap:
            status.add_variable("mocapEst.ageMs", "uint16_t")

        state.data_received_cb.add_callback(self._on_state)
        attitude.data_received_cb.add_callback(self._on_attitude)
        status.data_received_cb.add_callback(self._on_status)
        self._configs = (state, attitude, status)
        for c in self._configs:
            cf.log.add_config(c)

    def start(self) -> "Telemetry":
        for c in self._configs:
            c.start()
        return self

    def stop(self) -> None:
        for c in self._configs:
            c.stop()

    @property
    def latest(self) -> TelemetrySample:
        with self._lock:
            rpy = None
            if self._roll_pitch is not None and self._yaw is not None:
                rpy = (*self._roll_pitch, self._yaw)
            age = float("inf") if self._last_rx is None else time.monotonic() - self._last_rx
            return TelemetrySample(self._position, self._velocity, rpy, self._motors,
                                   self._battery_v, self._supervisor_info, age,
                                   self._onboard_mocap_age)

    def _on_state(self, _ts, data, _conf):
        with self._lock:
            if self._estimate:
                self._position = (data["stateEstimate.x"], data["stateEstimate.y"],
                                  data["stateEstimate.z"])
            self._yaw = data["stabilizer.yaw"]

    def _on_attitude(self, _ts, data, _conf):
        with self._lock:
            self._last_rx = time.monotonic()
            self._roll_pitch = (data["stabilizer.roll"], -data["stabilizer.pitch"])
            self._motors = tuple(data[f"motor.m{i}"] for i in range(1, 5))
            if self._estimate:
                self._velocity = tuple(data[f"stateEstimateZ.v{a}"] / 1000.0 for a in "xyz")

    def _on_status(self, _ts, data, _conf):
        with self._lock:
            self._battery_v = data["pm.vbat"]
            self._supervisor_info = data["supervisor.info"]
            if self._onboard_mocap:
                self._onboard_mocap_age = data["mocapEst.ageMs"] / 1000.0   # 65.535 = never


def reset_estimator(cf, timeout: float) -> None:
    """Reset the Kalman filter (it must already be receiving extpose) and wait for convergence."""
    cf.param.set_value("kalman.resetEstimation", 1)
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", 0)

    history = {axis: [] for axis in ("X", "Y", "Z")}
    converged = threading.Event()

    def on_var(_ts, data, _conf):
        for axis, h in history.items():
            h.append(data[f"kalman.varP{axis}"])
            del h[:-KALMAN_VAR_WINDOW]
        if all(len(h) == KALMAN_VAR_WINDOW and max(h) - min(h) < KALMAN_VAR_SPREAD
               for h in history.values()):
            converged.set()

    var_log = LogConfig(name="kalman_var", period_in_ms=100)
    for axis in history:
        var_log.add_variable(f"kalman.varP{axis}", "float")
    cf.log.add_config(var_log)
    var_log.data_received_cb.add_callback(on_var)
    var_log.start()
    try:
        if not converged.wait(timeout):
            raise TimeoutError(f"Kalman filter did not converge within {timeout:.0f} s — "
                               f"is extpose reaching the drone?")
    finally:
        var_log.stop()
        var_log.delete()


def self_test_passed(cf) -> bool:
    """A drone that failed its power-on self-test answers on the radio but never starts its
    sensors, estimator, stabilizer or motors: every value it logs is frozen and it ignores arming."""
    return int(cf.param.get_value("system.selftestPassed")) == 1


def send_attitude(cf, cmd) -> None:
    """mocap_control.AttitudeCommand -> commander. Conventions (firmware crtp_commander_rpyt.c):
    roll/pitch go through right-handed (cflib negates pitch into the legacy frame); the legacy yaw
    rate is clockwise-positive, so the CCW rate is negated."""
    cf.commander.send_setpoint(cmd.roll_deg, cmd.pitch_deg, -cmd.yawrate_deg_s, cmd.thrust)


def arm(cf) -> None:
    cf.supervisor.send_arming_request(True)   # falls back to the legacy channel on old firmware
    time.sleep(0.2)
    cf.commander.send_setpoint(0, 0, 0, 0)     # unlock the commander (first packet must be zero)


def stop_motors(cf) -> None:
    """Cut motors and disarm. Used after touchdown and for E-STOP alike."""
    cf.commander.send_stop_setpoint()
    cf.commander.send_notify_setpoint_stop()
    cf.supervisor.send_arming_request(False)
