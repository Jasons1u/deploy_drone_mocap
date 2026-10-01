"""Go/no-go checks. Each returns a human-readable reason to abort, or None when it is safe.

Pure functions of already-measured values, so every threshold is unit-tested without a drone.
"""

from __future__ import annotations

import math

from mocap_natnet.frames import rpy_from_quat

from .config import SafetyConfig


# supervisor.info bits (firmware supervisor.c; same as cflib.crazyflie.supervisor)
SUP_CAN_BE_ARMED, SUP_ARMED, SUP_TUMBLED, SUP_LOCKED, SUP_CRASHED = 1 << 0, 1 << 1, 1 << 5, 1 << 6, 1 << 7
TELEMETRY_TIMEOUT_S = 0.5   # no 50 Hz telemetry this long = the radio link is gone


def supervisor_problem(info: int | None, need: int) -> str | None:
    """Reason the supervisor bitfield lacks `need` (SUP_CAN_BE_ARMED or SUP_ARMED), else None."""
    if info is None:
        return "no supervisor state received from the Crazyflie"
    if info & need:
        return None
    for bit, what in ((SUP_CRASHED, "crashed"), (SUP_LOCKED, "locked (needs a reboot)"),
                      (SUP_TUMBLED, "tumbled (upside down or tilted)")):
        if info & bit:
            return f"supervisor reports the drone {what} (supervisor.info=0x{info:04x})"
    state = "can be armed" if need == SUP_CAN_BE_ARMED else "armed"
    return f"supervisor does not report '{state}' (supervisor.info=0x{info:04x})"


def tilt_deg(quat) -> float:
    """Angle between body up and world up [deg]. quat = (w, x, y, z)."""
    _, x, y, _ = quat
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def preflight(cfg: SafetyConfig, mocap_age: float, mocap_pos, mocap_quat, tel,
              onboard_estimate: bool = True) -> str | None:
    """tel: link.TelemetrySample. onboard_estimate (state_source mocap / kalman): the drone flies on
    its own state, so check that state against mocap. Otherwise (host control) check the onboard IMU
    attitude against mocap: a mismatch means the rigid body axes are wrong."""
    if mocap_age > cfg.mocap_timeout:
        return f"mocap not streaming (last valid frame {mocap_age:.2f} s ago)"
    if not cfg.geofence.contains(mocap_pos):
        return f"drone at {_fmt(mocap_pos)} is outside the geofence"
    if tilt_deg(mocap_quat) > cfg.max_preflight_tilt_deg:
        return f"drone tilted {tilt_deg(mocap_quat):.0f} deg — place it level"
    if tel.age_s > TELEMETRY_TIMEOUT_S:
        return f"no telemetry from the Crazyflie for {tel.age_s:.1f} s (radio link?)"
    reason = _onboard_mocap_problem(cfg, tel)
    if reason:
        return reason
    if onboard_estimate:
        if tel.position is None:
            return "no state estimate received from the Crazyflie"
        err = math.dist(mocap_pos, tel.position)
        if err > cfg.preflight_max_error:
            return (f"onboard estimate {_fmt(tel.position)} is {err * 100:.1f} cm from mocap "
                    f"{_fmt(mocap_pos)} — check rigid body ID / up axis / body orientation")
    else:
        if tel.rpy_deg is None:
            return "no attitude received from the Crazyflie"
        m_roll, m_pitch, _ = (math.degrees(a) for a in rpy_from_quat(mocap_quat))
        worst = max(abs(tel.rpy_deg[0] - m_roll), abs(tel.rpy_deg[1] - m_pitch))
        if worst > cfg.max_attitude_mismatch_deg:
            return (f"onboard roll/pitch ({tel.rpy_deg[0]:+.1f}, {tel.rpy_deg[1]:+.1f}) "
                    f"disagree with mocap ({m_roll:+.1f}, {m_pitch:+.1f}) by {worst:.1f} deg — "
                    f"rigid body axes don't match the drone; run --calibrate-tilt or recreate it")
    if tel.battery_v is None:
        return "no battery reading received"
    if tel.battery_v < cfg.min_battery_v:
        return f"battery {tel.battery_v:.2f} V below {cfg.min_battery_v:.2f} V"
    return supervisor_problem(tel.supervisor_info, SUP_CAN_BE_ARMED)


def _onboard_mocap_problem(cfg: SafetyConfig, tel) -> str | None:
    """Mocap-estimator firmware: the drone flies on the mocap it received; is it still arriving?"""
    age = getattr(tel, "onboard_mocap_age_s", None)
    if age is not None and age > cfg.mocap_timeout:
        return f"the drone has not received mocap for {age:.2f} s (extpose not reaching it)"
    return None


def in_flight(cfg: SafetyConfig, mocap_age: float, mocap_pos, estimate_pos,
              mocap_quat=None, telemetry_age: float = 0.0, tel=None) -> str | None:
    if mocap_age > cfg.mocap_timeout:
        return f"mocap lost for {mocap_age:.2f} s"
    if telemetry_age > TELEMETRY_TIMEOUT_S:
        return f"no telemetry from the Crazyflie for {telemetry_age:.1f} s — radio link lost"
    if tel is not None:
        reason = _onboard_mocap_problem(cfg, tel)
        if reason:
            return reason
    if not cfg.geofence.contains(mocap_pos):
        return f"left the geofence at {_fmt(mocap_pos)}"
    if mocap_quat is not None and tilt_deg(mocap_quat) > cfg.max_flight_tilt_deg:
        return f"tilted {tilt_deg(mocap_quat):.0f} deg — lost attitude control"
    if estimate_pos is not None:
        err = math.dist(mocap_pos, estimate_pos)
        if err > cfg.max_estimate_error:
            return f"onboard estimate diverged {err * 100:.0f} cm from mocap"
    return None


def _fmt(p) -> str:
    return "(" + ", ".join(f"{v:+.2f}" for v in p) + ")"
