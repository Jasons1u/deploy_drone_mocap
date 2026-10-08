"""Shared flight phases: preflight checks, deadman arming, the full-thrust launch + climb, and the landing descent."""

from __future__ import annotations

import math
import sys
import time

from cfdeploy.geometry import tilt_deg, wrap180, yaw_deg
from cfdeploy.radio import SUP_ARMED, SUP_CAN_BE_ARMED


# #### // Before takeoff // ####
def preflight(cfg, mocap, tel, sent: int) -> tuple:
    """Onboard estimate vs mocap, battery, level, supervisor; returns the takeoff pose (p0, yaw0)."""
    c = cfg.preflight
    body, age = mocap.body(cfg.mocap.drone_body)
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
    print(f"battery {vbat:.2f} V, supervisor.info 0x{sup:04x}, extpose sent {sent}")
    problems = []
    if age > c.max_mocap_age:
        problems.append(f"mocap stale ({age:.2f} s)")
    if onboard_age is None or onboard_age > c.max_onboard_age_ms:
        problems.append(f"drone not receiving mocap (ageMs {onboard_age})")
    if err > c.max_position_error:
        problems.append(f"onboard position {err * 100:.1f} cm from mocap")
    if abs(yaw_err) > c.max_yaw_error:
        problems.append(f"onboard yaw {yaw_err:+.1f} deg from mocap")
    if math.hypot(*vel) > c.max_rest_speed:
        problems.append(f"onboard velocity {math.hypot(*vel):.3f} m/s at rest")
    if tilt_deg(body.quat) > c.max_tilt:
        problems.append("not level")
    if vbat < c.min_vbat:
        problems.append(f"battery {vbat:.2f} V < {c.min_vbat}")
    if not sup & SUP_CAN_BE_ARMED:
        problems.append(f"supervisor cannot arm (0x{sup:04x})")
    if problems:
        raise RuntimeError("preflight: " + "; ".join(problems))
    print("Preflight OK")
    return p0, yaw0


def arm_with_deadman(cf, cfg, pad, kb, tel, label: str) -> str | None:
    """Wait for LT released then held, arm, confirm; returns None when armed, else why it was cancelled."""
    d = cfg.gamepad.deadman
    print(f"\nRelease LT, then hold it to take off ({label}) ({d.arm_window:.0f} s). "
          f"Releasing LT at any time = motors off.")
    seen_release, held_since, deadline = False, None, time.monotonic() + d.arm_window
    next_report = time.monotonic() + 5.0
    while held_since is None or time.monotonic() - held_since < d.arm_hold:
        if kb.kill or kb.land or time.monotonic() > deadline:
            a = pad.axis(d.axis)
            outcome = (f"cancelled before arming (LT {a if a is None else round(a, 2)}, "
                       f"release seen {seen_release}, gamepad alive {pad.alive})")
            print(outcome, file=sys.stderr)
            return outcome
        if time.monotonic() > next_report:
            a = pad.axis(d.axis)
            hint = ("LT is already pulled: release it, then pull again" if not seen_release
                    else f"need > {d.arm_threshold} for {d.arm_hold:g} s")
            print(f"  waiting for LT: reads {a if a is None else round(a, 2)} ({hint})")
            next_report += 5.0
        a = pad.axis(d.axis)
        seen_release |= a is not None and a < -d.arm_threshold
        held = seen_release and pad.trigger_held(d.axis, d.arm_threshold)
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


# #### // Takeoff: short level burst at launch.thrust (fast off the ground, motor headroom kept), then a climb ramp // ####
class Takeoff:
    def __init__(self, launch_cfg, p0, z_top: float, now: float):
        self.cfg, self.p0, self.z_top, self.t_start = launch_cfg, p0, z_top, now
        self.t_climb = self.z_climb = None

    @property
    def thrust(self) -> int:
        return int(self.cfg.thrust * 65535)

    def lifted_off(self, now, p) -> bool:
        """During the burst: True once above the launch height (the controller takes over here, level or not)."""
        if p[2] > self.p0[2] + self.cfg.height:
            self.t_climb, self.z_climb = now, p[2]
            return True
        return False

    def timed_out(self, now) -> bool:
        return now - self.t_start > self.cfg.timeout

    def climb(self, now) -> tuple:
        """Ramp setpoint from the lift-off height to z_top: ((x, y, z), done)."""
        frac = min(1.0, (now - self.t_climb) / self.cfg.ramp_time)
        return (self.p0[0], self.p0[1], self.z_climb + (self.z_top - self.z_climb) * frac), frac >= 1.0

    def tilt_guard_armed(self, now) -> bool:
        """The airborne flip guard starts a moment after lift-off (never during the burst)."""
        return self.t_climb is not None and now - self.t_climb > self.cfg.tilt_guard_delay


# #### // Landing: descend at constant speed, motors off once down // ####
class Landing:
    def __init__(self, landing_cfg, now: float, z_start: float, z_floor: float):
        self.cfg, self.t0, self.z_start, self.z_floor = landing_cfg, now, z_start, z_floor
        self.ground_since = None

    def z(self, now) -> float:
        return max(self.z_floor - self.cfg.below_floor, self.z_start - self.cfg.speed * (now - self.t0))

    def outcome(self, now, p) -> str | None:
        """'landed' after ground_hold near the floor point, 'landed (timeout)' if that never happens, else None."""
        self.ground_since = (self.ground_since or now) if p[2] < self.z_floor + self.cfg.ground_band else None
        if self.ground_since and now - self.ground_since > self.cfg.ground_hold:
            return "landed"
        if now - self.t0 > (self.z_start - self.z_floor) / self.cfg.speed + self.cfg.extra_time:
            return "landed (timeout)"
        return None
