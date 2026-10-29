"""Flight limits: E-STOP conditions checked every tick, and the soft arena boundary with its braking point."""

from __future__ import annotations

import math

import numpy as np

from cfdeploy.geometry import tilt_deg
from cfdeploy.safe_cf_obs import ARENA_HEIGHT, ARENA_RADIUS


# #### // E-STOP // ####
def estop_reason(cfg, kb, pad, body, age, tel, limit_problem, tilt_armed, mocap_at=None, mocap_armed=True) -> str | None:
    """Deadman, kill key, telemetry loss, position limit, flip; with mocap_armed, mocap_reason too (mocap_at(t) aligns the estimate)."""
    deadman, s = cfg.gamepad.deadman, cfg.safety
    if not pad.trigger_held(deadman.axis, deadman.hold_threshold):
        a = pad.axis(deadman.axis)
        return f"deadman (LT) released (reads {a if a is None else round(a, 2)})" if pad.alive else "gamepad disconnected"
    if kb.kill:
        return "SPACE"
    if body is None:
        return "drone never tracked"
    if tel.sample_age("pos") > s.telemetry_timeout:
        return f"no telemetry for {tel.sample_age('pos'):.1f} s"
    problem = limit_problem(body.position)
    if problem:
        return problem
    if tilt_armed and age <= s.mocap_timeout and tilt_deg(body.quat) > s.max_tilt:
        return f"tilt {tilt_deg(body.quat):.0f} deg"
    return mocap_reason(cfg, body, age, tel, mocap_at) if mocap_armed else None


def mocap_reason(cfg, body, age, tel, mocap_at=None) -> str | None:
    """Motive or the drone without a pose past their timeouts, or the onboard estimate off the mocap pose."""
    s = cfg.safety
    if age > s.mocap_timeout:
        return f"mocap lost {age:.2f} s"
    onboard_age = tel.get("mocapEst.ageMs", 65535)
    if onboard_age > s.onboard_mocap_timeout_ms:
        return f"drone has no mocap for {onboard_age} ms"
    est = [tel.get(f"stateEstimate.{a}") for a in "xyz"]
    t_sample = tel.sample_time("pos")
    ref = mocap_at(t_sample) if mocap_at is not None and t_sample is not None else None
    ref = body.position if ref is None else ref   # mocap when the drone sampled its estimate: a late downlink is not an error
    if math.dist(ref, est) > s.max_estimate_error:
        return f"onboard estimate {math.dist(ref, est) * 100:.0f} cm from mocap"
    return None


def left_arena(p, limits) -> bool:
    """Past the hard arena limits (E-STOP)."""
    return math.hypot(p[0], p[1]) > ARENA_RADIUS - limits.hard_wall_margin or p[2] > ARENA_HEIGHT - limits.hard_top_margin


def arena_limit(limits):
    """limit_problem for estop_reason: the arena walls and ceiling."""
    return lambda p: f"left the arena at ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})" if left_arena(p, limits) else None


def box_limit(p0, box, ceiling):
    """limit_problem for estop_reason: |x|, |y| from the takeoff point and a ceiling."""
    x0, y0, z0 = p0
    return lambda p: (f"left the box at ({p[0] - x0:+.2f}, {p[1] - y0:+.2f}, {p[2] - z0:+.2f})"
                      if abs(p[0] - x0) > box or abs(p[1] - y0) > box or p[2] > ceiling else None)


# #### // Soft arena boundary (manual sticks stop pushing out; the policy hands over) // ####
def keep_inside(v, p, limits) -> np.ndarray:
    """Manual mode: drop velocity components that push further past the soft arena boundary."""
    v = v.copy()
    r = math.hypot(p[0], p[1])
    if r > ARENA_RADIUS - limits.soft_wall and r > 1e-6:
        out = np.array([p[0] / r, p[1] / r])
        radial = v[:2] @ out
        if radial > 0:
            v[:2] -= radial * out
    if (p[2] > limits.soft_top and v[2] > 0) or (p[2] < limits.soft_bottom and v[2] < 0):
        v[2] = 0.0
    return v


def outside_soft(p, limits) -> str | None:
    if math.hypot(p[0], p[1]) > ARENA_RADIUS - limits.soft_wall:
        return "near the arena wall"
    if p[2] > limits.soft_top:
        return "near the ceiling"
    if p[2] < limits.soft_bottom:
        return "too low"
    return None


def braking_point(p, v, limits) -> np.ndarray:
    """Where the drone stops if it brakes now: outward-radial and vertical speed, v*delay + v^2/(2*a)."""
    q = np.array(p, float)
    r = math.hypot(q[0], q[1])
    stop = lambda u: u * limits.brake_delay + u * u / (2.0 * limits.brake_acc)
    if r > 1e-6:
        v_out = max(0.0, (v[0] * q[0] + v[1] * q[1]) / r)
        q[:2] *= (r + stop(v_out)) / r
    q[2] += math.copysign(stop(abs(v[2])), v[2])
    return q


def inside_soft(p, limits) -> tuple:
    """Nearest point hold_margin inside the soft arena boundary: the hold point after a boundary exit."""
    q = np.array(p, float)
    margin = limits.hold_margin
    r_max = ARENA_RADIUS - limits.soft_wall - margin
    r = math.hypot(q[0], q[1])
    if r > r_max:
        q[:2] *= r_max / r
    q[2] = min(max(q[2], limits.soft_bottom + margin), limits.soft_top - margin)
    return tuple(q)


# #### // Warnings that switch on and off cleanly at the edge // ####
class Hysteresis:
    """On/off state: turns on when its on-condition holds, back off only when its (stricter) off-condition holds."""

    def __init__(self):
        self.on = False

    def update(self, past_on: bool, past_off: bool) -> bool:
        """True when the state just changed."""
        new = (not past_off) if self.on else past_on
        changed, self.on = new != self.on, new
        return changed
