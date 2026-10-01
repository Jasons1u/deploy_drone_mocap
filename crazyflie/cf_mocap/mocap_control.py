"""Host-side position controller: OptiTrack state in, attitude + thrust setpoints out.

The default flight mode (without --kalman). Position, velocity and heading come straight from
mocap; the Crazyflie only stabilises attitude from its own IMU (complementary estimator + onboard
attitude PID). The onboard Kalman filter is not in the loop at all.

Cascade, in the mocap world frame (Z up):
    v_cmd = v_ff + Kp (p_ref - p)                    clamped to max_vel_xy / max_vel_z
    a_cmd = Kv (v_cmd - v) + Ki ∫(v_cmd - v) dt      the integral absorbs hover-thrust and tilt trim
    thrust = hover_thrust (g + a_z) / g / (cos roll cos pitch)
    a_xy, rotated into the heading frame (mocap yaw), becomes tilt:
        pitch = atan2(a_forward, g + a_z)      > 0 = nose down  -> accelerates forward
        roll  = atan2(-a_left, g + a_z)        > 0 = right down -> accelerates right
    yaw rate = Kp_yaw * wrap(yaw_ref - yaw)          CCW positive

Horizontal integration only runs once airborne, so ground friction cannot wind it up into a tilt
at liftoff. The vertical integral runs from the start: that is how it finds the real hover thrust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mocap_natnet.frames import yaw_from_quat

from .config import HostControlConfig

G = 9.81
AIRBORNE_M = 0.05   # above the takeoff height, the horizontal integrators start


@dataclass(frozen=True)
class AttitudeCommand:
    roll_deg: float        # right-handed
    pitch_deg: float       # right-handed (> 0 nose down)
    yawrate_deg_s: float   # counter-clockwise positive
    thrust: int            # PWM


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class MocapPositionController:
    def __init__(self, cfg: HostControlConfig, ground_z: float):
        self._cfg = cfg
        self._ground_z = ground_z
        self._integral = [0.0, 0.0, 0.0]   # [m/s^2], world frame
        self._prev_t = None

    def step(self, t: float, sp, pose) -> AttitudeCommand:
        """sp: flight_plan.Setpoint; pose: mocap_natnet.Pose (position, velocity, quat)."""
        c = self._cfg
        dt = 0.0 if self._prev_t is None else min(t - self._prev_t, 0.05)
        self._prev_t = t

        ref = (sp.x, sp.y, sp.z)
        v_ff = (0.0, 0.0, sp.vz)
        kp = (c.kp_xy, c.kp_xy, c.kp_z)
        v_cmd = [v_ff[i] + kp[i] * (ref[i] - pose.position[i]) for i in range(3)]
        speed_xy = math.hypot(v_cmd[0], v_cmd[1])
        if speed_xy > c.max_vel_xy:
            v_cmd[0] *= c.max_vel_xy / speed_xy
            v_cmd[1] *= c.max_vel_xy / speed_xy
        v_cmd[2] = _clamp(v_cmd[2], c.max_vel_z)

        v_err = [v_cmd[i] - pose.velocity[i] for i in range(3)]
        airborne = pose.position[2] > self._ground_z + AIRBORNE_M
        ki = (c.ki_xy if airborne else 0.0, c.ki_xy if airborne else 0.0, c.ki_z)
        limits = (c.i_limit_xy, c.i_limit_xy, c.i_limit_z)
        for i in range(3):
            self._integral[i] = _clamp(self._integral[i] + ki[i] * v_err[i] * dt, limits[i])
        kv = (c.kv_xy, c.kv_xy, c.kv_z)
        a = [kv[i] * v_err[i] + self._integral[i] for i in range(3)]

        a[2] = _clamp(a[2], c.max_accel_z)
        lift = G + a[2]                                    # > 0 since max_accel_z < G
        max_h = lift * math.tan(math.radians(c.max_tilt_deg))
        a_h = math.hypot(a[0], a[1])
        if a_h > max_h:
            a[0] *= max_h / a_h
            a[1] *= max_h / a_h

        yaw = yaw_from_quat(pose.quat)
        a_fwd = math.cos(yaw) * a[0] + math.sin(yaw) * a[1]
        a_left = -math.sin(yaw) * a[0] + math.cos(yaw) * a[1]
        pitch = math.atan2(a_fwd, lift)
        roll = math.atan2(-a_left, lift)

        thrust = c.hover_thrust * lift / G / (math.cos(roll) * math.cos(pitch))
        thrust = int(round(max(c.thrust_min, min(c.thrust_max, thrust))))

        yaw_err = _wrap_deg(sp.yaw_deg - math.degrees(yaw))
        yawrate = _clamp(c.kp_yaw * yaw_err, c.max_yawrate_deg_s)
        return AttitudeCommand(math.degrees(roll), math.degrees(pitch), yawrate, thrust)
