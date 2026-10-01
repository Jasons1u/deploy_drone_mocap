"""ROS 2 node: fly the RL hover policy through PX4 offboard direct-actuator control.

Nothing here arms. The pilot arms and switches to OFFBOARD from RC/QGC; until PX4 reports
ARMED + OFFBOARD the node only publishes the offboard heartbeat and never touches the motors.
On the first commanded tick it captures the drone's own position AND yaw as the hover target
(handover), so nothing jumps at engage. Switching out of OFFBOARD hands control back to PX4;
re-engaging captures a fresh target.

State (param state_source):
  px4    pos / vel / attitude / gyro from /fmu/out/vehicle_odometry — EKF2 fusing the mocap
         that mocap_bridge sends as vehicle_visual_odometry.
  mocap  pos / vel / attitude from /mocap/odom (mocap_bridge), gyro from vehicle_odometry.
If any input in use is older than stale_timeout_s while engaged, all motors go NaN (PX4 treats
that as disarmed) — the vehicle drops rather than flying on stale state.

Moving the target while engaged (RELATIVE only — a big absolute jump makes the policy lunge):
  ros2 topic pub --once /hover/nudge geometry_msgs/msg/Vector3 '{z: 0.2}'   # ENU metres
  ros2 topic pub --once /hover/nudge_yaw std_msgs/msg/Float32 '{data: 0.3}' # rad, CCW
Nudges larger than max_nudge_m / max_nudge_yaw_rad are rejected.

Telemetry: /hover/debug (HoverDebug, every tick), /hover/event (HoverEvent, latched).
Frames: see frames.py. Target yaw is ENU (0 = mocap +X, CCW positive).
"""

from __future__ import annotations

import hashlib
import math
import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from std_msgs.msg import Float32

from px4_mocap_hover_msgs.msg import HoverDebug, HoverEvent
from px4_msgs.msg import ActuatorMotors, OffboardControlMode, VehicleOdometry, VehicleStatus

from . import frames
from .hover_policy import DroneState, HoverPolicy

# PX4 state streams: keep only the freshest sample.
QOS_PX4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
QOS_EVENT = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST, depth=50)

# Real drone bridges versioned names, SITL the plain one; whichever publishes drives the gate.
VEHICLE_STATUS_TOPICS = ("/fmu/out/vehicle_status_v2", "/fmu/out/vehicle_status_v1",
                         "/fmu/out/vehicle_status")
STATE_SOURCES = ("px4", "mocap")
TRAIN_Z_RANGE = (0.5, 3.0)   # target heights the policy was trained on [m above floor]
UNKNOWN_U8 = 255
NAN3 = np.full(3, np.nan)


class HoverNode(Node):
    def __init__(self):
        super().__init__("hover_node")
        self.declare_parameter("policy_onnx", "new_hover.onnx")   # relative -> share/policy/
        self.declare_parameter("providers", ["CPUExecutionProvider"])
        self.declare_parameter("control_hz", 100.0)
        self.declare_parameter("state_source", "px4")
        self.declare_parameter("mocap_odom_topic", "/mocap/odom")
        self.declare_parameter("require_offboard", True)
        self.declare_parameter("stale_timeout_s", 0.1)
        self.declare_parameter("frame_check_tol_m", 0.1)
        self.declare_parameter("max_nudge_m", 0.5)
        self.declare_parameter("max_nudge_yaw_rad", 0.5)

        prm = lambda name: self.get_parameter(name).value  # noqa: E731
        self._state_source = prm("state_source")
        if self._state_source not in STATE_SOURCES:
            raise SystemExit(f"state_source must be one of {STATE_SOURCES}, got {self._state_source!r}")
        self._require_offboard = bool(prm("require_offboard"))
        self._stale_timeout = float(prm("stale_timeout_s"))
        self._frame_tol = float(prm("frame_check_tol_m"))
        self._max_nudge = float(prm("max_nudge_m"))
        self._max_nudge_yaw = float(prm("max_nudge_yaw_rad"))
        hz = float(prm("control_hz"))

        policy_path = _resolve_policy(prm("policy_onnx"))
        self._policy = HoverPolicy(policy_path, providers=prm("providers"))
        self.get_logger().info(
            f"policy {policy_path} (sha256 {_sha256(policy_path)[:12]}) obs_dim "
            f"{self._policy.obs_dim}, state_dim {self._policy.state_dim}, providers "
            f"{self._policy.session.get_providers()}")

        # latest inputs (ENU / FLU) and their receive times (monotonic)
        self._px4 = None      # DroneState from vehicle_odometry
        self._px4_t = -math.inf
        self._mocap = None    # DroneState from /mocap/odom (ang_vel unused)
        self._mocap_t = -math.inf
        self._nav_state = None
        self._arming_state = None

        self._offboard = False
        self._engaged = False
        self._stale = False
        self._target = np.zeros(3)
        self._target_yaw = 0.0
        self._prev_tick = None

        self._pub_motors = self.create_publisher(ActuatorMotors, "/fmu/in/actuator_motors", QOS_PX4)
        self._pub_offboard = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", QOS_PX4)
        self._pub_debug = self.create_publisher(HoverDebug, "/hover/debug", 10)
        self._pub_event = self.create_publisher(HoverEvent, "/hover/event", QOS_EVENT)

        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_px4_odom,
                                 QOS_PX4)
        self.create_subscription(Odometry, prm("mocap_odom_topic"), self._on_mocap_odom,
                                 qos_profile_sensor_data)
        for topic in VEHICLE_STATUS_TOPICS:
            self.create_subscription(VehicleStatus, topic, self._on_status, QOS_PX4)
        self.create_subscription(Vector3, "/hover/nudge", self._on_nudge, 10)
        self.create_subscription(Float32, "/hover/nudge_yaw", self._on_nudge_yaw, 10)

        if not self._require_offboard:
            self.get_logger().warning("require_offboard=false: the policy drives the motors whenever "
                                      "state is fresh, regardless of PX4 mode (SITL only!)")
        self.create_timer(1.0 / hz, self._tick)
        self._event(HoverEvent.NODE_STARTED, f"{hz:.0f} Hz, state_source={self._state_source}")
        self.get_logger().info(f"hover_node @ {hz:.0f} Hz, state from {self._state_source}; "
                               f"waiting for ARMED + OFFBOARD")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _on_px4_odom(self, msg: VehicleOdometry) -> None:
        if (msg.pose_frame != VehicleOdometry.POSE_FRAME_NED
                or msg.velocity_frame != VehicleOdometry.VELOCITY_FRAME_NED):
            self.get_logger().warning(
                f"vehicle_odometry frames pose={msg.pose_frame} vel={msg.velocity_frame} "
                f"(need NED=1) — ignored, state will go stale", throttle_duration_sec=5.0)
            return
        raw = np.concatenate([msg.position, msg.velocity, msg.q, msg.angular_velocity])
        if not np.all(np.isfinite(raw)):
            self.get_logger().warning("vehicle_odometry has NaN fields (EKF not ready?) — ignored",
                                      throttle_duration_sec=5.0)
            return
        self._px4 = DroneState(frames.ned_to_enu(msg.position), frames.ned_to_enu(msg.velocity),
                               frames.frd_ned_to_flu_enu(msg.q), frames.frd_to_flu(msg.angular_velocity))
        self._px4_t = time.monotonic()

    def _on_mocap_odom(self, msg: Odometry) -> None:
        p, o, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        quat = np.array([o.w, o.x, o.y, o.z])
        vel_world = frames.quat_apply(quat, [v.x, v.y, v.z])   # twist is in the body frame
        self._mocap = DroneState(np.array([p.x, p.y, p.z]), vel_world, quat, np.zeros(3))
        self._mocap_t = time.monotonic()

    def _on_status(self, msg: VehicleStatus) -> None:
        self._nav_state = msg.nav_state
        self._arming_state = msg.arming_state

    def _state(self, now: float):
        """The DroneState fed to the policy, and the age of its oldest input."""
        px4_age = now - self._px4_t
        if self._state_source == "px4":
            return self._px4, px4_age
        mocap_age = now - self._mocap_t
        if self._px4 is None or self._mocap is None:
            return None, max(px4_age, mocap_age)
        m = self._mocap
        return DroneState(m.pos_enu, m.vel_enu, m.quat_flu_enu, self._px4.ang_vel_flu), \
            max(px4_age, mocap_age)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = time.monotonic()
        dbg = self._begin_debug(now)
        try:
            self._control(now, dbg)
        finally:
            dbg.target_enu = _f32(self._target)   # after _control: engage may have just set it
            dbg.target_yaw = float(self._target_yaw)
            self._pub_debug.publish(dbg)

    def _control(self, now: float, dbg: HoverDebug) -> None:
        self._publish_heartbeat()
        state, age = self._state(now)
        dbg.state_age_ms = float(min(age, 1e6) * 1e3)
        if state is not None:
            dbg.pos_enu, dbg.vel_enu = _f32(state.pos_enu), _f32(state.vel_enu)
            dbg.quat_flu_enu, dbg.ang_vel_flu = _f32(state.quat_flu_enu), _f32(state.ang_vel_flu)

        if not self._update_offboard(dbg):
            self._check_frames(now)
            return

        if age > self._stale_timeout or state is None:
            dbg.status = HoverDebug.STATUS_STALE
            self._publish_disarm()
            if not self._stale:
                self._stale = True
                self._event(HoverEvent.STALE, f"state {age * 1e3:.0f} ms old — motors NaN")
            self.get_logger().error(f"state stale ({age * 1e3:.0f} ms) — motors NaN",
                                    throttle_duration_sec=1.0)
            return
        if self._stale:
            self._stale = False
            self._event(HoverEvent.STALE_RECOVERED, "state fresh")

        if not self._engaged:
            self._engage(state)

        obs = self._policy.build_obs(state, self._target, self._target_yaw)
        out = self._policy.act(obs)
        dbg.obs = obs.tolist()
        dbg.raw = _f32(out.raw)
        dbg.inf_ms = float(out.inf_ms)
        if not np.all(np.isfinite(out.raw)):
            dbg.status = HoverDebug.STATUS_NONFINITE
            self._publish_disarm()
            self._event(HoverEvent.NONFINITE_OUTPUT, f"policy output {out.raw.tolist()}")
            self.get_logger().error(f"policy returned {out.raw.tolist()} — motors NaN",
                                    throttle_duration_sec=1.0)
            return
        self._publish_motors(out.motor_cmd)
        dbg.motor_cmd = _f32(out.motor_cmd)
        dbg.status = HoverDebug.STATUS_OK
        dbg.commanded = True

    def _update_offboard(self, dbg: HoverDebug) -> bool:
        """Track ARMED + OFFBOARD. Returns True when the policy may drive the motors."""
        active = (not self._require_offboard) or (
            self._nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and self._arming_state == VehicleStatus.ARMING_STATE_ARMED)
        if active and not self._offboard:
            self._event(HoverEvent.OFFBOARD_ACQUIRED, "ARMED + OFFBOARD")
        elif not active and self._offboard:
            self._engaged = False
            self._event(HoverEvent.OFFBOARD_RELEASED, "pilot has control")
            self.get_logger().warning("OFFBOARD released — pilot has control, policy idle")
        self._offboard = active
        if not active:
            dbg.status = (HoverDebug.STATUS_NO_VEHICLE_STATUS if self._nav_state is None
                          else HoverDebug.STATUS_NOT_OFFBOARD)
            if self._nav_state is None:
                self.get_logger().warning("no VehicleStatus yet — is the uXRCE-DDS agent running?",
                                          throttle_duration_sec=5.0)
        return active

    def _engage(self, state: DroneState) -> None:
        self._policy.reset()
        self._target = np.array(state.pos_enu, dtype=np.float64)
        self._target_yaw = frames.yaw_from_quat(state.quat_flu_enu)
        self._engaged = True
        self._warn_if_out_of_training(self._target)
        self._event(HoverEvent.ENGAGE, "target captured at handover pose")
        self.get_logger().info(f"ENGAGED: holding {np.round(self._target, 3).tolist()} "
                               f"yaw {math.degrees(self._target_yaw):+.0f} deg")

    def _check_frames(self, now: float) -> None:
        """Before engage: warn if PX4's local frame disagrees with mocap (EKF2 not fusing vision?)."""
        if self._px4 is None or self._mocap is None:
            return
        if max(now - self._px4_t, now - self._mocap_t) > self._stale_timeout:
            return
        err = float(np.linalg.norm(self._px4.pos_enu - self._mocap.pos_enu))
        if err > self._frame_tol:
            self.get_logger().warning(
                f"PX4 position differs from mocap by {err:.2f} m — check EKF2 vision fusion "
                f"(EKF2_EV_CTRL, EKF2_HGT_REF) before engaging", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # Nudges
    # ------------------------------------------------------------------

    def _on_nudge(self, msg: Vector3) -> None:
        delta = np.array([msg.x, msg.y, msg.z])
        if not self._nudge_allowed(float(np.linalg.norm(delta)), self._max_nudge, "m"):
            return
        self._target = self._target + delta
        self._warn_if_out_of_training(self._target)
        self._setpoint_changed(f"nudge {delta.tolist()}")

    def _on_nudge_yaw(self, msg: Float32) -> None:
        if not self._nudge_allowed(abs(msg.data), self._max_nudge_yaw, "rad"):
            return
        self._target_yaw = frames.wrap_pi(self._target_yaw + msg.data)
        self._setpoint_changed(f"yaw nudge {msg.data:+.3f} rad")

    def _nudge_allowed(self, size: float, limit: float, unit: str) -> bool:
        if not self._engaged:
            self.get_logger().warning("nudge ignored: not engaged (target is captured at handover)")
            return False
        if not size <= limit:
            self.get_logger().error(f"nudge of {size:.2f} {unit} rejected (limit {limit} {unit})")
            return False
        return True

    def _setpoint_changed(self, reason: str) -> None:
        self._event(HoverEvent.SETPOINT_CHANGED, reason)
        self.get_logger().info(f"{reason} -> target {np.round(self._target, 3).tolist()} "
                               f"yaw {math.degrees(self._target_yaw):+.0f} deg")

    def _warn_if_out_of_training(self, target) -> None:
        lo, hi = TRAIN_Z_RANGE
        if not lo <= target[2] <= hi:
            self.get_logger().warning(
                f"target z {target[2]:.2f} m is outside the trained range [{lo}, {hi}] m")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _begin_debug(self, now: float) -> HoverDebug:
        d = HoverDebug()
        d.header.stamp = self.get_clock().now().to_msg()
        d.header.frame_id = "mocap"
        d.loop_dt_ms = 0.0 if self._prev_tick is None else float((now - self._prev_tick) * 1e3)
        self._prev_tick = now
        d.nav_state = UNKNOWN_U8 if self._nav_state is None else int(self._nav_state)
        d.arming_state = UNKNOWN_U8 if self._arming_state is None else int(self._arming_state)
        d.mocap_pos_enu = _f32(self._mocap.pos_enu if self._mocap is not None else NAN3)
        return d

    def _event(self, event: int, reason: str) -> None:
        ev = HoverEvent()
        ev.header.stamp = self.get_clock().now().to_msg()
        ev.event = event
        ev.reason = reason
        state = self._px4 if self._state_source == "px4" else self._mocap
        ev.pos_enu = _f32(state.pos_enu if state is not None else NAN3)
        ev.target_enu = _f32(self._target)
        ev.target_yaw = float(self._target_yaw)
        self._pub_event.publish(ev)

    def _us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def _publish_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self._us()
        msg.direct_actuator = True
        self._pub_offboard.publish(msg)

    def _publish_motors(self, cmd: np.ndarray) -> None:
        msg = ActuatorMotors()
        msg.timestamp = self._us()
        controls = np.full(12, np.nan, dtype=np.float32)   # channels 4..11 unused -> NaN
        controls[:4] = cmd
        msg.control = controls
        self._pub_motors.publish(msg)

    def _publish_disarm(self) -> None:
        msg = ActuatorMotors()
        msg.timestamp = self._us()
        msg.control = np.full(12, np.nan, dtype=np.float32)
        self._pub_motors.publish(msg)


def _f32(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def _resolve_policy(name: str) -> str:
    if os.path.isabs(name):
        return name
    return os.path.join(get_package_share_directory("px4_mocap_hover"), "policy", name)


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(args=None):
    rclpy.init(args=args)
    node = HoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
