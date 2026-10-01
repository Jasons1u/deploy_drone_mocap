"""ROS 2 node: OptiTrack (NatNet) -> ROS odometry + PX4 external vision.

Publishes, for every valid Motive frame of one rigid body:
  /mocap/odom                       nav_msgs/Odometry   ENU world "mocap", FLU body, twist in
                                                        the body frame (REP-105). hover_node's
                                                        state source when state_source=mocap.
  /fmu/in/vehicle_visual_odometry   px4_msgs/VehicleOdometry  NED / FRD, position + attitude only
                                                        (velocity NaN: EKF2 derives it). This is
                                                        what gives PX4 an indoor position — needed
                                                        for state_source=px4 AND for the pilot's
                                                        POSITION-mode takeover.

Timestamps are this host's ROS clock at frame arrival. PX4's uXRCE-DDS client shifts incoming
timestamps by the agent time offset, so run this node on the machine that runs the
MicroXRCEAgent, or keep the clocks synced (chrony). Motive's own pipeline latency is not in the
stamp; account for it with EKF2_EV_DELAY.

Run it on a machine wired to the Motive network (multicast over Wi-Fi is unreliable).
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from mocap_natnet import NatNetClient
from px4_msgs.msg import VehicleOdometry

from . import frames

STATUS_PERIOD_S = 5.0
FLOOR_TOL_M = -0.05   # a tracked body below this is almost surely a frame misconfiguration


class MocapBridge(Node):
    def __init__(self):
        super().__init__("mocap_bridge")
        self.declare_parameter("server_ip", "")
        self.declare_parameter("local_ip", "0.0.0.0")
        self.declare_parameter("rigid_body_id", 1)
        self.declare_parameter("up_axis", "z")
        self.declare_parameter("natnet_version", "")          # "" = ask Motive
        self.declare_parameter("odom_topic", "/mocap/odom")
        self.declare_parameter("frame_id", "mocap")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_px4", True)
        self.declare_parameter("px4_topic", "/fmu/in/vehicle_visual_odometry")
        self.declare_parameter("position_std_m", 0.005)      # reported to EKF2 (EKF2_EV_NOISE_MD=0)
        self.declare_parameter("orientation_std_rad", 0.02)

        prm = lambda name: self.get_parameter(name).value  # noqa: E731
        if not prm("server_ip"):
            raise SystemExit("mocap_bridge: parameter server_ip is required (IP of the Motive PC)")
        self._body_id = int(prm("rigid_body_id"))
        self._frame_id = prm("frame_id")
        self._child_frame_id = prm("child_frame_id")
        self._pos_var = [float(prm("position_std_m")) ** 2] * 3
        self._ori_var = [float(prm("orientation_std_rad")) ** 2] * 3
        self._published = 0
        self._last_frame = None

        self._pub_odom = self.create_publisher(Odometry, prm("odom_topic"), qos_profile_sensor_data)
        self._pub_px4 = None
        if prm("publish_px4"):
            self._pub_px4 = self.create_publisher(VehicleOdometry, prm("px4_topic"),
                                                  qos_profile_sensor_data)

        self._client = NatNetClient(prm("server_ip"), local_ip=prm("local_ip"),
                                    up_axis=prm("up_axis"),
                                    natnet_version=prm("natnet_version") or None,
                                    on_pose=self._on_pose)
        self._client.start()
        self.get_logger().info(
            f"NatNet {'.'.join(map(str, self._client.natnet_version))} from {prm('server_ip')}, "
            f"rigid body {self._body_id}, up axis {prm('up_axis')} -> {prm('odom_topic')}"
            + (f" + {prm('px4_topic')}" if self._pub_px4 else ""))
        self.create_timer(STATUS_PERIOD_S, self._report)

    def destroy_node(self):
        self._client.stop()
        super().destroy_node()

    def _on_pose(self, body_id: int, pose) -> None:
        """Runs on the NatNet receive thread (rclpy publishers are thread-safe)."""
        if body_id != self._body_id:
            return
        age_ns = int((time.monotonic() - pose.stamp) * 1e9)
        stamp = self.get_clock().now() - Duration(nanoseconds=age_ns)
        pos = np.asarray(pose.position)
        quat = np.asarray(pose.quat)
        try:
            self._pub_odom.publish(self._odometry(stamp, pos, quat, np.asarray(pose.velocity)))
            if self._pub_px4 is not None:
                self._pub_px4.publish(self._px4_odometry(stamp.nanoseconds // 1000, pos, quat))
        except Exception:
            # Ctrl+C shuts the rclpy context down before destroy_node() stops this thread.
            if self.context.ok():
                raise
            return
        self._published += 1

    def _odometry(self, stamp, pos, quat, vel_world) -> Odometry:
        msg = Odometry()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self._frame_id
        msg.child_frame_id = self._child_frame_id
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        p.x, p.y, p.z = (float(v) for v in pos)
        o.w, o.x, o.y, o.z = (float(v) for v in quat)
        v = msg.twist.twist.linear
        v.x, v.y, v.z = (float(c) for c in frames.quat_apply_inverse(quat, vel_world))
        cov = np.zeros((6, 6))
        cov[:3, :3] = np.diag(self._pos_var)
        cov[3:, 3:] = np.diag(self._ori_var)
        msg.pose.covariance = cov.ravel().tolist()
        return msg

    def _px4_odometry(self, stamp_us: int, pos, quat) -> VehicleOdometry:
        msg = VehicleOdometry()
        msg.timestamp = stamp_us
        msg.timestamp_sample = stamp_us
        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        msg.position = frames.enu_to_ned(pos).astype(np.float32)
        msg.q = frames.flu_enu_to_frd_ned(quat).astype(np.float32)
        msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_UNKNOWN
        msg.velocity = np.full(3, np.nan, dtype=np.float32)
        msg.angular_velocity = np.full(3, np.nan, dtype=np.float32)
        msg.position_variance = np.asarray(self._pos_var, dtype=np.float32)
        msg.orientation_variance = np.asarray(self._ori_var, dtype=np.float32)
        msg.velocity_variance = np.full(3, np.nan, dtype=np.float32)
        msg.reset_counter = 0
        msg.quality = 100
        return msg

    def _report(self) -> None:
        rate = self._published / STATUS_PERIOD_S
        self._published = 0
        pose = self._client.pose(self._body_id)
        age = self._client.age(self._body_id)
        if pose is None:
            self.get_logger().warning(
                f"rigid body {self._body_id} never seen (frames: {self._client.frames_received}, "
                f"bodies: {self._client.body_ids() or 'none'})")
        elif age > 0.1:
            self.get_logger().warning(f"rigid body {self._body_id} not tracked for {age:.1f} s")
        elif pose.position[2] < FLOOR_TOL_M:
            self.get_logger().warning(
                f"rigid body {self._body_id} is at z = {pose.position[2]:+.2f} m, below the floor "
                f"— wrong up_axis, or the Motive ground plane is not on the floor")
        else:
            self.get_logger().info(f"{rate:.0f} Hz, body {self._body_id} tracked",
                                   throttle_duration_sec=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = MocapBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
