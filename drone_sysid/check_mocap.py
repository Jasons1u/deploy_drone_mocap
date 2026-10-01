"""Pre-flight mocap gate: is the pose topic live, fast enough, and actually moving (not a frozen rigid body)?"""

import argparse
import sys
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pose_topic")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = rclpy.create_node("check_mocap")
    stamps, poses = [], []
    node.create_subscription(PoseStamped, args.pose_topic,
                             lambda m: (stamps.append(time.monotonic()),
                                        poses.append((m.pose.position.x, m.pose.position.y, m.pose.position.z))), 10)
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()

    if len(poses) < 2:
        print(f"NO DATA on {args.pose_topic} -- check the topic name and the mocap bridge")
        sys.exit(1)
    p = np.array(poses)
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    moves = np.linalg.norm(np.diff(p, axis=0), axis=1)
    frozen = float((moves < 1e-5).mean())
    print(f"{args.pose_topic}: {rate:.0f} Hz, position {np.round(p[-1], 3).tolist()}, "
          f"jitter std {p.std(0).max() * 1e3:.2f} mm, identical frames {frozen * 100:.0f}%")
    if rate < 100 or frozen > 0.5:
        print("NOT LIVE -- low rate or frozen pose (Motive streams the last pose after losing the body)")
        sys.exit(1)
    print("LIVE")


if __name__ == "__main__":
    main()
