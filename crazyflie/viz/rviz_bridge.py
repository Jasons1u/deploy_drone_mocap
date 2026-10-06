#!/usr/bin/env python3
"""ROS 2 bridge for fly_policy.py --viz / sim_policy.py: state over UDP -> RViz markers + TF; draggable target (+ sim obstacles) back.

Runs on the system python with ROS 2 Humble (crazyflie/viz/run_rviz.sh); fly_policy itself stays ROS-free.
"""

import argparse
import json
import math
import socket
import time
from collections import deque

import rclpy
from geometry_msgs.msg import Point, Pose, TransformStamped
from interactive_markers import InteractiveMarkerServer
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker, MarkerArray

FRAME = "world"
S2 = math.sqrt(0.5)


def rgba(r, g, b, a):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def pt(xyz):
    p = Point()
    p.x, p.y, p.z = (float(v) for v in xyz)
    return p


def circle(r, z, n=48, centre=(0.0, 0.0)):
    return [pt((centre[0] + r * math.cos(2 * math.pi * k / n), centre[1] + r * math.sin(2 * math.pi * k / n), z)) for k in range(n + 1)]


def wire_cylinder(r, z0, z1, n=48, posts=8):
    """LINE_LIST points: rim circles + vertical posts (a solid cylinder would swallow the mouse clicks on handles inside it)."""
    pts = []
    for z in (z0, z1):
        c = circle(r, z, n)
        for a, b in zip(c[:-1], c[1:]):
            pts += [a, b]
    for k in range(posts):
        a = 2 * math.pi * k / posts
        pts += [pt((r * math.cos(a), r * math.sin(a), z0)), pt((r * math.cos(a), r * math.sin(a), z1))]
    return pts


class Bridge(Node):
    def __init__(self, state_port: int, cmd_port: int):
        super().__init__("cf_viz")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", state_port))
        self.sock.setblocking(False)
        self.cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_dst = ("127.0.0.1", cmd_port)
        self.pub = self.create_publisher(MarkerArray, "/cf_viz/markers", 10)
        self.tf = TransformBroadcaster(self)
        self.server = InteractiveMarkerServer(self, "cf_target")
        self.state, self.last_rx = None, 0.0
        self.trail = deque(maxlen=900)
        self.dragging, self.last_release, self.shown = set(), {}, {}
        self.create_timer(1 / 30, self.tick)
        self.get_logger().info(f"listening for fly_policy state on udp:{state_port}, sending targets to udp:{cmd_port}")

    # --- draggable handles: the target, and sim obstacles ---
    def insert_handle(self, name, xyz, description, color, size=0.14, scale=0.45):
        im = InteractiveMarker()
        im.header.frame_id = FRAME
        im.name = name
        im.description = description
        im.scale = scale
        im.pose.position = pt(xyz)
        im.pose.orientation.w = 1.0
        ball = Marker()
        ball.type = Marker.SPHERE
        ball.scale.x = ball.scale.y = ball.scale.z = size
        ball.color = color
        vis = InteractiveMarkerControl()
        vis.name = "grab"
        vis.always_visible = True
        vis.orientation.w, vis.orientation.y = S2, S2
        vis.interaction_mode = InteractiveMarkerControl.MOVE_PLANE   # drag the ball itself: horizontal, arrows for height
        vis.markers.append(ball)
        im.controls.append(vis)
        for ctrl, q in (("move_x", (S2, S2, 0.0, 0.0)), ("move_y", (S2, 0.0, 0.0, S2)), ("move_z", (S2, 0.0, S2, 0.0))):
            c = InteractiveMarkerControl()
            c.name = ctrl
            c.orientation.w, c.orientation.x, c.orientation.y, c.orientation.z = q
            c.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            im.controls.append(c)
        plane = InteractiveMarkerControl()
        plane.name = "move_xy"
        plane.orientation.w, plane.orientation.y = S2, S2
        plane.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        im.controls.append(plane)
        self.server.insert(im, feedback_callback=self.on_feedback)
        self.server.applyChanges()
        self.shown[name] = list(xyz)

    def on_feedback(self, fb):
        name, p = fb.marker_name, fb.pose.position
        if name in self.shown and fb.event_type in (InteractiveMarkerFeedback.POSE_UPDATE, InteractiveMarkerFeedback.MOUSE_UP):
            self.shown[name] = [p.x, p.y, p.z]   # where the handle is now, so a refused request snaps it back
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self.dragging.add(name)
        elif fb.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.dragging.discard(name)
            self.last_release[name] = time.monotonic()
            if name == "target":
                self.send({"target": [p.x, p.y, p.z]})
                self.get_logger().info(f"target request ({p.x:+.2f}, {p.y:+.2f}, {p.z:+.2f})")
        if name.startswith("obstacle_") and fb.event_type in (InteractiveMarkerFeedback.POSE_UPDATE, InteractiveMarkerFeedback.MOUSE_UP):
            self.send({"obstacle": int(name.split("_")[1]), "pos": [p.x, p.y, p.z]})   # live while dragging

    def send(self, msg):
        self.cmd.sendto(json.dumps(msg).encode(), self.cmd_dst)

    def sync_handle(self, name, xyz, **insert):
        """Snap a handle to the accepted value (clamped, or unchanged if refused) whenever nobody is dragging it."""
        if name not in self.shown:
            self.insert_handle(name, xyz, **insert)
            return
        if name in self.dragging or time.monotonic() - self.last_release.get(name, 0.0) < 0.5:
            return
        if max(abs(a - b) for a, b in zip(xyz, self.shown[name])) > 0.005:
            pose = Pose()
            pose.position = pt(xyz)
            pose.orientation.w = 1.0
            self.server.setPose(name, pose)
            self.server.applyChanges()
            self.shown[name] = list(xyz)

    def sync_obstacle_handles(self, goals, radii):
        for i, g in enumerate(goals):
            self.sync_handle(f"obstacle_{i}", g, description=f"obstacle {i}: drag", color=rgba(1.0, 0.5, 0.1, 0.35),
                             size=2 * radii[i] + 0.06)   # a shell around the obstacle sphere, so clicks land on the handle
        for name in [n for n in self.shown if n.startswith("obstacle_") and int(n.split("_")[1]) >= len(goals)]:
            self.server.erase(name)
            self.server.applyChanges()
            del self.shown[name]

    # --- state -> markers ---
    def tick(self):
        if not rclpy.ok():   # shutting down (Ctrl+C / SIGTERM from run_rviz.sh)
            return
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                self.state, self.last_rx = json.loads(data), time.monotonic()
            except ValueError:
                pass
        if self.state is None:
            return
        st = self.state
        if st.get("target"):
            self.sync_handle("target", st["target"], description="target: drag, release to send", color=rgba(0.2, 1.0, 0.3, 0.5))
        self.sync_obstacle_handles(st.get("obstacle_goals") or [], [o["radius"] for o in st["obstacles"]])
        self.pub.publish(self.markers(st, time.monotonic() - self.last_rx > 1.0))

    def marker(self, ns, mid, mtype, color, scale=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 0.0)):
        m = Marker()
        m.header.frame_id = FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.position = pt(pos)
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.color = color
        return m

    def markers(self, st, stale):
        arena, out = st["arena"], MarkerArray()
        R, H, TR, (z0, z1) = arena["radius"], arena["height"], arena["target_radius"], arena["target_z"]
        wall = self.marker("arena", 0, Marker.LINE_LIST, rgba(0.8, 0.8, 0.8, 0.5), (0.01, 0.0, 0.0))
        wall.points = wire_cylinder(R, 0.0, H)
        soft = self.marker("arena", 1, Marker.LINE_STRIP, rgba(1.0, 0.6, 0.1, 0.6), (0.01, 0.0, 0.0))
        soft.points = circle(R - arena["soft_wall"], 0.0)
        region = self.marker("arena", 2, Marker.LINE_LIST, rgba(0.2, 0.9, 0.3, 0.35), (0.008, 0.0, 0.0))
        region.points = wire_cylinder(TR, z0, z1, posts=4)
        out.markers += [wall, soft, region]
        drone = st.get("drone")
        if drone:
            p = drone["pos"]
            w, x, y, z = drone["quat"]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id, t.child_frame_id = FRAME, "cf"
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in p)
            t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = (float(v) for v in (w, x, y, z))
            self.tf.sendTransform(t)
            out.markers.append(self.marker("drone", 0, Marker.SPHERE, rgba(0.2, 0.5, 1.0, 1.0), (0.1, 0.1, 0.04), p))
            head = self.marker("drone", 1, Marker.ARROW, rgba(0.2, 0.5, 1.0, 1.0), (0.015, 0.03, 0.03))
            head.points = [pt(p), pt((p[0] + 0.15 * math.cos(yaw), p[1] + 0.15 * math.sin(yaw), p[2]))]
            out.markers.append(head)
            if not self.trail or math.dist(self.trail[-1], p) > 0.005:
                self.trail.append(tuple(p))
            trail = self.marker("drone", 2, Marker.LINE_STRIP, rgba(0.4, 0.6, 1.0, 0.6), (0.008, 0.0, 0.0))
            trail.points = [pt(q) for q in self.trail]
            out.markers.append(trail)
            cmd = self.marker("drone", 3, Marker.ARROW, rgba(1.0, 0.55, 0.1, 0.9 if st.get("cmd") else 0.0), (0.02, 0.04, 0.05))
            c = st.get("cmd") or [0.0, 0.0, 0.0]
            cmd.points = [pt(p), pt((p[0] + 0.5 * c[0], p[1] + 0.5 * c[1], p[2] + 0.5 * c[2]))]
            out.markers.append(cmd)
            to_tgt = self.marker("drone", 4, Marker.LINE_LIST, rgba(0.2, 1.0, 0.3, 0.5 if st["mode"] == "policy" else 0.0), (0.006, 0, 0))
            to_tgt.points = [pt(p), pt(st["target"] or p)]
            out.markers.append(to_tgt)
        hold = st.get("hold")
        out.markers.append(self.marker("drone", 5, Marker.SPHERE, rgba(1.0, 1.0, 0.2, 0.8 if hold else 0.0), (0.04, 0.04, 0.04),
                                       hold or (0, 0, 0)))
        for i, o in enumerate(st["obstacles"]):
            r, unsafe = o["radius"], o["radius"] + arena["unsafe_extra"]
            out.markers.append(self.marker("obstacles", 3 * i, Marker.SPHERE, rgba(0.9, 0.15, 0.1, 0.9), (2 * r,) * 3, o["pos"]))
            ring = self.marker("obstacles", 3 * i + 1, Marker.LINE_STRIP, rgba(0.9, 0.15, 0.1, 0.6), (0.008, 0.0, 0.0))
            ring.points = circle(unsafe, o["pos"][2], 32, o["pos"][:2])   # unsafe zone; a sphere would block clicks on the handle
            out.markers.append(ring)
            vel = self.marker("obstacles", 3 * i + 2, Marker.ARROW, rgba(0.9, 0.3, 0.3, 0.9), (0.015, 0.03, 0.03))
            v = o["vel"]
            vel.points = [pt(o["pos"]), pt((o["pos"][0] + 0.5 * v[0], o["pos"][1] + 0.5 * v[1], o["pos"][2] + 0.5 * v[2]))]
            out.markers.append(vel)
        for i in range(len(st["obstacles"]), 8):
            for k in range(3):
                gone = self.marker("obstacles", 3 * i + k, Marker.SPHERE, rgba(0, 0, 0, 0))
                gone.action = Marker.DELETE
                out.markers.append(gone)
        vb = st.get("vbat")
        text = "NO DATA FROM fly_policy / sim_policy" if stale else f"{st['mode'].upper()}" + (f"   battery {vb:.2f} V" if vb else "")
        if st.get("msg"):
            text += f"\n{st['msg']}"
        label = self.marker("text", 0, Marker.TEXT_VIEW_FACING, rgba(1, 1, 1, 1), (0, 0, 0.12), (0, 0, H + 0.25))
        label.text = text
        out.markers.append(label)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-port", type=int, default=47800)
    ap.add_argument("--cmd-port", type=int, default=47801)
    args = ap.parse_args()
    rclpy.init()
    node = Bridge(args.state_port, args.cmd_port)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise   # a timer racing the SIGTERM shutdown is fine; anything else is a real error
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
