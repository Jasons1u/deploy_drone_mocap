#!/usr/bin/env python3
"""Headless check of rviz_bridge.py: fake fly_policy state in, markers / TF / interactive target out, drag -> UDP target.

    env -i HOME=$HOME bash -c 'source /opt/ros/humble/setup.bash && /usr/bin/python3 crazyflie/viz/test_bridge.py'
"""

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import InteractiveMarkerFeedback, InteractiveMarkerUpdate, MarkerArray
from visualization_msgs.srv import GetInteractiveMarkers

STATE_PORT, CMD_PORT = 47910, 47911
ARENA = {"radius": 1.8288, "height": 2.8956, "target_radius": 1.4, "target_z": [0.5, 2.0], "soft_wall": 0.15, "soft_top": 2.6,
         "soft_bottom": 0.25, "unsafe_extra": 0.1, "keep_out_extra": 0.2}
state = {"mode": "policy", "cmd": [0.3, 0.0, 0.0], "hold": None,
         "drone": {"pos": [0.1, -0.2, 0.9], "quat": [1.0, 0.0, 0.0, 0.0], "age": 0.004},
         "target": [0.5, 0.5, 1.0], "msg": "target (+0.50, +0.50, +1.00)",
         "obstacles": [{"pos": [0.0, 0.6, 1.2], "radius": 0.1, "vel": [0.2, 0.0, 0.0]}], "vbat": 3.95, "arena": ARENA}


def sender(stop):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not stop.is_set():
        s.sendto(json.dumps(state).encode(), ("127.0.0.1", STATE_PORT))
        time.sleep(1 / 30)


class Probe(Node):
    def __init__(self):
        super().__init__("cf_viz_probe")
        self.markers, self.tf, self.updates = None, [], []
        self.create_subscription(MarkerArray, "/cf_viz/markers", lambda m: setattr(self, "markers", m), 10)
        self.create_subscription(TFMessage, "/tf", lambda m: self.tf.extend(t.child_frame_id for t in m.transforms), 10)
        self.create_subscription(InteractiveMarkerUpdate, "/cf_target/update", self.updates.append, 10)
        self.fb = self.create_publisher(InteractiveMarkerFeedback, "/cf_target/feedback", 10)
        self.cli = self.create_client(GetInteractiveMarkers, "/cf_target/get_interactive_markers")

    def handles(self):
        """Current handles as RViz fetches them on connect (the update topic only carries changes)."""
        if not self.cli.wait_for_service(timeout_sec=3.0):
            return {}
        fut = self.cli.call_async(GetInteractiveMarkers.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        return {im.name: im for im in (fut.result().markers if fut.result() else [])}


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def main() -> int:
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    bridge = subprocess.Popen([sys.executable, str(Path(__file__).with_name("rviz_bridge.py")),
                               "--state-port", str(STATE_PORT), "--cmd-port", str(CMD_PORT)])
    stop = threading.Event()
    threading.Thread(target=sender, args=(stop,), daemon=True).start()
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", CMD_PORT))
    rx.settimeout(3.0)
    rclpy.init()
    probe = Probe()
    try:
        spin_for(probe, 4.0)
        m = probe.markers
        ns = sorted({x.ns for x in m.markers}) if m else []
        check("marker array published with arena / drone / obstacles / text", ns == ["arena", "drone", "obstacles", "text"], f"({ns})")
        sphere = [x for x in m.markers if x.ns == "obstacles" and x.id == 0][0] if m else None
        check("obstacle sphere at its position with diameter 2r", sphere is not None and abs(sphere.scale.x - 0.2) < 1e-6
              and abs(sphere.pose.position.y - 0.6) < 1e-6)
        label = [x for x in m.markers if x.ns == "text"][0].text if m else ""
        check("status label shows mode, battery, last target message", "POLICY" in label and "3.95" in label and "target" in label,
              f"({label!r})")
        check("TF world -> cf published", "cf" in probe.tf)
        solid = [(x.ns, x.id) for x in m.markers if x.ns == "arena" or (x.ns == "obstacles" and x.id % 3 == 1)
                 if x.action == x.ADD and x.type not in (x.LINE_LIST, x.LINE_STRIP)] if m else ["no markers"]
        check("arena and unsafe zones drawn as lines (solid see-through shapes swallow clicks on the handles)", not solid, f"({solid})")
        tgt = probe.handles().get("target")
        check("interactive target marker created at the state's target", tgt is not None and abs(tgt.pose.position.x - 0.5) < 1e-6
              and abs(tgt.pose.position.z - 1.0) < 1e-6, f"({tgt.pose.position if tgt else None})")

        for ev, xyz in ((InteractiveMarkerFeedback.MOUSE_DOWN, (0.5, 0.5, 1.0)), (InteractiveMarkerFeedback.POSE_UPDATE, (-0.2, 0.3, 1.2)),
                        (InteractiveMarkerFeedback.MOUSE_UP, (-0.4, 0.2, 1.3))):
            fb = InteractiveMarkerFeedback()
            fb.header.frame_id, fb.client_id, fb.marker_name, fb.event_type = "world", "/test", "target", ev
            fb.pose.position.x, fb.pose.position.y, fb.pose.position.z = xyz
            fb.pose.orientation.w = 1.0
            probe.fb.publish(fb)
            spin_for(probe, 0.2)
        got = json.loads(rx.recvfrom(65535)[0])
        check("drag release sends exactly the released pose as a target request", got == {"target": [-0.4, 0.2, 1.3]}, f"({got})")
        try:
            rx.settimeout(0.3)
            extra = rx.recvfrom(65535)[0]
        except socket.timeout:
            extra = None
        check("no request while dragging (only on release)", extra is None)

        state["target"] = [0.7, 0.1, 1.1]
        probe.updates.clear()
        spin_for(probe, 1.5)
        poses = [p.pose.position for u in probe.updates for p in u.poses if p.name == "target"]
        check("handle snaps to the accepted target after release", poses and abs(poses[-1].x - 0.7) < 1e-6 and abs(poses[-1].z - 1.1) < 1e-6,
              f"({[(p.x, p.y, p.z) for p in poses[-1:]]})")
        for ev, xyz in ((InteractiveMarkerFeedback.MOUSE_DOWN, (0.7, 0.1, 1.1)), (InteractiveMarkerFeedback.POSE_UPDATE, (0.0, 0.6, 1.2)),
                        (InteractiveMarkerFeedback.MOUSE_UP, (0.0, 0.6, 1.2))):
            fb = InteractiveMarkerFeedback()
            fb.header.frame_id, fb.client_id, fb.marker_name, fb.event_type = "world", "/test", "target", ev
            fb.pose.position.x, fb.pose.position.y, fb.pose.position.z = xyz
            fb.pose.orientation.w = 1.0
            probe.fb.publish(fb)
            spin_for(probe, 0.2)
        rx.settimeout(2.0)
        rx.recvfrom(65535)
        probe.updates.clear()
        spin_for(probe, 1.5)
        poses = [p.pose.position for u in probe.updates for p in u.poses if p.name == "target"]
        check("refused request (target unchanged): handle snaps back", poses and abs(poses[-1].x - 0.7) < 1e-6 and abs(poses[-1].y - 0.1) < 1e-6,
              f"({[(p.x, p.y, p.z) for p in poses[-1:]]})")
        names = {im.name for u in probe.updates for im in u.markers}
        check("fly_policy state (no obstacle_goals): obstacles are not draggable", not any(n.startswith("obstacle_") for n in names))

        state.update(sim=True, mode="sim", obstacle_goals=[[0.0, 0.6, 1.2]])
        probe.updates.clear()
        spin_for(probe, 1.0)
        handles = [im for name, im in probe.handles().items() if name == "obstacle_0"]
        check("sim state: obstacle handle created at its goal", handles and abs(handles[-1].pose.position.y - 0.6) < 1e-6
              and abs(handles[-1].pose.position.z - 1.2) < 1e-6)
        grab = [c for c in handles[-1].controls if c.name == "grab"] if handles else []
        check("obstacle grab ball is draggable and a shell around the obstacle sphere (d 0.20)", grab
              and grab[0].interaction_mode == grab[0].MOVE_PLANE and grab[0].markers[0].scale.x > 0.2, f"({grab[0].markers[0].scale.x if grab else None})")
        for ev, xyz in ((InteractiveMarkerFeedback.MOUSE_DOWN, (0.0, 0.6, 1.2)), (InteractiveMarkerFeedback.POSE_UPDATE, (0.2, 0.5, 1.2))):
            fb = InteractiveMarkerFeedback()
            fb.header.frame_id, fb.client_id, fb.marker_name, fb.event_type = "world", "/test", "obstacle_0", ev
            fb.pose.position.x, fb.pose.position.y, fb.pose.position.z = xyz
            fb.pose.orientation.w = 1.0
            probe.fb.publish(fb)
            spin_for(probe, 0.2)
        rx.settimeout(2.0)
        got = json.loads(rx.recvfrom(65535)[0])
        check("dragging an obstacle streams its pose before release", got == {"obstacle": 0, "pos": [0.2, 0.5, 1.2]}, f"({got})")
        state["obstacle_goals"] = []
        probe.updates.clear()
        spin_for(probe, 1.0)
        check("handle removed when the obstacle goes away", any("obstacle_0" in u.erases for u in probe.updates))
    finally:
        stop.set()
        bridge.terminate()
        bridge.wait(timeout=5)
        probe.destroy_node()
        rclpy.try_shutdown()
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
