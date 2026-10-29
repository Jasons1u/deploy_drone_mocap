"""Link to crazyflie/viz/rviz_bridge.py: state JSON out over localhost UDP, drag requests (target, sim obstacles) back."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading

from cfdeploy.config import repo_path
from cfdeploy.safe_cf_obs import ARENA_HEIGHT, ARENA_RADIUS, DRONE_RADIUS, TARGET_KEEP_OUT_EXTRA, TARGET_RADIUS, TARGET_Z, UNSAFE_MARGIN


def arena_state(limits) -> dict:
    """Arena, trained target region and safety margins for the bridge's static markers."""
    return {"radius": ARENA_RADIUS, "height": ARENA_HEIGHT, "target_radius": TARGET_RADIUS, "target_z": list(TARGET_Z),
            "soft_wall": limits.soft_wall, "soft_top": limits.soft_top, "soft_bottom": limits.soft_bottom,
            "unsafe_extra": DRONE_RADIUS + UNSAFE_MARGIN, "keep_out_extra": DRONE_RADIUS + UNSAFE_MARGIN + TARGET_KEEP_OUT_EXTRA}


def obstacle_state(obstacles) -> list:
    return [{"pos": list(o.pos), "radius": o.radius, "vel": list(o.vel)} for o in obstacles]


class VizSocket:
    """Non-blocking UDP pair: send(state) to the bridge, poll() its requests."""

    def __init__(self, viz_cfg):
        self.dst = ("127.0.0.1", viz_cfg.state_port)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.inp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.inp.bind(("127.0.0.1", viz_cfg.cmd_port))
        self.inp.setblocking(False)

    def send(self, state: dict) -> None:
        try:
            self.out.sendto(json.dumps(state).encode(), self.dst)
        except OSError:
            pass

    def poll(self) -> list:
        msgs = []
        while True:
            try:
                data, _ = self.inp.recvfrom(65535)
            except (BlockingIOError, OSError):
                return msgs
            try:
                msgs.append(json.loads(data))
            except ValueError:
                pass

    def close(self) -> None:
        self.inp.close()
        self.out.close()


class VizLink:
    """Background thread for the flight runner: state_fn() out at the viz rate, each request to on_message(msg)."""

    def __init__(self, viz_cfg, state_fn, on_message):
        self.sock, self.period = VizSocket(viz_cfg), 1.0 / viz_cfg.rate
        self.state_fn, self.on_message = state_fn, on_message
        self.stop_evt = threading.Event()

    def start(self) -> "VizLink":
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def run(self) -> None:
        while not self.stop_evt.is_set():
            for msg in self.sock.poll():
                self.on_message(msg)
            self.sock.send(self.state_fn())
            self.stop_evt.wait(self.period)
        self.sock.close()

    def close(self) -> None:
        self.stop_evt.set()


# #### // The RViz process (crazyflie/viz/run_rviz.sh) // ####
def launch_rviz(cfg):
    """crazyflie/viz/run_rviz.sh (bridge + RViz) in its own process group on our ports; output to logs/rviz.log."""
    if not os.environ.get("DISPLAY"):
        print("no DISPLAY: not starting RViz (run crazyflie/viz/run_rviz.sh on a machine with a screen)")
        return None
    log = repo_path(cfg.paths.log_dir) / "rviz.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        proc = subprocess.Popen([str(repo_path("crazyflie/viz/run_rviz.sh")), "--state-port", str(cfg.viz.state_port), "--cmd-port", str(cfg.viz.cmd_port)],
                                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"RViz starting (log {log})")
    return proc


def stop_rviz(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=wait)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue
