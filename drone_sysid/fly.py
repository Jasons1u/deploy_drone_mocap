"""Fly a sysID plan on the Crazyflie (ROS 2 mocap pose + appchannel mu), or against the simulated plant with --sim."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import Config
from controller import LOG_HEADER, SysIdController
from plans import build_plan

APP_PACKET = "@ifff"


def write_log(ctrl: SysIdController, path: Path, meta: dict) -> Path:
    """Write the CSV and a JSON sidecar with the run settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_HEADER)
        w.writerows(ctrl.rows)
    meta = {**meta, "rows": len(ctrl.rows), "stop_reason": ctrl.stop_reason, "events": ctrl.events}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[sysid] wrote {len(ctrl.rows)} rows -> {path}", flush=True)
    return path


def plan_seconds(cfg: Config, segments) -> float:
    return cfg.pose_warmup + cfg.takeoff_time + cfg.settle_time + sum(s.duration for s in segments) + cfg.land_time


def run_sim(plan: str, out: Path, params=None, seed: int = 0, hover_seconds: float = 30.0, passes: int = 1,
            cfg: Config | None = None, inject=None, dry_run: bool = False) -> tuple[Path, SysIdController]:
    """Fly the plan against SimPlant at 240 Hz mocap; `inject(t, plant, ctrl) -> pose or None` can fake faults."""
    from sim_plant import PlantParams, SimPlant

    cfg = cfg or Config()
    params = params or PlantParams()
    segments, hover_z = build_plan(plan, hover_seconds, passes)
    ctrl = SysIdController(cfg, segments, hover_z, dry_run=dry_run)
    plant = SimPlant(params, seed=seed)
    dt, t, next_wd = 1.0 / 240.0, 0.0, 0.0
    ctrl.sent_flags = []
    t_max = plan_seconds(cfg, segments) + 60.0
    while not ctrl.done and t < t_max:
        plant.step(dt)
        t += dt
        ctrl.vbat = params.vbat
        pose = plant.pose() if inject is None else inject(t, plant, ctrl)
        if pose is not None:
            flag, mu = ctrl.on_pose(t, pose, (1.0, 0.0, 0.0, 0.0))
            plant.send(flag, mu)
            ctrl.sent_flags.append(flag)
        if t >= next_wd:
            ctrl.check_watchdog(t)
            next_wd += 0.05
        if ctrl.stopped:
            plant.send(0, np.zeros(3))
            ctrl.sent_flags.append(0)
    meta = {"mode": "sim", "plan": plan, "hover_z": hover_z, "plant": asdict(params), "config": asdict(cfg)}
    return write_log(ctrl, out, meta), ctrl


def kill_listener(on_kill):
    """SPACE or q in this terminal kills the motors; returns a restore() callback for the terminal."""
    import select
    import termios
    import tty

    if not sys.stdin.isatty():
        print("[sysid] WARNING: stdin is not a terminal -- SPACE kill unavailable, use Ctrl+C", flush=True)
        return lambda: None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    stop = threading.Event()

    def loop():
        tty.setcbreak(fd)
        while not stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready and sys.stdin.read(1) in (" ", "q"):
                on_kill()
                return

    threading.Thread(target=loop, daemon=True).start()

    def restore():
        stop.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return restore


def run_hardware(args, cfg: Config) -> None:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig

    segments, hover_z = build_plan(args.plan, args.hover_seconds, args.passes)
    ctrl = SysIdController(cfg, segments, hover_z, dry_run=args.dry_run)
    out = Path(args.out_dir) / f"sysid_{args.plan}_{'dry_' if args.dry_run else ''}{int(time.time())}.csv"
    meta = {"mode": "dry-run" if args.dry_run else "live", "plan": args.plan, "hover_z": hover_z, "uri": cfg.uri,
            "pose_topic": args.pose_topic, "config": asdict(cfg)}

    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache="./cache")

    def send(flag, mu):
        cf.appchannel.send_packet(struct.pack(APP_PACKET, int(flag), float(mu[0]), float(mu[1]), float(mu[2])))

    def send_zeros():
        for _ in range(5):
            try:
                send(0, np.zeros(3))
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.02)

    def shutdown_ros():
        try:
            rclpy.try_shutdown()
        except Exception:  # noqa: BLE001
            pass

    class SysIdNode(Node):
        def __init__(self):
            super().__init__("cf_sysid")
            self.create_subscription(PoseStamped, args.pose_topic, self.on_pose, 10)
            self.create_timer(0.01, self.extpos)
            self.create_timer(0.05, self.watchdog)

        def on_pose(self, msg):
            p, o = msg.pose.position, msg.pose.orientation
            flag, mu = ctrl.on_pose(time.monotonic(), (p.x, p.y, p.z), (o.w, o.x, o.y, o.z))
            try:
                send(flag, mu)
            except Exception as exc:  # noqa: BLE001
                ctrl.request_stop(f"send failed: {exc}")
            if ctrl.done:
                self.finish()

        def extpos(self):
            if ctrl.initialized:
                r, q = ctrl.r.copy(), ctrl.q.copy()
                cf.extpos.send_extpose(x=float(r[0]), y=float(r[1]), z=float(r[2]),
                                       qx=float(q[1]), qy=float(q[2]), qz=float(q[3]), qw=float(q[0]))

        def watchdog(self):
            ctrl.check_watchdog(time.monotonic())
            if ctrl.done:
                self.finish()

        def finish(self):
            send_zeros()
            shutdown_ros()

    def on_kill():
        ctrl.request_stop("kill key")
        send_zeros()
        shutdown_ros()

    battery_log = None
    restore_tty = lambda: None  # noqa: E731
    try:
        print(f"[sysid] connecting {cfg.uri}", flush=True)
        cf.open_link(cfg.uri)
        deadline = time.monotonic() + 15.0
        while not cf.is_connected():
            if time.monotonic() > deadline:
                raise TimeoutError("radio connection timed out")
            time.sleep(0.1)
        battery_log = LogConfig(name="battery", period_in_ms=100)
        battery_log.add_variable("pm.vbat", "float")
        cf.log.add_config(battery_log)
        battery_log.data_received_cb.add_callback(lambda ts, data, lc: setattr(ctrl, "vbat", float(data["pm.vbat"])))
        battery_log.start()
        for _ in range(40):
            if math.isfinite(ctrl.vbat):
                break
            time.sleep(0.05)
        print(f"[sysid] plan {args.plan}: {len(segments)} segments, ~{plan_seconds(cfg, segments):.0f} s, hover {hover_z} m, "
              f"topic {args.pose_topic}, battery {ctrl.vbat:.2f} V", flush=True)
        if args.dry_run:
            print("[sysid] DRY RUN -- start_flag is forced to 0, motors will NOT arm", flush=True)
        else:
            reply = input("[sysid] *** LIVE FLIGHT *** props on, area clear, finger on SPACE. Type 'fly' to arm: ")
            if reply.strip() != "fly":
                print("[sysid] aborted before arming", flush=True)
                return
        rclpy.init()
        node = SysIdNode()
        restore_tty = kill_listener(on_kill)
        rclpy.spin(node)
    except KeyboardInterrupt:
        ctrl.request_stop("Ctrl+C")
    except Exception as exc:  # noqa: BLE001
        ctrl.request_stop(f"fatal: {exc}")
    finally:
        write_log(ctrl, out, meta)  # first: radio teardown can hang on a lost link
        send_zeros()
        restore_tty()
        if battery_log is not None:
            try:
                battery_log.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            cf.close_link()
        except Exception:  # noqa: BLE001
            pass
        shutdown_ros()
        print(f"[sysid] finished: {ctrl.stop_reason or 'plan complete'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan", choices=["hover", "vsteps", "thrust"])
    ap.add_argument("--pose-topic", help="ROS 2 PoseStamped topic of this drone's rigid body (required on hardware)")
    ap.add_argument("--uri", default=Config.uri)
    ap.add_argument("--dry-run", action="store_true", help="full pipeline against live mocap, motors never arm")
    ap.add_argument("--sim", action="store_true", help="fly against the simulated plant (no radio, no ROS)")
    ap.add_argument("--hover-seconds", type=float, default=30.0)
    ap.add_argument("--passes", type=int, default=1, help="thrust plan: repeat the staircase")
    ap.add_argument("--out-dir", default="logs")
    args = ap.parse_args()

    cfg = Config(uri=args.uri)
    if args.sim:
        out = Path(args.out_dir) / f"sysid_{args.plan}_sim_{int(time.time())}.csv"
        run_sim(args.plan, out, hover_seconds=args.hover_seconds, passes=args.passes, cfg=cfg)
        return
    if not args.pose_topic:
        ap.error("--pose-topic is required on hardware (check it with check_mocap.py first)")
    run_hardware(args, cfg)


if __name__ == "__main__":
    main()
