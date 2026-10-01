"""Fly a sysID plan on the Crazyflie (OptiTrack via NatNet + appchannel mu), or against the simulated plant (--sim).

Mocap and radio settings come from the same config as cf-hover (crazyflie/config/hover.yaml: server
IP, rigid body ID, up axis, and the body yaw/tilt offsets), so both tools see the drone identically.
"""

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


def run_hardware(args, cfg: Config, mocap_cfg) -> None:
    """mocap_cfg: cf_mocap MocapConfig (the `mocap:` section of crazyflie/config/hover.yaml)."""
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from mocap_natnet import NatNetClient

    segments, hover_z = build_plan(args.plan, args.hover_seconds, args.passes)
    ctrl = SysIdController(cfg, segments, hover_z, dry_run=args.dry_run)
    out = Path(args.out_dir) / f"sysid_{args.plan}_{'dry_' if args.dry_run else ''}{int(time.time())}.csv"
    body = mocap_cfg.rigid_body_id
    meta = {"mode": "dry-run" if args.dry_run else "live", "plan": args.plan, "hover_z": hover_z, "uri": cfg.uri,
            "mocap": {"server_ip": mocap_cfg.server_ip, "rigid_body_id": body, "up_axis": mocap_cfg.up_axis,
                      "body_yaw_offset_deg": mocap_cfg.body_yaw_offset_deg,
                      "body_tilt_offset_deg": list(mocap_cfg.body_tilt_offset_deg)},
            "config": asdict(cfg)}

    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache="./cache")
    finished = threading.Event()

    def send(flag, mu):
        cf.appchannel.send_packet(struct.pack(APP_PACKET, int(flag), float(mu[0]), float(mu[1]), float(mu[2])))

    def send_zeros():
        for _ in range(5):
            try:
                send(0, np.zeros(3))
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.02)

    def on_pose(body_id, pose):
        """NatNet receive thread, once per tracked frame (Motive rate)."""
        if body_id != body or finished.is_set():
            return
        flag, mu = ctrl.on_pose(pose.stamp, pose.position, pose.quat)
        try:
            send(flag, mu)
        except Exception as exc:  # noqa: BLE001
            ctrl.request_stop(f"send failed: {exc}")
        if ctrl.done:
            finished.set()

    def on_kill():
        ctrl.request_stop("kill key")
        finished.set()
        send_zeros()

    mocap = NatNetClient(mocap_cfg.server_ip, local_ip=mocap_cfg.local_ip, up_axis=mocap_cfg.up_axis,
                         natnet_version=mocap_cfg.natnet_version, body_rotation=mocap_cfg.body_rotation(),
                         on_pose=on_pose)
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
        if cf.param.toc.get_element_by_complete_name("mocapEst.velAlpha") is not None:
            raise RuntimeError("the drone runs the mocap-estimator firmware (crazyflie/firmware), which has no "
                               "appchannel controller; sysID needs Ryan's firmware flashed")
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
              f"mocap {mocap_cfg.server_ip} body {body}, battery {ctrl.vbat:.2f} V", flush=True)
        if args.dry_run:
            print("[sysid] DRY RUN -- start_flag is forced to 0, motors will NOT arm", flush=True)
        else:
            reply = input("[sysid] *** LIVE FLIGHT *** props on, area clear, finger on SPACE. Type 'fly' to arm: ")
            if reply.strip() != "fly":
                print("[sysid] aborted before arming", flush=True)
                return
        mocap.start()
        restore_tty = kill_listener(on_kill)
        next_extpos = next_watchdog = time.monotonic()
        while not finished.is_set():
            now = time.monotonic()
            if now >= next_extpos:                       # 100 Hz: mocap into the onboard estimator
                next_extpos += 0.01
                if ctrl.initialized:
                    r, q = ctrl.r.copy(), ctrl.q.copy()
                    cf.extpos.send_extpose(x=float(r[0]), y=float(r[1]), z=float(r[2]),
                                           qx=float(q[1]), qy=float(q[2]), qz=float(q[3]), qw=float(q[0]))
            if now >= next_watchdog:                     # 20 Hz: stale mocap stops, frozen mocap lands
                next_watchdog += 0.05
                ctrl.check_watchdog(now)
                if ctrl.done:
                    finished.set()
            time.sleep(0.002)
    except KeyboardInterrupt:
        ctrl.request_stop("Ctrl+C")
    except Exception as exc:  # noqa: BLE001
        ctrl.request_stop(f"fatal: {exc}")
    finally:
        finished.set()                                   # on_pose sends nothing from here on
        write_log(ctrl, out, meta)  # first: radio teardown can hang on a lost link
        send_zeros()
        restore_tty()
        mocap.stop()
        if battery_log is not None:
            try:
                battery_log.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            cf.close_link()
        except Exception:  # noqa: BLE001
            pass
        print(f"[sysid] finished: {ctrl.stop_reason or 'plan complete'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan", choices=["hover", "vsteps", "thrust"])
    ap.add_argument("--config", default=None,
                    help="mocap + radio config shared with cf-hover (default: crazyflie/config/hover.yaml)")
    ap.add_argument("--uri", help="radio URI (default: radio.uri from the config)")
    ap.add_argument("--dry-run", action="store_true", help="full pipeline against live mocap, motors never arm")
    ap.add_argument("--sim", action="store_true", help="fly against the simulated plant (no radio, no ROS)")
    ap.add_argument("--hover-seconds", type=float, default=30.0)
    ap.add_argument("--passes", type=int, default=1, help="thrust plan: repeat the staircase")
    ap.add_argument("--out-dir", default="logs")
    args = ap.parse_args()

    if args.sim:
        cfg = Config(uri=args.uri or Config.uri)
        out = Path(args.out_dir) / f"sysid_{args.plan}_sim_{int(time.time())}.csv"
        run_sim(args.plan, out, hover_seconds=args.hover_seconds, passes=args.passes, cfg=cfg)
        return
    from cf_mocap.config import DEFAULT_CONFIG, load_config
    shared = load_config(args.config or DEFAULT_CONFIG)
    if not shared.mocap.server_ip:
        ap.error(f"mocap.server_ip is not set in {shared.source}")
    run_hardware(args, Config(uri=args.uri or shared.radio.uri), shared.mocap)


if __name__ == "__main__":
    main()
