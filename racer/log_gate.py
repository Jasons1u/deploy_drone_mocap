#!/usr/bin/env python3
"""Mocap log of a racer and a gate (wall-clock time, poses, gate in the racer body frame) to check a vision stack against."""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import add_override_args, apply_overrides, load, repo_path  # noqa: E402
from cfdeploy.geometry import quat_rotate  # noqa: E402
from cfdeploy.natnet import NatNetClient  # noqa: E402

POSE = ["x", "y", "z", "qw", "qx", "qy", "qz", "roll", "pitch", "yaw"]
HEADER = (["t_unix", "time", "t", "frame"]
          + [f"racer_{k}" for k in ["ok"] + POSE + ["err"]]
          + [f"gate_{k}" for k in ["ok"] + POSE + ["err"]]
          + [f"rel_{k}" for k in POSE] + ["dist"])
NAN_POSE = [math.nan] * len(POSE)


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def euler_deg(q) -> list:
    """Roll, pitch, yaw [deg] (ZYX intrinsic) of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def pose(p, q) -> list:
    return [*p, *q, *euler_deg(q)]


def relative(racer, gate) -> tuple[list, float]:
    """Gate pose in the racer body frame (position, gate->racer quaternion, its Euler angles) and the distance."""
    qr = np.array(racer.quat)
    d = np.array(gate.position) - np.array(racer.position)
    q = quat_mul(quat_conj(qr), np.array(gate.quat))
    q = q if q[0] >= 0 else -q
    return pose(quat_rotate(quat_conj(qr), d).tolist(), q.tolist()), float(np.linalg.norm(d))


def frame_row(frame, racer_id: int, gate_id: int, wall_offset: float, latency: float) -> list:
    """One CSV row: exposure time (wall clock), both world poses (NaN when untracked), the relative pose."""
    t_unix = frame.arrival + wall_offset - latency
    stamp = datetime.datetime.fromtimestamp(t_unix).isoformat(timespec="microseconds")
    row = [f"{t_unix:.6f}", stamp, frame.arrival, frame.number]
    bodies = []
    for bid in (racer_id, gate_id):
        b = frame.bodies.get(bid)
        ok = b is not None and b.valid
        bodies.append(b if ok else None)
        row += [int(ok)] + (pose(b.position, b.quat) if ok else NAN_POSE) + [b.mean_error if ok else math.nan]
    if all(bodies):
        rel, dist = relative(*bodies)
        row += rel + [dist]
    else:
        row += NAN_POSE + [math.nan]
    return row


class PairLogger:
    """NatNet on_frame callback: one CSV row per mocap frame, plus counters and the newest row for the status line."""

    def __init__(self, path: Path, racer_id: int, gate_id: int, latency: float):
        self.racer_id, self.gate_id, self.latency = racer_id, gate_id, latency
        self.wall_offset = time.time() - time.monotonic()
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(HEADER)
        self.frames = self.racer_frames = self.gate_frames = self.both_frames = 0
        self.last = None
        self.seen: set[int] = set()

    def __call__(self, frame) -> None:
        if self.file.closed:
            return
        row = frame_row(frame, self.racer_id, self.gate_id, self.wall_offset, self.latency)
        self.writer.writerow(row)
        r_ok, g_ok = row[4], row[4 + len(POSE) + 2]
        self.frames += 1
        self.racer_frames += r_ok
        self.gate_frames += g_ok
        self.both_frames += r_ok and g_ok
        self.seen.update(frame.bodies)
        self.last = row

    def close(self) -> None:
        self.file.close()


def status(log: PairLogger) -> str:
    row = log.last
    if row is None:
        return "waiting for mocap frames..."
    i = 4 + 2 * (len(POSE) + 2)
    r_ok, g_ok = row[4], row[4 + len(POSE) + 2]
    rel = "gate in racer frame: --" if not (r_ok and g_ok) else (
        f"gate in racer frame [{row[i]:+.3f} {row[i + 1]:+.3f} {row[i + 2]:+.3f}] m  dist {row[-1]:.3f} m  "
        f"rel yaw {row[i + 9]:+6.1f} deg")
    return f"{row[1][11:23]}  frames {log.frames:6d}  racer {'OK ' if r_ok else 'LOST'}  gate {'OK ' if g_ok else 'LOST'}  {rel}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--racer", type=int, help="Motive rigid body ID of the racer (default gate_log.racer_body)")
    ap.add_argument("--gate", type=int, help="Motive rigid body ID of the gate (default gate_log.gate_body)")
    ap.add_argument("--duration", type=float, help="[s] stop after this long (default: Ctrl+C)")
    ap.add_argument("--tag", default="", help="appended to the log folder name")
    add_override_args(ap)
    opts = ap.parse_args()
    cfg = apply_overrides(load(), opts.set)
    racer_id = opts.racer if opts.racer is not None else cfg.gate_log.racer_body
    gate_id = opts.gate if opts.gate is not None else cfg.gate_log.gate_body

    out = repo_path(cfg.gate_log.log_dir) / (time.strftime("%Y%m%d_%H%M%S") + "_gate" + (f"_{opts.tag}" if opts.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    log = PairLogger(out / "gate_pose.csv", racer_id, gate_id, cfg.mocap.latency)
    mocap = NatNetClient(cfg.mocap.server, on_frame=log).start()
    start = time.time()
    print(f"Motive {cfg.mocap.server} (NatNet {'.'.join(map(str, mocap.natnet_version))}), racer {racer_id}, gate {gate_id}")
    print(f"logging to {out / 'gate_pose.csv'}  (Ctrl+C to stop)")

    warned = False
    try:
        while opts.duration is None or time.time() - start < opts.duration:
            time.sleep(1.0 / cfg.gate_log.status_rate)
            if not warned and time.time() - start > 2.0:
                missing = [b for b in (racer_id, gate_id) if b not in log.seen]
                if missing:
                    print(f"\nWARNING: body {missing} not in the stream; streaming IDs: {sorted(log.seen)}")
                warned = True
            print("\r" + status(log) + "\033[K", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mocap.stop()
        log.close()
    elapsed = time.time() - start
    n = max(log.frames, 1)
    meta = {"racer_body": racer_id, "gate_body": gate_id, "server": cfg.mocap.server,
            "natnet_version": ".".join(map(str, mocap.natnet_version)),
            "start": datetime.datetime.fromtimestamp(start).isoformat(), "duration_s": f"{elapsed:.2f}",
            "frames": log.frames, "rate_hz": f"{log.frames / max(elapsed, 1e-9):.1f}",
            "racer_tracked": f"{log.racer_frames / n:.3f}", "gate_tracked": f"{log.gate_frames / n:.3f}",
            "both_tracked": f"{log.both_frames / n:.3f}",
            "t_unix": f"wall-clock exposure time = arrival - mocap.latency ({cfg.mocap.latency} s)",
            "rel": "gate pivot in the racer body frame; rel_q = gate -> racer body; angles ZYX deg"}
    (out / "meta.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n")
    print(f"\n{log.frames} frames in {elapsed:.1f} s, both tracked {log.both_frames / n:.1%} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
