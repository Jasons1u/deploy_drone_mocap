"""Run logs under crazyflie/logs/<time>[_tag]/: mocap, telemetry blocks, setpoints and a meta file (fit_sysid.py reads these)."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from cfdeploy.config import repo_path
from cfdeploy.radio import LOG_BLOCKS


def run_dir(cfg, tag: str = "") -> Path:
    return repo_path(cfg.paths.log_dir) / (time.strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else ""))


def write_csv(path: Path, header, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_flight_logs(out: Path, mocap_rows, tel_rows: dict, sp_rows, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "mocap.csv", ["t", "frame", "x", "y", "z", "qw", "qx", "qy", "qz"], mocap_rows)
    for name, variables in LOG_BLOCKS.items():
        write_csv(out / f"cf_{name}.csv", ["t"] + [v for v, _ in variables] + ["cf_ms"], tel_rows[name])
    write_csv(out / "setpoints.csv", ["t", "kind", "seg", "mode", "x", "y", "z", "vx", "vy", "vz"], sp_rows)
    (out / "meta.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n")


def read_rows(path: Path) -> list:
    """Every row of a CSV (header first), cells as text."""
    with open(path, newline="") as f:
        return list(csv.reader(f))


def read_meta(run: Path) -> dict:
    """meta.txt as {key: value text}; empty if missing."""
    path = run / "meta.txt"
    return dict(line.partition(": ")[::2] for line in path.read_text().splitlines()) if path.exists() else {}


def obstacle_tracks(run: Path) -> dict:
    """obstacles.csv per obstacle id: rows (t, x, y, z, vx, vy, vz)."""
    tracks = {}
    for r in read_rows(run / "obstacles.csv")[1:]:
        tracks.setdefault(int(r[2]), []).append([float(x) for x in (r[0], *r[3:9])])
    return {oid: np.array(rows) for oid, rows in sorted(tracks.items())}


def save_link_logs(out: Path, stats, sends) -> None:
    """link.csv: cflib link statistics over time; extpose.csv: each pose sent and how long cflib blocked on it."""
    write_csv(out / "link.csv", ["t", "stat", "value"], stats)
    write_csv(out / "extpose.csv", ["t", "frame", "send_s"], sends)
