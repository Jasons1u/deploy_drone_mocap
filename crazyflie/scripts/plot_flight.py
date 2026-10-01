#!/usr/bin/env python3
"""Plot a cf-hover log (flight or --filter-test).

    python3 scripts/plot_flight.py                 # newest log in logs/
    python3 scripts/plot_flight.py logs/hover_20260928_101500.csv

Rows: x/y/z (setpoint, mocap, Kalman) | roll/pitch/yaw (mocap vs Kalman) | motor PWM + battery.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def load(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path.name} has no data rows")
    return {k: [r[k] if k == "phase" else float(r[k]) for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("log", nargs="?", type=Path)
    args = ap.parse_args()
    path = args.log or max(LOG_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    d = load(path)
    t = d["t"]
    flying = not all(math.isnan(v) for v in d["sp_z"])

    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(11, 13))
    fig.suptitle(path.name)
    for ax, axis in zip(axes[:3], "xyz"):
        if flying:
            ax.plot(t, d[f"sp_{axis}"], "k--", lw=1, label="setpoint")
        ax.plot(t, d[f"mocap_{axis}"], lw=1.5, label="mocap")
        ax.plot(t, d[f"est_{axis}"], lw=1, label="Kalman")
        ax.set_ylabel(f"{axis} [m]")
    for ax, angle in zip(axes[3:6], ("roll", "pitch", "yaw")):
        ax.plot(t, d[f"mocap_{angle}_deg"], lw=1.5, label="mocap")
        ax.plot(t, d[f"est_{angle}_deg"], lw=1, label="Kalman")
        ax.set_ylabel(f"{angle} [deg]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[3].legend(loc="upper right", fontsize=8)

    ax = axes[6]
    for i in range(1, 5):
        ax.plot(t, d[f"m{i}"], lw=1, label=f"m{i}")
    ax.set_ylim(0, 66000)
    ax.set_ylabel("motor PWM")
    ax.set_xlabel("t [s]")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    bat = ax.twinx()
    bat.plot(t, d["battery_v"], "k:", lw=1)
    bat.set_ylabel("battery [V]")

    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
