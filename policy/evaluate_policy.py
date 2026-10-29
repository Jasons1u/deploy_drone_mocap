#!/usr/bin/env python3
"""Compare exported policies in the desktop sim: parking near targets, attacks, a recorded pinch, hover calm, stand-off."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")   # tiny matrices: multithreaded BLAS only burns CPU (10x slower here)
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import load, repo_path  # noqa: E402
from cfdeploy.mlp_policy import load_policy  # noqa: E402
from cfdeploy.sim import Sim  # noqa: E402

TESTS = ("parking", "attack", "pinch", "hover", "standoff")
PINCH_LOG = "crazyflie/logs/20261006_174617_policy"   # obstacles 25 + 26 hand-moved around the hovering drone
HOLD = np.array([0.0, 0.0, 1.0])
P2P_START, P2P_TARGET = np.array([0.1, -0.2, 0.62]), np.array([1.0, -0.2, 0.62])


# #### // Sim helpers // ####
def make_sim(path, start, obstacles) -> Sim:
    cfg = load()
    policy, _, _ = load_policy(path)
    cfg.sim.start, cfg.sim.obstacles = [float(x) for x in start], [[float(x) for x in o] for o in obstacles]
    cfg.sim.obstacle_max_speed = policy.max_obstacle_speed()
    with contextlib.redirect_stdout(io.StringIO()):
        return Sim(cfg, policy)


def run(sim, seconds, each=None) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        for k in range(int(seconds * 100)):
            if each:
                each(sim, k)
            sim.tick()


def commands(sim, t0, t1) -> tuple:
    """Mean |change| of the velocity command per step, % sign flips (both sides > 0.02 m/s), max |command|."""
    c = np.array([r[16:19] for r in sim.rows if t0 <= r[0] < t1])
    flips = (np.sign(c[1:]) != np.sign(c[:-1])) & (np.abs(c[1:]) > 0.02) & (np.abs(c[:-1]) > 0.02)
    return np.abs(np.diff(c, axis=0)).mean(), flips.mean() * 100, np.abs(c).max()


def distance_to(sim, point, t0, t1) -> np.ndarray:
    return np.linalg.norm(np.array([r[1:4] for r in sim.rows if t0 <= r[0] < t1]) - point, axis=1)


# #### // The tests: each returns {condition: text} for one policy // ####
def parking(path) -> dict:
    """Hover at the start, then a target 0.9 m away with a static obstacle d short of it on the path; error at 16 s."""
    out = {}
    for d in (0.35, 0.45, 0.6):
        sim = make_sim(path, P2P_START, [(P2P_TARGET[0] - d, -0.2, 0.62, 0.10)])
        run(sim, 16.0, lambda s, k: k == 200 and s.handle({"target": list(P2P_TARGET)}))
        out[f"obstacle {d:.2f} m short: error at 16 s"] = f"{np.linalg.norm(sim.model.p - P2P_TARGET) * 100:.0f} cm"
    return out


def attack(path) -> dict:
    """An obstacle flown straight through the hold point from 1.3 m away (starts at 2 s)."""
    out = {}
    for speed in (0.4, 1.0):
        sim = make_sim(path, HOLD, [(1.3, 0.0, 1.0, 0.10)])
        sim.obstacles[0].follow(lambda t, s=speed: (max(1.3 - s * max(0.0, t - 2.0), -1.3), 0.0, 1.0))
        run(sim, 14.0)
        passed = 2.0 + 2.6 / speed
        back = [r[0] for r in sim.rows if r[0] > passed and np.linalg.norm(np.array(r[1:4]) - HOLD) < 0.10]
        out[f"{speed:.1f} m/s: min gap / max drift / back <10 cm"] = (
            f"{sim.min_gap * 100:.0f} cm / {distance_to(sim, HOLD, 0, 14).max() * 100:.0f} cm / "
            + (f"{back[0] - passed:.1f} s" if back else "never"))
    return out


def pinch(path, log) -> dict:
    """Every tracked obstacle of a real policy flight replayed exactly; the drone starts where and how that flight engaged."""
    log = repo_path(log)
    rows = [r for r in csv.reader(open(log / "control.csv"))][1:]
    engaged = next(r for r in rows if r[1] == "policy")
    t0, start, target = float(engaged[0]), np.array(engaged[2:5], float), np.array(engaged[8:11], float)
    seconds = float(next(r for r in reversed(rows) if r[1] == "policy")[0]) - t0
    raw = [r for r in csv.reader(open(log / "obstacles.csv"))][1:]
    tracks = {}
    for oid in sorted({r[2] for r in raw}):
        tracks[oid] = np.array([[float(r[0]) - t0, float(r[3]), float(r[4]), float(r[5])] for r in raw if r[2] == oid])
    first = [a[np.searchsorted(a[:, 0], 0.0), 1:] for a in tracks.values()]
    sim = make_sim(path, start, [(*f, 0.10) for f in first])
    sim.target.reset(target)
    for o, a in zip(sim.obstacles, tracks.values()):
        o.follow(lambda t, a=a: [np.interp(t, a[:, 0], a[:, k]) for k in (1, 2, 3)])
    run(sim, seconds)
    closest = [int(r[19]) for r in sim.rows]
    dcmd, flips, cmax = commands(sim, 0, seconds)
    err = distance_to(sim, target, 0, seconds)
    return {f"{log.name} ({seconds:.0f} s): min gap / collisions": f"{sim.min_gap * 100:.0f} cm / {sim.collisions}",
            "closest-obstacle switches": f"{sum(a != b for a, b in zip(closest, closest[1:])) / seconds:.2f}/s",
            "|cmd change| per step / sign flips / max |cmd|": f"{dcmd:.3f} / {flips:.1f}% / {cmax:.2f}",
            "distance from target median / max": f"{np.median(err) * 100:.0f} / {err.max() * 100:.0f} cm"}


def hover(path) -> dict:
    """Holding with a static obstacle 0.8 m away, 3-13 s."""
    sim = make_sim(path, HOLD, [(0.8, 0.0, 1.0, 0.10)])
    run(sim, 13.0)
    dcmd, flips, cmax = commands(sim, 3, 13)
    return {"|cmd change| per step / sign flips": f"{dcmd:.4f} / {flips:.1f}%",
            "max |cmd| / mean hold error": f"{cmax:.3f} / {distance_to(sim, HOLD, 3, 13).mean() * 100:.1f} cm"}


def standoff(path) -> dict:
    """Mean hold offset (6-13 s) with a static obstacle D away along +x; the vector shows which way it backs off."""
    out = {}
    for d in (0.6, 0.8, 1.0, 1.3):
        sim = make_sim(path, HOLD, [(d, 0.0, 1.0, 0.10)])
        run(sim, 13.0)
        m = (np.array([r[1:4] for r in sim.rows if 6 <= r[0] < 13]) - HOLD).mean(axis=0) * 100
        out[f"obstacle at {d:.1f} m: offset (x, y, z)"] = f"{np.linalg.norm(m):.0f} cm ({m[0]:+.0f}, {m[1]:+.0f}, {m[2]:+.0f})"
    return out


# #### // CLI: one table per test, one column per policy // ####
def table(title, labels, results) -> None:
    rows = list(results[0])
    w0 = max(len(r) for r in rows)
    widths = [max(len(lab), *(len(res[r]) for r in rows)) for lab, res in zip(labels, results)]
    print(f"\n{title}")
    print("  " + " " * w0 + "  " + "  ".join(lab.rjust(w) for lab, w in zip(labels, widths)))
    for r in rows:
        print("  " + r.ljust(w0) + "  " + "  ".join(res[r].rjust(w) for res, w in zip(results, widths)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("policies", nargs="+", type=Path, help="exported .npz files to compare")
    ap.add_argument("--label", action="append", help="column name per policy (default: file name)")
    ap.add_argument("--tests", default=",".join(TESTS), help=f"comma list of {', '.join(TESTS)}")
    ap.add_argument("--pinch-log", default=PINCH_LOG, help="policy-flight log whose obstacles the pinch test replays")
    args = ap.parse_args(argv)
    labels = args.label or [p.stem for p in args.policies]
    if len(labels) != len(args.policies):
        ap.error("give one --label per policy")
    paths = [str(p.resolve()) for p in args.policies]
    for name in [t.strip() for t in args.tests.split(",") if t.strip()]:
        if name not in TESTS:
            ap.error(f"unknown test {name} (choose from {', '.join(TESTS)})")
        fn = {"parking": parking, "attack": attack, "hover": hover, "standoff": standoff,
              "pinch": lambda p: pinch(p, args.pinch_log)}[name]
        table(f"{name}: {(pinch if name == 'pinch' else fn).__doc__}", labels, [fn(p) for p in paths])
    return 0


if __name__ == "__main__":
    sys.exit(main())
