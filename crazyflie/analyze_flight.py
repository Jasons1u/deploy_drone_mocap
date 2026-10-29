#!/usr/bin/env python3
"""Flight-log diagnosis: how each launch went, and whether the radio delayed telemetry (logs from fly_policy / sysid_flight)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

CORNERS = {"right": "M1/M2 side", "left": "M3/M4 side", "front": "M1/M4 side", "back": "M2/M3 side"}


# #### // Log loading // ####
def load(path: Path):
    """(header, float array) of a CSV; empty or text cells become nan, short rows are padded; None if missing or empty."""
    if not path.exists():
        return None, None
    rows = list(csv.reader(open(path)))
    if len(rows) < 2:
        return rows[0] if rows else None, None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return math.nan
    width = max(len(r) for r in rows)
    return rows[0], np.array([[num(x) for x in r] + [math.nan] * (width - len(r)) for r in rows[1:]])


def tilt(q) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))


def outcome(run: Path) -> str:
    _, ev = load(run / "events.csv")
    if ev is not None:
        return list(csv.reader(open(run / "events.csv")))[-1][-1]
    meta = (run / "meta.txt").read_text() if (run / "meta.txt").exists() else ""
    return next((line[9:] for line in meta.splitlines() if line.startswith("outcome: ")), "?")


# #### // Launch: burst length, lift-off, tilt, which side dropped, battery sag // ####
def launch_report(run: Path) -> dict | None:
    if (run / "control.csv").exists():
        rows = list(csv.reader(open(run / "control.csv")))[1:]
        burst_t = [float(r[0]) for r in rows if r[1] == "launch"]
        ev = list(csv.reader(open(run / "events.csv")))[1:] if (run / "events.csv").exists() else []
        lift = [float(e[0]) for e in ev if e[2].startswith("lift-off")]
    elif (run / "setpoints.csv").exists():
        rows = list(csv.reader(open(run / "setpoints.csv")))[1:]
        burst_t = [float(r[0]) for r in rows if r[3] == "thrust"]
        lift = [float(r[0]) for r in rows if r[3] != "thrust" and burst_t and float(r[0]) > burst_t[-1]][:1]
    else:
        return None
    if not burst_t:
        return None
    t0, t1 = burst_t[0], burst_t[-1]
    out = dict(burst=(lift[0] if lift else t1) - t0, lifted=bool(lift), outcome=outcome(run))
    _, m = load(run / "mocap.csv")
    if m is not None:
        w = (m[:, 0] >= t0) & (m[:, 0] <= t0 + 1.5)
        out["max_tilt"] = max((tilt(r[5:9]) for r in m[w]), default=math.nan)
    hv, vel = load(run / "cf_vel.csv")
    hm, mot = load(run / "cf_mot.csv")
    if vel is not None and mot is not None:
        ir, ip = hv.index("stabilizer.roll"), hv.index("stabilizer.pitch")
        w = (vel[:, 0] >= t0) & (vel[:, 0] <= t1 + 0.05)
        imu = np.hypot(vel[w, ir], vel[w, ip])
        if (imu > 20).any():
            k = int(np.argmax(imu > 20))
            t, roll, pitch = vel[w, 0][k], vel[w, ir][k], vel[w, ip][k]   # legacy: roll > 0 = right down, pitch > 0 = nose up
            side = ("right" if roll > 0 else "left") if abs(roll) >= abs(pitch) else ("front" if pitch < 0 else "back")
            motors = [np.interp(t, mot[:, 0], mot[:, i]) / 655.35 for i in range(1, 5)]
            out["tipped"] = f"{side} down ({CORNERS[side]}) at {t - t0:.2f} s, motors " + " ".join(f"{x:.0f}" for x in motors) + " %"
    ha, acc = load(run / "cf_acc.csv")
    if acc is not None:
        j = ha.index("pm.vbat")
        before, during = acc[acc[:, 0] < t0], acc[(acc[:, 0] >= t0) & (acc[:, 0] <= t1 + 0.1)]
        if len(before) and len(during):
            out["vbat"] = f"{before[-1, j]:.2f} -> {during[:, j].min():.2f} V"
    return out


# #### // Radio: downlink delay of telemetry, link statistics, pose sends // ####
def lag(t_tel, y_tel, t_ref, y_ref, t0, t1, min_swing, sign=1.0):
    """Best-fit lag [ms] of a telemetry signal behind a reference over [t0, t1]; None without enough signal or at the search edge."""
    w = (t_tel >= t0) & (t_tel <= t1)
    r = (t_ref >= t0) & (t_ref <= t1)
    if w.sum() < 20 or r.sum() < 20 or np.ptp(y_ref[r]) < min_swing:
        return None
    lags = np.arange(-0.05, 0.4, 0.0025)
    errs = [np.sqrt(np.mean((sign * y_tel[w] - np.interp(t_tel[w] - L, t_ref, y_ref)) ** 2)) for L in lags]
    i = int(np.argmin(errs))
    return None if i in (0, len(lags) - 1) else lags[i] * 1e3


def policy_end(run: Path, fallback: float) -> float:
    """Time of the last policy-mode row (where the runner compared telemetry with mocap), else fallback."""
    if not (run / "control.csv").exists():
        return fallback
    t = [float(r[0]) for r in list(csv.reader(open(run / "control.csv")))[1:] if r[1] == "policy"]
    return t[-1] if t else fallback


def radio_report(run: Path) -> list:
    lines = []
    hp, pos = load(run / "cf_pos.csv")
    if pos is not None and "cf_ms" in hp:
        gap = pos[:, 0] - pos[:, hp.index("cf_ms")] / 1000.0
        delay = (gap - gap.min()) * 1e3   # beyond the fastest packet (the clock offset); wrap-free over a flight
        worst = max(((delay[(pos[:, 0] >= a) & (pos[:, 0] < a + 1.0)].mean(), a - pos[0, 0]) for a in pos[:: 25, 0]
                     if a + 1.0 <= pos[-1, 0]), default=(math.nan, math.nan))
        lines.append(f"telemetry delay (drone timestamps): median {np.median(delay):.0f} ms, p95 {np.percentile(delay, 95):.0f}, "
                     f"max {delay.max():.0f}; worst 1 s window {worst[0]:.0f} ms at +{worst[1]:.1f} s; last 1 s "
                     f"{delay[pos[:, 0] > pos[-1, 0] - 1.0].mean():.0f} ms")
    else:
        _, m = load(run / "mocap.csv")
        hv, vel = load(run / "cf_vel.csv")
        if m is not None and vel is not None:
            qw, qx, qy, qz = m[:, 5], m[:, 6], m[:, 7], m[:, 8]
            roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
            end = policy_end(run, vel[-1, 0])
            est = lag(vel[:, 0], vel[:, hv.index("stabilizer.roll")], m[:, 0], roll, end - 1.0, end, min_swing=3.0)
            lines.append("old log (no drone timestamps): IMU roll lags mocap by "
                         + ("? (not enough attitude motion)" if est is None else f"{est:.0f} ms")
                         + " in the last 1 s of policy flight (a downlink delay shows here)")
    hc, ctl = load(run / "control.csv")
    hr, ref = load(run / "cf_ref.csv")
    if ctl is not None and ref is not None and "cmd_vx" in hc:
        tc, cmd = ctl[:, 0], ctl[:, hc.index("cmd_vx")]
        ok = ~np.isnan(cmd)
        if ok.sum() > 50:
            end = policy_end(run, ref[-1, 0])
            rt = [lag(ref[:, 0], ref[:, 1], tc[ok], cmd[ok], a, a + 0.5, min_swing=0.1) for a in (end - 3.0, end - 0.5)]
            lines.append("command round trip (host cmd -> onboard vRef -> host): " + ", ".join(
                f"{label} " + ("? (command too steady)" if x is None else f"{x:.0f} ms")
                for label, x in zip(("3 s before the policy ended", "its last 0.5 s"), rt)))
    hl, _ = load(run / "link.csv")
    if hl is not None and (run / "link.csv").stat().st_size > 20:
        stats = {}
        for r in list(csv.reader(open(run / "link.csv")))[1:]:
            try:
                stats.setdefault(r[1], []).append(float(r[2]))
            except ValueError:
                pass
        lines.append("link: " + "; ".join(f"{k} min {min(v):.3g} mean {np.mean(v):.3g} max {max(v):.3g}" for k, v in sorted(stats.items())))
    he, sends = load(run / "extpose.csv")
    if sends is not None:
        dur = sends[-1, 0] - sends[0, 0]
        meta = (run / "meta.txt").read_text() if (run / "meta.txt").exists() else ""
        skipped = next((line.split(": ")[1] for line in meta.splitlines() if line.startswith("extpose_skipped")), "?")
        lines.append(f"poses sent {len(sends) / max(dur, 1e-6):.0f}/s (skipped {skipped}); send blocked median "
                     f"{np.median(sends[:, 2]) * 1e3:.1f} ms, p95 {np.percentile(sends[:, 2], 95) * 1e3:.1f}, max {sends[:, 2].max() * 1e3:.1f}")
    return lines or ["no radio data in this log"]


# #### // CLI // ####
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="log folders (crazyflie/logs/<time>[_policy])")
    ap.add_argument("--only", choices=("launch", "radio"), help="one report only")
    args = ap.parse_args(argv)
    runs = [r for r in args.runs if r.is_dir()]
    if args.only != "radio":
        print(f"{'launch':26s}{'burst':8s}{'lift-off':10s}{'max tilt':11s}{'battery':18s}{'tipped':58s}outcome")
        fails = 0
        for run in runs:
            r = launch_report(run)
            if r is None:
                continue
            fails += not r["lifted"]
            print(f"  {run.name:24s}{r['burst']:4.2f} s  {'yes' if r['lifted'] else 'NO':10s}{r.get('max_tilt', math.nan):5.1f} deg  "
                  f"{r.get('vbat', '-'):18s}{r.get('tipped', '-'):58s}{r['outcome'][:45]}")
        print(f"  {fails} launch(es) never lifted off")
    if args.only != "launch":
        for run in runs:
            print(f"\nradio: {run.name}")
            for line in radio_report(run):
                print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
