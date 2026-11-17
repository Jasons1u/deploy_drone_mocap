#!/usr/bin/env python3
"""Fit the fixed camera and gate-centre transforms from mocap + vision logs (M Y = X V), then report the vision pose error."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import load, repo_path  # noqa: E402

VISION_COLUMNS = ["t_unix", "x", "y", "z", "qw", "qx", "qy", "qz"]


def to_rot(q_wxyz) -> Rotation:
    q = np.asarray(q_wxyz, float).reshape(-1, 4)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def to_wxyz(r: Rotation) -> np.ndarray:
    return r.as_quat()[..., [3, 0, 1, 2]]


def so3(m) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def rms(a) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(a), axis=1)))) if len(a) else float("nan")


# #### // Logs // ####
def load_mocap(path) -> dict:
    """gate_pose.csv frames with both bodies tracked: t_unix and the gate pivot pose in the racer pivot frame (M)."""
    path = Path(path)
    path = path / "gate_pose.csv" if path.is_dir() else path
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["racer_ok"] == "1" and r["gate_ok"] == "1"]
    t = np.array([float(r["t_unix"]) for r in rows])
    p = np.array([[float(r[f"rel_{k}"]) for k in "xyz"] for r in rows]).reshape(-1, 3)
    q = np.array([[float(r[f"rel_q{k}"]) for k in "wxyz"] for r in rows]).reshape(-1, 4)
    return {"t": t, "p": p, "q": q}


def load_vision(path) -> dict:
    """Vision CSV (t_unix, x, y, z [m], qw, qx, qy, qz: gate pose in the camera frame); rows with NaN (no detection) skipped."""
    with open(path, newline="") as f:
        a = np.array([[float(r[k]) for k in VISION_COLUMNS] for r in csv.DictReader(f)]).reshape(-1, 8)
    a = a[np.all(np.isfinite(a), axis=1)]
    return {"t": a[:, 0], "p": a[:, 1:4], "q": a[:, 4:8] / np.linalg.norm(a[:, 4:8], axis=1, keepdims=True)}


def match(mocap: dict, vision: dict, offset: float, max_gap: float) -> dict:
    """Mocap M interpolated at each vision time (+offset, onto the mocap clock); vision samples in mocap gaps dropped."""
    tm, t = mocap["t"], vision["t"] + offset
    ok = np.zeros(len(t), bool)
    if len(tm) >= 2:
        i = np.clip(np.searchsorted(tm, t), 1, len(tm) - 1)
        ok = (t >= tm[i - 1]) & (t <= tm[i]) & (tm[i] - tm[i - 1] <= max_gap)
    if not ok.any():
        return {"t": np.zeros(0), "RM": np.zeros((0, 3, 3)), "tM": np.zeros((0, 3)), "RV": np.zeros((0, 3, 3)), "tV": np.zeros((0, 3))}
    i, t = i[ok], t[ok]
    a = (t - tm[i - 1]) / np.maximum(tm[i] - tm[i - 1], 1e-9)
    r0, r1 = to_rot(mocap["q"][i - 1]), to_rot(mocap["q"][i])
    rm = r0 * Rotation.from_rotvec((r0.inv() * r1).as_rotvec() * a[:, None])
    tM = mocap["p"][i - 1] + a[:, None] * (mocap["p"][i] - mocap["p"][i - 1])
    return {"t": t, "RM": rm.as_matrix(), "tM": tM, "RV": to_rot(vision["q"][ok]).as_matrix(), "tV": vision["p"][ok]}


def matched(runs: list, offset: float, max_gap: float) -> dict:
    """match() over every (name, mocap, vision) run, concatenated, with a run name per sample."""
    parts = [(name, match(m, v, offset, max_gap)) for name, m, v in runs]
    d = {k: np.concatenate([p[k] for _, p in parts]) for k in ("t", "RM", "tM", "RV", "tV")}
    d["run"] = np.concatenate([[name] * len(p["t"]) for name, p in parts]).astype(str)
    return d


def require(d: dict, runs: list, offset: float) -> None:
    if len(d["t"]) >= 10:
        return
    def span(t):
        return f"{t.min():.3f}..{t.max():.3f}" if len(t) else "empty"
    for name, m, v in runs:
        print(f"  {name}: mocap t_unix {span(m['t'])}, vision {span(v['t'] + offset)} (offset {offset:+.3f} s)")
    raise SystemExit(f"only {len(d['t'])} vision samples fall inside tracked mocap: clocks or --time-offset off?")


# #### // Fit: M Y = X V, X = camera in the racer pivot frame, Y = vision gate frame in the gate pivot frame // ####
def init_fit(d: dict) -> tuple:
    """Closed form: rotations from the null space of R_M R_Y = R_X R_V, then translations by linear least squares."""
    RM, RV, eye = d["RM"], d["RV"], np.eye(3)
    n = len(RM)
    K = np.concatenate([np.einsum("ac,nbd->nabcd", eye, RM).reshape(n, 9, 9),
                        -np.einsum("nca,bd->nabcd", RV, eye).reshape(n, 9, 9)], axis=2)
    v = np.linalg.eigh(np.einsum("nij,nik->jk", K, K))[1][:, 0]
    Ry, Rx = v[:9].reshape(3, 3, order="F"), v[9:].reshape(3, 3, order="F")
    s = np.cbrt(np.linalg.det(Rx))
    Rx, Ry = so3(Rx / s), so3(Ry / s)
    A = np.concatenate([RM, -np.broadcast_to(eye, RM.shape)], axis=2).reshape(-1, 6)
    sol = np.linalg.lstsq(A, (np.einsum("ij,nj->ni", Rx, d["tV"]) - d["tM"]).ravel(), rcond=None)[0]
    return (Rx, sol[3:]), (Ry, sol[:3])


def errors(X, Y, d: dict) -> tuple:
    """Vision minus truth (X^-1 M Y): position error in the camera frame [m], rotation error vectors [rad], truth position."""
    (Rx, tX), (Ry, tY) = X, Y
    RVh = np.einsum("ji,njk,kl->nil", Rx, d["RM"], Ry)
    tVh = np.einsum("ji,nj->ni", Rx, np.einsum("nij,j->ni", d["RM"], tY) + d["tM"] - tX)
    er = Rotation.from_matrix(np.einsum("nji,njk->nik", RVh, d["RV"])).as_rotvec()
    return d["tV"] - tVh, er, tVh


def refine(X, Y, d: dict) -> tuple:
    """Nonlinear least squares on position + weighted rotation error around the closed form (soft-L1 for bad detections)."""
    ep, er, _ = errors(X, Y, d)
    w = float(np.clip(rms(ep) / max(rms(er), 1e-9), 0.01, 1.0))   # [m/rad] balances position and rotation residuals

    def unpack(x):
        return ((X[0] @ Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]),
                (Y[0] @ Rotation.from_rotvec(x[6:9]).as_matrix(), x[9:12]))

    def res(x):
        ep, er, _ = errors(*unpack(x), d)
        return np.concatenate([ep, w * er], axis=1).ravel()

    x0 = np.concatenate([np.zeros(3), X[1], np.zeros(3), Y[1]])
    scale = max(2.0 * float(np.median(np.abs(res(x0)))), 1e-4)
    return unpack(least_squares(res, x0, loss="soft_l1", f_scale=scale).x)


def scan_offset(runs: list, centre: float, half: float, max_gap: float) -> tuple:
    """Time offset in centre +- half (1 ms steps) with the lowest closed-form position rms; (best, rms at centre, rms at best)."""
    best, at_centre, best_rms = centre, float("nan"), float("inf")
    for off in centre + np.arange(-half, half + 1e-9, 0.001):
        d = matched(runs, off, max_gap)
        if len(d["t"]) < 10:
            continue
        r = rms(errors(*init_fit(d), d)[0])
        at_centre = r if abs(off - centre) < 5e-4 else at_centre
        if r < best_rms:
            best, best_rms = float(off), r
    return best, at_centre, best_rms


# #### // Report // ####
def observability(X, Y, d: dict) -> list:
    """Warnings when the calibration motion leaves the camera vs gate-centre split loose (too little relative tilt)."""
    warn = []
    tilt = np.degrees(np.arccos(np.clip(d["RM"][:, 2, 2], -1.0, 1.0)))
    print(f"  racer tilt vs gate z: median {np.median(tilt):.1f} deg, p90 {np.percentile(tilt, 90):.1f} deg, max {tilt.max():.1f} deg")
    A = np.concatenate([d["RM"], -np.broadcast_to(np.eye(3), d["RM"].shape)], axis=2).reshape(-1, 6)
    sigma = rms(errors(X, Y, d)[0]) / np.sqrt(3) / np.linalg.svd(A, compute_uv=False)[-1]
    print(f"  weakest camera/gate-centre translation split: +-{sigma * 1000:.1f} mm (1 sigma)")
    if np.percentile(tilt, 90) < 10.0:
        warn.append("calibration racer barely tilts relative to the gate: the camera vs gate-centre split (yaw, height) is loose; tilt it 20-30 deg in roll and pitch")
    if sigma > 0.005:
        warn.append(f"translation split uncertain to {sigma * 1000:.0f} mm: more varied angles and positions around the gate")
    return warn


def describe(name: str, T) -> None:
    R, t = T
    ypr = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    print(f"  {name}: t [{t[0] * 1000:+.1f} {t[1] * 1000:+.1f} {t[2] * 1000:+.1f}] mm  "
          f"q(wxyz) [{' '.join(f'{c:+.4f}' for c in to_wxyz(Rotation.from_matrix(R)))}]  "
          f"yaw/pitch/roll [{ypr[0]:+.1f} {ypr[1]:+.1f} {ypr[2]:+.1f}] deg")


def report(name: str, ep, er, truth) -> None:
    en, ed, dist = np.linalg.norm(ep, axis=1) * 1000, np.degrees(np.linalg.norm(er, axis=1)), np.linalg.norm(truth, axis=1)
    b, s = np.mean(ep, axis=0) * 1000, np.sqrt(np.mean(ep ** 2, axis=0)) * 1000
    print(f"\n{name}: {len(en)} samples, distance {dist.min():.2f}-{dist.max():.2f} m")
    print(f"  position error (camera frame) mean [{b[0]:+.1f} {b[1]:+.1f} {b[2]:+.1f}] mm, rms [{s[0]:.1f} {s[1]:.1f} {s[2]:.1f}] mm")
    print(f"  |position error| median {np.median(en):.1f}  rms {np.sqrt(np.mean(en ** 2)):.1f}  p95 {np.percentile(en, 95):.1f}  max {en.max():.1f} mm")
    print(f"  rotation error   median {np.median(ed):.2f}  rms {np.sqrt(np.mean(ed ** 2)):.2f}  p95 {np.percentile(ed, 95):.2f}  max {ed.max():.2f} deg")
    print("  by distance:  range [m]      n   |pos| median/rms [mm]   rot median [deg]")
    for lo in np.arange(np.floor(dist.min() * 2) / 2, dist.max(), 0.5):
        m = (dist >= lo) & (dist < lo + 0.5)
        if m.any():
            print(f"               {lo:.1f}-{lo + 0.5:.1f}  {m.sum():6d}   {np.median(en[m]):7.1f} / {np.sqrt(np.mean(en[m] ** 2)):7.1f}"
                  f"        {np.median(ed[m]):6.2f}")


def save_transform(T) -> dict:
    return {"t": T[1].tolist(), "q_wxyz": to_wxyz(Rotation.from_matrix(T[0])).tolist()}


def load_transform(j) -> tuple:
    return to_rot(j["q_wxyz"]).as_matrix()[0], np.array(j["t"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", nargs=2, action="append", default=[], metavar=("MOCAP_RUN", "VISION_CSV"),
                    help="calibration run: log_gate.py folder (or its gate_pose.csv) + the vision log of the same run; repeatable")
    ap.add_argument("--eval", nargs=2, action="append", default=[], metavar=("MOCAP_RUN", "VISION_CSV"),
                    help="held-out run to score with the fitted transforms; repeatable")
    ap.add_argument("--load", metavar="CALIB_JSON", help="transforms from an earlier fit instead of --calib")
    ap.add_argument("--time-offset", type=float, help="[s] added to vision timestamps to put them on the mocap clock (default 0, or the loaded one)")
    ap.add_argument("--scan-offset", type=float, metavar="SECONDS", help="pick the time offset within +- this of --time-offset that fits the calibration best")
    ap.add_argument("--max-gap", type=float, default=0.02, help="[s] a vision sample inside a longer mocap dropout is skipped")
    ap.add_argument("--out", help="output folder (default <gate_log.log_dir>/<time>_calib)")
    opts = ap.parse_args()
    if not opts.calib and not opts.load:
        ap.error("give --calib MOCAP_RUN VISION_CSV (or --load CALIB_JSON)")
    cfg = load()
    out = Path(opts.out) if opts.out else repo_path(cfg.gate_log.log_dir) / (time.strftime("%Y%m%d_%H%M%S") + "_calib")
    out.mkdir(parents=True, exist_ok=True)

    def runs(pairs):
        return [(Path(m).name if Path(m).is_dir() else Path(m).parent.name, load_mocap(m), load_vision(v)) for m, v in pairs]

    calib_runs, eval_runs = runs(opts.calib), runs(opts.eval)
    saved = json.loads(Path(opts.load).read_text()) if opts.load else None
    offset = opts.time_offset if opts.time_offset is not None else (saved["time_offset"] if saved else 0.0)
    if opts.scan_offset:
        if not calib_runs:
            ap.error("--scan-offset needs --calib runs")
        offset, at_centre, best = scan_offset(calib_runs, offset, opts.scan_offset, opts.max_gap)
        print(f"time offset {offset * 1000:+.0f} ms (closed-form position rms {best * 1000:.1f} mm, {at_centre * 1000:.1f} mm without the scan)")

    rows, warn = [], []
    if saved:
        X, Y = load_transform(saved["X_racer_to_camera"]), load_transform(saved["Y_gate_to_vision_gate"])
        print(f"transforms from {opts.load}")
    if calib_runs:
        dc = matched(calib_runs, offset, opts.max_gap)
        require(dc, calib_runs, offset)
        if not saved:
            X, Y = refine(*init_fit(dc), dc)
            print("fit (M Y = X V):")
        describe("X camera in racer pivot frame      ", X)
        describe("Y vision gate frame in gate pivot  ", Y)
        warn += observability(X, Y, dc)
        ep, er, truth = errors(X, Y, dc)
        report("calibration runs (in-sample: a constant vision bias is absorbed into X)", ep, er, truth)
        rows += [("calib", dc, ep, er, truth)]
    if eval_runs:
        de = matched(eval_runs, offset, opts.max_gap)
        require(de, eval_runs, offset)
        ep, er, truth = errors(X, Y, de)
        report("held-out runs", ep, er, truth)
        rows += [("eval", de, ep, er, truth)]

    (out / "calib.json").write_text(json.dumps({
        "X_racer_to_camera": save_transform(X), "Y_gate_to_vision_gate": save_transform(Y), "time_offset": offset,
        "calib_runs": [list(p) for p in opts.calib] or (saved or {}).get("calib_runs", []),
        "model": "M Y = X V; M = gate pivot in racer pivot frame (mocap), V = vision gate pose in camera frame"}, indent=2) + "\n")
    with open(out / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "run", "t_unix", "truth_x", "truth_y", "truth_z", "vision_x", "vision_y", "vision_z",
                    "ex_mm", "ey_mm", "ez_mm", "e_pos_mm", "e_rot_deg"])
        for name, d, ep, er, truth in rows:
            for k in range(len(ep)):
                w.writerow([name, d["run"][k], f"{d['t'][k]:.6f}", *truth[k].round(6), *d["tV"][k].round(6),
                            *(ep[k] * 1000).round(3), round(float(np.linalg.norm(ep[k]) * 1000), 3),
                            round(float(np.degrees(np.linalg.norm(er[k]))), 4)])
    for w_ in warn:
        print(f"WARNING: {w_}")
    print(f"\n-> {out / 'calib.json'}, {out / 'errors.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
