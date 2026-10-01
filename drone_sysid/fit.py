"""Fit deploy-chain plant parameters from fly.py sysID CSVs and print the matching sim VelocityActionCfg values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from config import Config

CFG = Config()
KPV = np.array(CFG.kpv)
STR_COLS = {"phase", "seg_name", "kind"}
TAU_GRID = np.arange(0.03, 1.5, 0.005)
TD_GRID = np.arange(0.0, 0.125, 0.005)
AXES = "xyz"


def load(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"{path} is empty")
    return {k: np.array([r[k] for r in rows]) if k in STR_COLS else np.array([float(r[k]) for r in rows]) for k in rows[0]}


def segment_rows(d: dict) -> dict[int, np.ndarray]:
    """Row indices of each test segment (by segment index)."""
    out: dict[int, list] = {}
    for i, (ph, s) in enumerate(zip(d["phase"], d["seg"])):
        if ph == "test" and s >= 0:
            out.setdefault(int(s), []).append(i)
    return {k: np.array(v) for k, v in out.items()}


def time_window(d: dict, t0: float, t1: float) -> np.ndarray:
    return np.where((d["t"] >= t0) & (d["t"] <= t1) & (d["start_flag"] > 0))[0]


# --- classic fits ---
def fit_step(t: np.ndarray, v: np.ndarray) -> dict:
    """v = v0 before td, then v_ss + (v0 - v_ss) exp(-(t - td)/tau); grid over (td, tau), closed-form LS for (v_ss, v0)."""
    t = t - t[0]
    best = None
    for td in TD_GRID:
        e = np.exp(-np.clip(t - td, 0.0, None)[:, None] / TAU_GRID[None, :])
        a, b = 1.0 - e, e
        saa, sab, sbb = (a * a).sum(0), (a * b).sum(0), (b * b).sum(0)
        say, sby = (a * v[:, None]).sum(0), (b * v[:, None]).sum(0)
        det = saa * sbb - sab * sab
        ok = np.abs(det) > 1e-12
        vss = np.where(ok, (say * sbb - sby * sab) / np.where(ok, det, 1.0), 0.0)
        v0 = np.where(ok, (saa * sby - sab * say) / np.where(ok, det, 1.0), 0.0)
        cost = ((v[:, None] - a * vss - b * v0) ** 2).sum(0)
        cost[~ok] = np.inf
        j = int(np.argmin(cost))
        if best is None or cost[j] < best[0]:
            best = (cost[j], td, TAU_GRID[j], vss[j], v0[j])
    cost, td, tau, vss, v0 = best
    tot = float(((v - v.mean()) ** 2).sum())
    return {"td": float(td), "tau": float(tau), "v_ss": float(vss), "v0": float(v0), "r2": 1.0 - cost / tot if tot > 0 else float("nan")}


def step_fits(d: dict) -> list[dict]:
    out = []
    for s, idx in segment_rows(d).items():
        if d["kind"][idx[0]] != "vel":
            continue
        cmd = np.array([d["cmd_x"][idx[0]], d["cmd_y"][idx[0]], d["cmd_z"][idx[0]]])
        ax = int(np.argmax(np.abs(cmd)))
        f = fit_step(d["t"][idx], d[f"v_{AXES[ax]}"][idx])
        out.append({"seg": s, "axis": AXES[ax], "cmd": float(cmd[ax]), **f})
    return out


def pair_summary(fits: list[dict], min_r2: float = 0.8) -> dict:
    """Mirrored pulses: k = (v+ - v-)/2U cancels trim, b = (v+ + v-)/2 cancels gain."""
    res = {}
    for ax in AXES:
        good = [f for f in fits if f["axis"] == ax and f["r2"] >= min_r2]
        ks, bs = [], []
        for mag in sorted({round(abs(f["cmd"]), 3) for f in good}):
            pos = [f["v_ss"] for f in good if math.isclose(f["cmd"], mag, abs_tol=1e-6)]
            neg = [f["v_ss"] for f in good if math.isclose(f["cmd"], -mag, abs_tol=1e-6)]
            if pos and neg:
                ks.append((np.mean(pos) - np.mean(neg)) / (2.0 * mag))
                bs.append((np.mean(pos) + np.mean(neg)) / 2.0)
        if good:
            res[ax] = {"tau": float(np.median([f["tau"] for f in good])), "td": float(np.median([f["td"] for f in good])),
                       "k": float(np.mean(ks)) if ks else float("nan"), "trim_v": float(np.mean(bs)) if bs else float("nan"),
                       "n": len(good)}
    return res


def hover_stats(d: dict) -> dict | None:
    """Steady hover rows: mu_z there is the true hover crossing, whatever feedforward was flown."""
    flying = d["phase"] != "wait_pose"
    az = np.zeros_like(d["t"])
    az[flying] = np.convolve(np.gradient(d["v_z"][flying], d["t"][flying]), np.ones(49) / 49, mode="same")
    speed = np.linalg.norm(np.column_stack([d["v_x"], d["v_y"], d["v_z"]]), axis=1)
    m = ((d["phase"] == "test") & (d["kind"] == "hold") & (d["start_flag"] > 0) & (d["r_z"] > 0.3)
         & (speed < 0.05) & (np.abs(az) < 0.1))
    if m.sum() < 200:
        return None
    return {"hover_mu": float(d["mu_z"][m].mean()), "hover_mu_std": float(d["mu_z"][m].std()),
            "vbat": float(np.nanmean(d["vbat"][m])), "n": int(m.sum())}


def thrust_fits(d: dict) -> dict | None:
    pts = []
    for s, idx in segment_rows(d).items():
        if d["kind"][idx[0]] != "mu_open":
            continue
        t = d["t"][idx]
        use = idx[t >= t[0] + 0.1]
        if len(use) < 20 or d["t"][use[-1]] - d["t"][use[0]] < 0.15:
            continue
        a_z = float(np.polyfit(d["t"][use], d["v_z"][use], 1)[0])
        pts.append({"seg": s, "mu_z": float(d["mu_z"][use].mean()), "a_z": a_z, "settled": int(d["settled"][idx[0]]),
                    "vbat": float(np.nanmean(d["vbat"][use]))})
    if not pts:
        return None
    lin = [p for p in pts if p["mu_z"] <= 5.0]
    out = {"points": pts}
    if len(lin) >= 2:
        alpha, b = np.polyfit([p["mu_z"] for p in lin], [p["a_z"] for p in lin], 1)
        out.update(alpha_z=float(alpha), b_z=float(b), hover_mu=float(-b / alpha))
    sat = [p["a_z"] for p in pts if p["mu_z"] >= 6.0]
    if sat:
        out["a_z_max"] = float(np.mean(sat))
    return out


# --- model replay fit ---
def replay(t, mu, v_log, alpha, drag, b, delay, ceiling=None) -> np.ndarray:
    """Logged mu (sent at t) -> delayed -> a = alpha*mu + b (ceiling) - drag*v -> 6 Hz filter, as the controller sees v."""
    mu_d = np.interp(t - delay, t, mu)
    vt = vf = v_log[0]
    out = np.empty_like(v_log)
    out[0] = vf
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        a = alpha * mu_d[i - 1] + b
        if ceiling is not None:
            a = min(a, ceiling)
        vt += (a - drag * vt) * dt
        if CFG.dt_min <= dt <= CFG.dt_max:
            vf += (1.0 - math.exp(-2.0 * math.pi * CFG.vel_cutoff_hz * dt)) * (vt - vf)
        out[i] = vf
    return out


def replay_windows(d: dict, ax: int) -> list[np.ndarray]:
    """Pulse windows on this axis, thrust steps (z), and 2 s chunks of steady hover."""
    wins = []
    segs = segment_rows(d)
    for s, idx in segs.items():
        kind = d["kind"][idx[0]]
        if kind == "vel" and abs([d["cmd_x"], d["cmd_y"], d["cmd_z"]][ax][idx[0]]) > 0:
            end = segs.get(s + 2, segs.get(s + 1, idx))
            wins.append(time_window(d, d["t"][idx[0]] - 0.3, d["t"][end[-1]]))
        elif kind == "mu_open" and ax == 2:
            wins.append(time_window(d, d["t"][idx[0]] - 0.2, d["t"][idx[-1]] + 1.0))
        elif kind == "hold" and d["seg_name"][idx[0]] == "hover":
            for t0 in np.arange(d["t"][idx[0]], d["t"][idx[-1]] - 2.0, 2.0):
                wins.append(time_window(d, t0, t0 + 2.0))
    return [w for w in wins if len(w) > 50]


def fit_axis(datasets: list[dict], ax: int) -> dict | None:
    wins = [(d, w) for d in datasets for w in replay_windows(d, ax)]
    if not wins:
        return None
    has_open = ax == 2 and any(d["kind"][w].tolist().count("mu_open") for d, w in wins)
    name = AXES[ax]

    def unpack(x):
        if ax < 2:
            alpha, drag, b, delay = x
            return alpha, drag, b, delay, None
        alpha, hover_mu, delay = x[:3]
        return alpha, 0.0, -alpha * hover_mu, delay, (x[3] if has_open else None)

    def residual(x):
        alpha, drag, b, delay, ceiling = unpack(x)
        res = [replay(d["t"][w], d[f"mu_{name}"][w], d[f"v_{name}"][w], alpha, drag, b, delay, ceiling) - d[f"v_{name}"][w]
               for d, w in wins]
        return np.concatenate(res)

    if ax < 2:
        x0, lo, hi = [0.6, 0.8, 0.0, 0.02], [0.05, 0.0, -2.0, 0.0], [3.0, 5.0, 2.0, 0.12]
    else:
        x0, lo, hi = [0.56, 3.0, 0.02], [0.05, 0.5, 0.0], [3.0, 8.0, 0.12]
        if has_open:
            x0, lo, hi = x0 + [1.5], lo + [0.2], hi + [6.0]
    sol = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-3)
    alpha, drag, b, delay, ceiling = unpack(sol.x)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    out = {"alpha": float(alpha), "drag": float(drag), "delay": float(delay), "rms_v": rms, "windows": len(wins)}
    if ax < 2:
        out["trim_acc"] = float(b)
        out["tau"] = 1.0 / (alpha * KPV[ax] + drag)
        out["k"] = alpha * KPV[ax] / (alpha * KPV[ax] + drag)
    else:
        out["hover_mu"] = float(-b / alpha)
        out["tau"] = 1.0 / (alpha * KPV[ax])
        if ceiling is not None:
            out["a_z_max"] = float(ceiling)
    return out


def fit_joint(datasets: list[dict]) -> dict:
    """All three axes with one shared transport delay (one radio link)."""
    wins = {ax: [(d, w) for d in datasets for w in replay_windows(d, ax)] for ax in range(3)}
    if not all(wins.values()):
        return {AXES[ax]: r for ax in range(3) if (r := fit_axis(datasets, ax))}
    has_open = any("mu_open" in d["kind"][w].tolist() for d, w in wins[2])

    def unpack(x):
        p = {"x": (x[0], x[1], x[2]), "y": (x[3], x[4], x[5]), "z": (x[6], 0.0, -x[6] * x[7])}
        return p, x[8], (x[9] if has_open else None)

    def residual(x):
        p, delay, ceiling = unpack(x)
        res = []
        for ax in range(3):
            alpha, drag, b = p[AXES[ax]]
            c = ceiling if ax == 2 else None
            res += [replay(d["t"][w], d[f"mu_{AXES[ax]}"][w], d[f"v_{AXES[ax]}"][w], alpha, drag, b, delay, c)
                    - d[f"v_{AXES[ax]}"][w] for d, w in wins[ax]]
        return np.concatenate(res)

    x0 = [0.6, 0.8, 0.0, 0.6, 0.8, 0.0, 0.56, 3.0, 0.02]
    lo = [0.05, 0.0, -2.0, 0.05, 0.0, -2.0, 0.05, 0.5, 0.0]
    hi = [3.0, 5.0, 2.0, 3.0, 5.0, 2.0, 3.0, 8.0, 0.12]
    if has_open:
        x0, lo, hi = x0 + [1.5], lo + [0.2], hi + [6.0]
    sol = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-3)
    p, delay, ceiling = unpack(sol.x)
    out = {}
    for ax in range(3):
        alpha, drag, b = (float(v) for v in p[AXES[ax]])
        r = {"alpha": alpha, "drag": drag, "delay": float(delay), "windows": len(wins[ax]),
             "rms_v": float(np.sqrt(np.mean(np.concatenate([replay(d["t"][w], d[f"mu_{AXES[ax]}"][w], d[f"v_{AXES[ax]}"][w], alpha,
                                                                   drag, b, delay, ceiling if ax == 2 else None)
                                                            - d[f"v_{AXES[ax]}"][w] for d, w in wins[ax]]) ** 2)))}
        if ax < 2:
            r.update(trim_acc=b, tau=1.0 / (alpha * KPV[ax] + drag), k=alpha * KPV[ax] / (alpha * KPV[ax] + drag))
        else:
            r.update(hover_mu=-b / alpha, tau=1.0 / (alpha * KPV[ax]))
            if ceiling is not None:
                r["a_z_max"] = float(ceiling)
        out[AXES[ax]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("fit.json"))
    args = ap.parse_args()
    data = [load(p) for p in args.csv]
    report: dict = {"files": [str(p) for p in args.csv]}

    fits = [f for d in data for f in step_fits(d)]
    report["steps"] = pair_summary(fits) if fits else {}
    report["hover"] = [h for d in data if (h := hover_stats(d))]
    report["thrust"] = [t for d in data if (t := thrust_fits(d))]
    report["replay"] = fit_joint(data)

    print("== step fits (per axis, mirrored pairs)")
    for ax, s in report["steps"].items():
        print(f"  {ax}: tau {s['tau']:.3f} s  apparent lag {s['td'] * 1e3:.0f} ms (incl. filter, not transport delay)  k {s['k']:.3f}  trim {s['trim_v']:+.3f} m/s  (n={s['n']})")
    for h in report["hover"]:
        print(f"== hover: mu_z {h['hover_mu']:.3f} +- {h['hover_mu_std']:.3f} at {h['vbat']:.2f} V (n={h['n']})")
    for th in report["thrust"]:
        if "alpha_z" in th:
            print(f"== thrust: alpha_z {th['alpha_z']:.3f}  b_z {th['b_z']:+.3f}  hover mu_z {th['hover_mu']:.3f}"
                  + (f"  ceiling a_z {th['a_z_max']:.2f} m/s^2" if "a_z_max" in th else ""))
    print("== model replay (sim units)")
    for ax, r in report["replay"].items():
        extra = f"  trim {r['trim_acc']:+.3f} m/s^2" if "trim_acc" in r else f"  hover mu_z {r['hover_mu']:.3f}"
        extra += f"  ceiling {r['a_z_max']:.2f}" if "a_z_max" in r else ""
        print(f"  {ax}: alpha {r['alpha']:.3f}  drag {r['drag']:.3f}  delay {r['delay'] * 1e3:.1f} ms  tau {r['tau']:.3f}"
              f"{extra}  rms {r['rms_v']:.3f} m/s ({r['windows']} windows)")

    rp = report["replay"]
    if {"x", "y", "z"} <= rp.keys():
        hover = [rp["z"]["hover_mu"]] + [h["hover_mu"] for h in report["hover"]]
        hover += [th["hover_mu"] for th in report["thrust"] if "hover_mu" in th]
        ceil = [th["a_z_max"] for th in report["thrust"] if "a_z_max" in th] + ([rp["z"]["a_z_max"]] if "a_z_max" in rp["z"] else [])
        delay = float(np.median([rp[a]["delay"] for a in AXES]))
        sug = {
            "alpha": [round(rp[a]["alpha"], 3) for a in AXES],
            "drag": [round(rp["x"]["drag"], 3), round(rp["y"]["drag"], 3), 0.0],
            "hover_crossing_range": [round(min(hover) - 0.1, 2), round(max(hover) + 0.1, 2)],
            "delay": round(delay, 4),
        }
        if ceil:
            sug["a_z_max"] = round(float(np.mean(ceil)), 2)
        report["suggested_velocity_action_cfg"] = sug
        print("== suggested VelocityActionCfg (tasks/safe_cf/mdp/velocity_action.py)")
        for k, v in sug.items():
            print(f"    {k}={tuple(v) if isinstance(v, list) else v}")
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
