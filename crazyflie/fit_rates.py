#!/usr/bin/env python3
"""Fit the firmware rate loop (Ixx, Iyy, motor lag, delay) and the collective-thrust lag from `fly_rates.py --plan rsteps` logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.rate_loop import DT, simulate  # noqa: E402
from fit_vsteps import mocap_kinematics  # noqa: E402

G = 9.81
CF_MS_WRAP = 1 << 24          # log timestamps are 24-bit milliseconds since boot
PLAN_TAU = 0.03               # [s] the mj-nlp planner's rate-loop time constant: kv = J / PLAN_TAU


# #### // Logs -> doublets and onboard signals on the host clock // ####
def read(run: Path, name: str):
    return np.genfromtxt(run / name, delimiter=",", names=True, dtype=None, encoding=None)


def onboard_time(*blocks) -> list:
    """Host time of each block's samples: unwrapped cf_ms plus the smallest arrival gap over all blocks (the clock offset)."""
    ms = [np.asarray(b["cf_ms"], float) for b in blocks]
    ms = [m + CF_MS_WRAP * np.cumsum(np.r_[0, np.diff(m) < -CF_MS_WRAP / 2]) for m in ms]
    offset = min(float(np.min(b["t"] - m / 1e3)) for b, m in zip(blocks, ms))
    return [m / 1e3 + offset for m in ms]


def doublets(rt) -> list:
    """(column 0-3, send times, command, hover thrust) per doublet in rates.csv; column 0 as a fraction of hover (the last row)."""
    out = []
    for seg in np.unique(rt["seg"]):
        r = rt[rt["seg"] == seg]
        cmd = np.column_stack([r["T"] / r["T"][-1], r["p"], r["q"], r["r"]])
        col = int(np.argmax(np.abs(cmd - [1.0, 0.0, 0.0, 0.0]).max(axis=0)))
        out.append((col, np.asarray(r["t"], float), cmd[:, col], float(r["T"][-1])))
    return out


def load_run(run: Path) -> dict:
    rt, rate = read(run, "rates.csv"), read(run, "cf_rate.csv")
    (t_rate,) = onboard_time(rate)
    t_m, _, vel, _ = mocap_kinematics(read(run, "mocap.csv"))
    gyro = np.radians(np.column_stack([rate["gyrox"], rate["gyroy"], rate["gyroz"]]))
    return dict(doublets=doublets(rt), t_rate=t_rate, gyro=gyro, t_m=t_m, vz=vel[:, 2])


# #### // Exact first-order response to a command held from each send time (smooth in tau and the delay) // ####
def first_order(t, t_cmd, u, base, y0, tau, d, out="y"):
    """y' = (u(t - d) - y) / tau from y(t[0]) = y0, u = base before the first send; out "y" or "int" (integral from t[0])."""
    t_on, du = t_cmd + d, np.diff(np.r_[base, u])
    late = t_on > t[0]
    u0 = base + du[~late].sum()
    s = t[:, None] - t_on[None, late]
    e0, es = np.exp(-(t - t[0]) / tau), np.exp(-np.clip(s, 0.0, None) / tau)
    if out == "int":
        return u0 * (t - t[0]) + (y0 - u0) * tau * (1.0 - e0) + (np.clip(s, 0.0, None) - tau * (1.0 - es)) @ du[late]
    return u0 + (y0 - u0) * e0 + (1.0 - es) @ du[late]


def window(t, t_cmd, before=0.03, after=0.03):
    return (t >= t_cmd[0] - before) & (t <= t_cmd[-1] + after)


def fit_lag(cases, model) -> tuple:
    """Joint (tau, d) over cases (t, measured, ...): least squares of model(case, tau, d) - measured."""
    res = least_squares(lambda x: np.concatenate([model(c, *x) - c[1] for c in cases]), [0.03, 0.02],
                        bounds=([0.003, 0.0], [0.3, 0.12]), x_scale=[0.01, 0.01])
    return res.x, float(np.sqrt(np.mean(res.fun ** 2)))


# #### // Roll and pitch through the firmware loop (cfdeploy.rate_loop): Ixx, Iyy, motor lag; the delay grid-searched // ####
def fit_loop(runs) -> dict | None:
    cases = [(r["t_rate"][w], r["gyro"][w, col - 1], tc, u, col - 1) for r in runs for col, tc, u, _ in r["doublets"] if col in (1, 2)
             for w in [window(r["t_rate"], tc)]]
    if not cases:
        return None
    n = int(max(c[0][-1] - c[0][0] for c in cases) / DT) + 2
    grid = np.arange(n) * DT

    def residual(x, d):
        cmd, w0 = np.zeros((n, len(cases), 3)), np.zeros((len(cases), 3))
        for b, (t, y, tc, u, j) in enumerate(cases):
            i = np.searchsorted(tc + d, t[0] + grid, side="right") - 1
            cmd[:, b, j], w0[b, j] = np.where(i >= 0, u[np.clip(i, 0, None)], 0.0), y[0]
        gyro = simulate(cmd, np.r_[np.exp(x[:2]), 1.0], np.exp(x[2]), w0=w0)[1]
        return np.concatenate([np.interp(t - t[0], grid, gyro[:, b, j]) - y for b, (t, y, _, _, j) in enumerate(cases)])

    best, x0 = None, np.log([3e-5, 3e-5, 0.04])
    for d in np.arange(0.0, 0.031, 0.002):
        res = least_squares(residual, x0, args=(d,), bounds=(np.log([5e-6, 5e-6, 0.005]), np.log([2e-4, 2e-4, 0.2])))
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        if best is None or rms < best["rms"]:
            best, x0 = dict(J=np.exp(res.x[:2]), tau_m=float(np.exp(res.x[2])), d=float(d), rms=rms, n=len(cases)), res.x
    return best


def fit_thrust(runs) -> dict | None:
    """Collective lag on the mocap vertical velocity: vz' = g (f - 1), f first order behind the commanded fraction of hover."""
    cases = [(r["t_m"][w], r["vz"][w], tc, u, r["vz"][w][0])
             for r in runs for col, tc, u, _ in r["doublets"] if col == 0 for w in [window(r["t_m"], tc)]]
    if not cases:
        return None
    vz = lambda c, tau, d: c[4] + G * (first_order(c[0], c[2], c[3], 1.0, 1.0, tau, d, out="int") - (c[0] - c[0][0]))
    (tau, d), rms = fit_lag(cases, vz)
    return dict(tau=float(tau), d=float(d), rms=rms, n=len(cases))


def fit(paths) -> dict:
    runs = [load_run(Path(p)) for p in paths]
    return {"loop": fit_loop(runs), "T": fit_thrust(runs)}


# #### // CLI // ####
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="log dirs from fly_rates.py --plan rsteps")
    out = fit(ap.parse_args(argv).runs)
    f, T = out["loop"], out["T"]
    if f:
        print(f"rate loop: Ixx {f['J'][0]:.3e}, Iyy {f['J'][1]:.3e} kg m^2, motor lag {f['tau_m'] * 1e3:.1f} ms, delay {f['d'] * 1e3:.0f} ms "
              f"(host send -> gyro), rms {f['rms']:.3f} rad/s over {f['n']} roll/pitch doublets")
    if T:
        print(f"collective: tau {T['tau'] * 1e3:.1f} ms, delay {T['d'] * 1e3:.1f} ms (host send -> mocap; expect ~ motor lag + mocap latency), "
              f"rms {T['rms']:.3f} m/s over {T['n']} doublets")
    if f:
        J = np.r_[f["J"], 0.88 * f["J"].sum()]   # yaw is torque-limited (velCtl.yawLim): Izz as CF2.0, 0.88 (Ixx + Iyy)
        print("\nmj-nlp models/crazyflie/crazyflie.xml:")
        print(f'  <inertial ... diaginertia="{J[0]:.4e} {J[1]:.4e} {J[2]:.4e}"/>   (Izz = 0.88 (Ixx + Iyy), not identified)')
        for a, j in zip("xyz", J):
            print(f'  rate_{a}: kv="{j / PLAN_TAU:.5e}"')
        print(f'  <numeric name="motor_lag" data="{f["tau_m"]:.4f}"/>')
        print("then re-solve the flip; fly_rates.py compensates the tape for this loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
