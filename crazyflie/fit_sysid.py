#!/usr/bin/env python3
"""Fit sysID flights: `vsteps` = velocity response (sysid_flight.py; --cascade = the Isaac model), `rsteps` = rate loop + thrust lag (fly_rates.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import load as load_config  # noqa: E402
from cfdeploy.drone_model import G, step_response  # noqa: E402
from cfdeploy.rate import DT, simulate  # noqa: E402

CFG = load_config()
RATE = CFG.mocap.rate                       # [Hz] mocap frame rate
MOCAP_LATENCY = CFG.mocap.latency           # [s] exposure -> host arrival
KV = tuple(CFG.drone_model.kv)              # firmware velCtl.kvX/kvY/kvZ during the sysID flights
D_FB = CFG.drone_model.nominal.d_fb         # [s] mocap -> onboard alpha-beta estimate
AXES = "xyz"
CF_MS_WRAP = 1 << 24                        # log timestamps are 24-bit milliseconds since boot


# #### // Logs -> mocap kinematics // ####
def read(run: Path, name: str):
    return np.genfromtxt(run / name, delimiter=",", names=True, dtype=None, encoding=None)


def load(run: Path):
    return read(run, "setpoints.csv"), read(run, "mocap.csv"), read(run, "cf_vel.csv")


def mocap_kinematics(m):
    """Positions on the uniform frame grid -> zero-phase Savitzky-Golay velocity/acceleration, timed at exposure."""
    f = m["frame"].astype(int)
    grid = np.arange(f.min(), f.max() + 1)
    pos = np.stack([np.interp(grid, f, m[a]) for a in AXES], 1)
    t_on_time = (m["t"] - (f - f[0]) / RATE).min()
    t = t_on_time + (grid - f[0]) / RATE - MOCAP_LATENCY
    vel = savgol_filter(pos, 13, 2, deriv=1, delta=1 / RATE, axis=0)
    acc = savgol_filter(pos, 31, 2, deriv=2, delta=1 / RATE, axis=0)
    return t, pos, vel, acc


# #### // vsteps: velocity-step edges // ####
def edges(sp):
    """Velocity-segment starts: (t, seg, axis, v_from, v_to, duration)."""
    out, prev_v = [], np.zeros(3)
    for seg in np.unique(sp["seg"][sp["seg"] >= 0]):
        r = sp[sp["seg"] == seg]
        if r["mode"][0] != "vel":
            prev_v = np.zeros(3)
            continue
        v = np.array([r["vx"][0], r["vy"][0], r["vz"][0]])
        dv = v - prev_v
        if np.abs(dv).max() > 1e-6:
            ax = int(np.argmax(np.abs(dv)))
            out.append((r["t"][0], int(seg), ax, prev_v[ax], v[ax], r["t"][-1] - r["t"][0]))
        prev_v = v
    return out


# #### // vsteps: joint fit of the cascade model over many edges (the model in config drone_model / Isaac VelCtlAction) // ####
def cascade_edges(runs, axes):
    """Per-edge measured window (t from the step, velocity on the step axis, v and a at the step) for the given axes."""
    out = []
    for run in runs:
        sp, m, _ = load(run)
        t, pos, vel, acc = mocap_kinematics(m)
        for t_s, seg, ax, v_from, v_to, dur in edges(sp):
            if ax not in axes:
                continue
            w = (t >= t_s) & (t <= t_s + min(dur, 1.2))
            v0, a0 = np.interp(t_s, t, vel[:, ax]), np.interp(t_s, t, acc[:, ax])
            out.append((t[w] - t_s, vel[w, ax], v0, a0, v_from, v_to))
    return out


def fit_cascade(edge_data, kv, fit_amax, fit_drag):
    """Joint fit of (tau_a, d_cmd[, a_max][, drag]); edge_data: (t_rel, v_meas, v0, a0, v_from, v_to)."""
    def unpack(x):
        i = 2
        a_max = x[i] if fit_amax else 20.0
        i += fit_amax
        drag = x[i] if fit_drag else 0.0
        return x[0], x[1], a_max, drag

    def resid(x):
        tau_a, d_cmd, a_max, drag = unpack(x)
        return np.concatenate([step_response(tr, v0, a0, vf, vt, kv, tau_a, d_cmd, a_max, drag, D_FB) - vm
                               for tr, vm, v0, a0, vf, vt in edge_data])

    x0 = [0.08, 0.03] + ([3.0] if fit_amax else []) + ([0.2] if fit_drag else [])
    lo = [0.01, 0.0] + ([1.0] if fit_amax else []) + ([0.0] if fit_drag else [])
    hi = [0.4, 0.15] + ([8.0] if fit_amax else []) + ([3.0] if fit_drag else [])
    res = least_squares(resid, x0, bounds=(lo, hi), diff_step=1e-3)
    tau_a, d_cmd, a_max, drag = unpack(res.x)
    per_edge = [float(np.sqrt(np.mean((step_response(tr, v0, a0, vf, vt, kv, tau_a, d_cmd, a_max, drag, D_FB) - vm) ** 2)))
                for tr, vm, v0, a0, vf, vt in edge_data]
    return dict(tau_a=tau_a, d_cmd=d_cmd, a_max=a_max, drag=drag, rms=per_edge)


def cascade_report(runs) -> None:
    for name, axes, kv, fit_amax, fit_drag in (("xy", (0, 1), KV[0], True, False), ("z", (2,), KV[2], False, True)):
        data = cascade_edges(runs, axes)
        if not data:
            continue
        r = fit_cascade(data, kv, fit_amax, fit_drag)
        print(f"{name}: {len(data)} edges, kv {kv} (fixed), tau_a {r['tau_a'] * 1e3:.0f} ms, d_cmd {r['d_cmd'] * 1e3:.0f} ms, "
              f"a_max {r['a_max']:.2f} m/s^2, drag {r['drag']:.2f} 1/s; edge rms mean {np.mean(r['rms']):.3f} "
              f"max {np.max(r['rms']):.3f} m/s")


# #### // vsteps diagnostics: per-edge first-order fits, attitude tracking, onboard estimate lag // ####
def first_order(t, t_s, v0, slope, v_to, k, tau, d, a_max):
    """Pre-step trend v0 + slope*(t - t_s) until t_s + d, then first order toward k*v_to with an acceleration limit."""
    dt = 0.0005
    tt = t_s + d + np.arange(0.0, max(t[-1] - t_s - d, 0.0) + 2 * dt, dt)
    out, v = np.empty_like(tt), v0 + slope * d
    for i in range(len(tt)):
        out[i] = v
        v += np.clip((k * v_to - v) / tau, -a_max, a_max) * dt
    return np.where(t < t_s + d, v0 + slope * (t - t_s), np.interp(t, tt, out))


def fit_edge(t, v, t_s, v_from, v_to, dur, fit_gain, fit_amax):
    w = (t >= t_s - 0.15) & (t <= t_s + min(dur, 1.2))
    tw, vw = t[w], v[w]
    pre = tw < t_s
    slope, v0 = np.polyfit(tw[pre] - t_s, vw[pre], 1) if pre.sum() > 5 else (0.0, v_from)

    def unpack(x):
        k = x[0] if fit_gain else 1.0
        tau, d = x[-3 if fit_amax else -2], x[-2 if fit_amax else -1]
        a_max = x[-1] if fit_amax else 1e9
        return k, tau, d, a_max

    x0 = ([1.0] if fit_gain else []) + [0.2, 0.03] + ([4.0] if fit_amax else [])
    lo = ([0.5] if fit_gain else []) + [0.02, 0.0] + ([0.5] if fit_amax else [])
    hi = ([1.5] if fit_gain else []) + [1.5, 0.2] + ([20.0] if fit_amax else [])
    res = least_squares(lambda x: first_order(tw, t_s, v0, slope, v_to, *unpack(x)) - vw, x0, bounds=(lo, hi))
    k, tau, d, a_max = unpack(res.x)
    return dict(k=k, tau=tau, d=d, a_max=a_max, rms=float(np.sqrt(np.mean(res.fun ** 2))), v0=v0, slope=slope)


def estimator_lag(t, v_mocap, tel, axis):
    """Lag [ms] of the onboard velocity estimate behind mocap (cross-correlation over the flight) and RMS difference."""
    name = f"stateEstimatev{AXES[axis]}"
    w = (t > tel["t"][0]) & (t < tel["t"][-1])
    onboard = np.interp(t[w], tel["t"], tel[name])
    ref = v_mocap[w, axis]
    lags = np.arange(-0.01, 0.08, 1 / RATE)
    err = [np.sqrt(np.mean((onboard - np.interp(t[w] - L, t[w], ref)) ** 2)) for L in lags]
    i = int(np.argmin(err))
    return lags[i] * 1e3, err[i], float(np.sqrt(np.mean((onboard - ref) ** 2)))


def attitude_report(run: Path, sp):
    """Onboard tilt (IMU) vs desired tilt, both legacy convention: best-fit lag, rms, error spectrum peak above 3 Hz."""
    vel, ref = read(run, "cf_vel.csv"), read(run, "cf_ref.csv")
    t0, t1 = sp["t"][0] + 3.0, sp["t"][-1] - 5.0
    tt = np.arange(t0, t1, 0.01)
    out = {}
    for name, act, des in (("roll", "stabilizerroll", "velCtlrollDes"), ("pitch", "stabilizerpitch", "velCtlpitchDes")):
        a = np.interp(tt, vel["t"], vel[act])
        lags = np.arange(-0.02, 0.25, 0.002)
        err = [np.sqrt(np.mean((np.interp(tt - L, ref["t"], ref[des]) - a) ** 2)) for L in lags]
        i = int(np.argmin(err))
        e = a - np.interp(tt - lags[i], ref["t"], ref[des])
        spec = np.abs(np.fft.rfft(e - e.mean())) ** 2
        freq = np.fft.rfftfreq(len(e), 0.01)
        band = freq > 3.0
        j = int(np.argmax(spec[band]))
        share = spec[band].sum() / spec[1:].sum()
        out[name] = dict(lag=lags[i] * 1e3, rms=err[i], peak_hz=freq[band][j], hf_share=share)
    return out


def edge_report(runs) -> None:
    rows = []
    for run in runs:
        sp, m, tel = load(run)
        t, pos, vel, acc = mocap_kinematics(m)
        print(f"\n{run.name}")
        print("  seg axis  v_from -> v_to   tau[ms] delay[ms]  gain   a_max   peak|a|  rms[m/s]  cross|v|  dz[cm]")
        for t_s, seg, ax, v_from, v_to, dur in edges(sp):
            big = abs(v_to - v_from) >= 0.9
            fit = fit_edge(t, vel[:, ax], t_s, v_from, v_to, dur, fit_gain=abs(v_to) > 0, fit_amax=big)
            w = (t >= t_s) & (t <= t_s + min(dur, 1.2))
            others = [i for i in range(3) if i != ax]
            cross = np.abs(vel[w][:, others]).max()
            dz = (pos[w, 2].max() - pos[w, 2][0]) * 100 if ax != 2 else np.nan
            peak = np.abs(acc[w, ax]).max()
            rows.append(dict(run=run.name, seg=seg, axis=AXES[ax], v_from=v_from, v_to=v_to, peak=peak, **fit))
            amax = f"{fit['a_max']:5.2f}" if big else "   - "
            print(f"  {seg:3d}  {AXES[ax]}   {v_from:+.2f} -> {v_to:+.2f}   {fit['tau'] * 1e3:6.0f}  {fit['d'] * 1e3:6.0f}    "
                  f"{fit['k']:.3f}  {amax}   {peak:5.2f}   {fit['rms']:.3f}    {cross:.3f}   {dz:+.1f}")
        for name, a in attitude_report(run, sp).items():
            print(f"  attitude {name}: IMU follows desired with lag {a['lag']:.0f} ms, rms {a['rms']:.2f} deg; "
                  f"error power above 3 Hz {a['hf_share'] * 100:.0f}% (peak {a['peak_hz']:.1f} Hz)")
        for ax in range(3):
            if any(r["axis"] == AXES[ax] for r in rows if r["run"] == run.name):
                lag, err_lag, err0 = estimator_lag(t, vel, tel, ax)
                print(f"  onboard v{AXES[ax]} vs mocap: lag {lag:.0f} ms, rms {err0:.3f} m/s ({err_lag:.3f} after removing the lag)")

    print("\nsummary per axis (accel = leaving rest, brake = stopping):")
    for ax in AXES:
        r = [x for x in rows if x["axis"] == ax]
        if not r:
            continue
        acc_e = [x for x in r if x["v_to"] != 0 and abs(x["v_to"] - x["v_from"]) < 0.9]
        brk = [x for x in r if x["v_to"] == 0]
        big = [x for x in r if abs(x["v_to"] - x["v_from"]) >= 0.9]
        fmt = lambda xs, k: f"{np.mean([x[k] for x in xs]) * (1e3 if k in ('tau', 'd') else 1):.3g}" if xs else "-"
        print(f"  {ax}: tau accel {fmt(acc_e, 'tau')} ms / brake {fmt(brk, 'tau')} ms; delay {fmt(r, 'd')} ms; "
              f"gain {fmt(acc_e, 'k')}; accel limit (|step| >= 0.9) {fmt(big, 'a_max')} m/s^2, peak |a| {fmt(big, 'peak')}")


# #### // rsteps: doublets and onboard signals on the host clock // ####
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


def load_rsteps(run: Path) -> dict:
    rt, rate = read(run, "rates.csv"), read(run, "cf_rate.csv")
    (t_rate,) = onboard_time(rate)
    t_m, _, vel, _ = mocap_kinematics(read(run, "mocap.csv"))
    gyro = np.radians(np.column_stack([rate["gyrox"], rate["gyroy"], rate["gyroz"]]))
    return dict(doublets=doublets(rt), t_rate=t_rate, gyro=gyro, t_m=t_m, vz=vel[:, 2])


# #### // rsteps: exact first-order response to a command held from each send time (smooth in tau and the delay) // ####
def lag_response(t, t_cmd, u, base, y0, tau, d, out="y"):
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


# #### // rsteps: roll and pitch through the firmware loop (cfdeploy.rate): Ixx, Iyy, motor lag; the delay grid-searched // ####
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
    vz = lambda c, tau, d: c[4] + G * (lag_response(c[0], c[2], c[3], 1.0, 1.0, tau, d, out="int") - (c[0] - c[0][0]))
    (tau, d), rms = fit_lag(cases, vz)
    return dict(tau=float(tau), d=float(d), rms=rms, n=len(cases))


def fit_rsteps(paths) -> dict:
    runs = [load_rsteps(Path(p)) for p in paths]
    return {"loop": fit_loop(runs), "T": fit_thrust(runs)}


def rsteps_report(runs) -> None:
    out = fit_rsteps(runs)
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
        print(f'  every actuator: dynprm="{f["tau_m"]:.4f}"   (motor lag)')
        print("then re-solve the flip; fly_rates.py fits the rate PID's commands onto its planned rates")


# #### // CLI // ####
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="plan", required=True)
    vs = sub.add_parser("vsteps", help="velocity steps (sysid_flight.py --plan vsteps)")
    vs.add_argument("runs", nargs="+", type=Path, help="log dirs from sysid_flight.py --plan vsteps")
    vs.add_argument("--cascade", action="store_true", help="joint fit of the firmware-structured cascade model")
    rs = sub.add_parser("rsteps", help="rate and thrust doublets (fly_rates.py --plan rsteps)")
    rs.add_argument("runs", nargs="+", type=Path, help="log dirs from fly_rates.py --plan rsteps")
    args = ap.parse_args(argv)
    if args.plan == "rsteps":
        rsteps_report(args.runs)
    elif args.cascade:
        cascade_report(args.runs)
    else:
        edge_report(args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
