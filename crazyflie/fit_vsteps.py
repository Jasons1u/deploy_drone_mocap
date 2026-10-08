#!/usr/bin/env python3
"""Fit the closed-loop velocity response from `sysid_flight.py --plan vsteps` logs (--cascade: the model Isaac uses)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import load as load_config  # noqa: E402
from cfdeploy.drone_model import step_response  # noqa: E402

CFG = load_config()
RATE = CFG.mocap.rate                       # [Hz] mocap frame rate
MOCAP_LATENCY = CFG.mocap.latency           # [s] exposure -> host arrival
KV = tuple(CFG.drone_model.kv)              # firmware velCtl.kvX/kvY/kvZ during the sysID flights
D_FB = CFG.drone_model.nominal.d_fb         # [s] mocap -> onboard alpha-beta estimate
AXES = "xyz"


# #### // Logs -> mocap kinematics and velocity-step edges // ####
def load(run: Path):
    rd = lambda f: np.genfromtxt(run / f, delimiter=",", names=True, dtype=None, encoding=None)
    return rd("setpoints.csv"), rd("mocap.csv"), rd("cf_vel.csv")


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


# #### // Joint fit of the cascade model over many edges (the model in config drone_model / Isaac VelCtlAction) // ####
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


# #### // Diagnostics: per-edge first-order fits, attitude tracking, onboard estimate lag // ####
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
    rd = lambda name: np.genfromtxt(run / name, delimiter=",", names=True)
    vel, ref = rd("cf_vel.csv"), rd("cf_ref.csv")
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


# #### // CLI // ####
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="log dirs from sysid_flight.py --plan vsteps")
    ap.add_argument("--cascade", action="store_true", help="joint fit of the firmware-structured cascade model")
    args = ap.parse_args(argv)
    if args.cascade:
        cascade_report(args.runs)
    else:
        edge_report(args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
