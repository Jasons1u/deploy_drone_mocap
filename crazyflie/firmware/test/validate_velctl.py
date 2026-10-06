#!/usr/bin/env python3
"""Host validation of mocap_velctl/src/velctl_math.c: alpha-beta gains, sign conventions, closed-loop velocity response."""

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "mocap_velctl" / "src"
G = 9.81
F3 = ctypes.c_float * 3


class AbFilter(ctypes.Structure):
    _fields_ = [("p", F3), ("v", F3), ("init", ctypes.c_bool)]


class FrameClock(ctypes.Structure):
    _fields_ = [("lag", ctypes.c_float)]


class Gains(ctypes.Structure):
    _fields_ = [("kv", F3), ("ki", F3), ("iLimit", F3), ("maxTiltRad", ctypes.c_float),
                ("maxAccUp", ctypes.c_float), ("maxAccDown", ctypes.c_float)]


class CtlState(ctypes.Structure):
    _fields_ = [("integ", F3)]


def load_lib():
    so = Path(tempfile.mkdtemp()) / "libvelctl.so"
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-o", str(so),
                    str(SRC / "velctl_math.c"), "-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    lib.abUpdate.argtypes = [ctypes.POINTER(AbFilter), F3, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.abReset.argtypes = [ctypes.POINTER(AbFilter), F3]
    lib.abFramesElapsed.argtypes = [ctypes.POINTER(AbFilter), ctypes.POINTER(FrameClock), F3, ctypes.c_float,
                                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.abFramesElapsed.restype = ctypes.c_int
    lib.velCtlAccel.argtypes = [ctypes.POINTER(CtlState), ctypes.POINTER(Gains), F3, F3, ctypes.c_float,
                                ctypes.c_bool, ctypes.c_bool, F3]
    lib.velCtlAccelToZb.argtypes = [ctypes.POINTER(Gains), F3, F3, ctypes.POINTER(ctypes.c_float)]
    lib.velCtlZbToRollPitch.argtypes = [F3, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_float)]
    lib.velCtlThrustN.argtypes = [ctypes.c_float, F3, ctypes.c_float, F3]
    lib.velCtlThrustN.restype = ctypes.c_float
    return lib


def kalata(lam):
    s = np.sqrt(lam * lam + 8 * lam)
    return -(lam * lam + 8 * lam - (lam + 4) * s) / 8, (lam * lam + 4 * lam - lam * s) / 4


# ---------------- mocap stream model ----------------

def true_trajectory(rng, t):
    """Smooth 3-axis velocity: random 0.4 s-filtered steps up to 1 m/s, plus a still segment at the start."""
    dt = t[1] - t[0]
    v = np.zeros((len(t), 3))
    target = np.zeros(3)
    for k in range(1, len(t)):
        if t[k] > 2.0 and rng.random() < dt / 0.6:
            target = rng.uniform(-1.0, 1.0, 3) * np.array([1, 1, 0.5])
        v[k] = v[k - 1] + (target - v[k - 1]) * dt / 0.15
    p = np.cumsum(v, axis=0) * dt
    return p, v


def deliver(rng, frame_t, send_mask, latency, jitter_p, jitter_max, drop_p):
    """Arrival times of the frames the host sends: fixed latency, random radio bunching delay, drops, order kept."""
    idx = np.flatnonzero(send_mask & (rng.random(len(frame_t)) > drop_p))
    arr = frame_t[idx] + latency + np.where(rng.random(len(idx)) < jitter_p, rng.uniform(0, jitter_max, len(idx)), 0)
    arr = np.maximum.accumulate(arr)
    return idx, arr


def run_ab(lib, z, idx, arr, rate, alpha, beta, oracle=False, max_delay=0.7, clock_gain=0.01, min_step=0.0015):
    """Firmware onPosition(): frame clock on arrival gaps (or the true frame count with oracle=True) -> alpha-beta."""
    f, clk = AbFilter(), FrameClock()
    out_t, out_v, miscount, moving_miscount = [], [], 0, 0
    last, last_i = None, None
    for i, a in zip(idx, arr):
        zi = F3(*z[i])
        if last is None:
            lib.abReset(ctypes.byref(f), zi)
        else:
            n = lib.abFramesElapsed(ctypes.byref(f), ctypes.byref(clk), zi, a - last, 1.0 / rate, max_delay, clock_gain,
                                    min_step)
            miscount += n != i - last_i
            moving_miscount += n != i - last_i and np.linalg.norm(z[i] - z[last_i]) / ((i - last_i) / rate) > 0.5
            lib.abUpdate(ctypes.byref(f), zi, (i - last_i if oracle else n) / rate, alpha, beta)
        last, last_i = a, i
        out_t.append(a)
        out_v.append(list(f.v))
    run_ab.miscount = (miscount, moving_miscount)
    return np.array(out_t), np.array(out_v)


def run_ema(z, idx, arr, vel_alpha=0.3, min_dt=0.008):
    """The current mocapEst: EMA on finite differences against a sample >= min_dt old (arrival timestamps)."""
    v = np.zeros(3)
    ref_p, ref_t = z[idx[0]], arr[0]
    out_v = [v.copy()]
    for i, a in zip(idx[1:], arr[1:]):
        if a - ref_t >= min_dt:
            v = v + vel_alpha * ((z[i] - ref_p) / (a - ref_t) - v)
            ref_p, ref_t = z[i], a
        out_v.append(v.copy())
    return arr, np.array(out_v)


def score(t_est, v_est, t_true, v_true, still_until=1.8):
    """RMS velocity error seen by a 500 Hz controller (zero-order hold), still-segment noise, and best-fit lag."""
    tc = np.arange(t_est[0] + 0.05, t_true[-1] - 0.05, 0.002)
    held = v_est[np.searchsorted(t_est, tc, side="right") - 1]
    truth = np.stack([np.interp(tc, t_true, v_true[:, i]) for i in range(3)], 1)
    err = held - truth
    still = tc < still_until
    lags = np.arange(0, 0.12, 0.001)
    lag_rms = [np.sqrt(np.mean((held - np.stack([np.interp(tc - L, t_true, v_true[:, i]) for i in range(3)], 1))[~still] ** 2))
               for L in lags]
    return (np.sqrt(np.mean(err[~still] ** 2)), held[still].std(axis=0).mean(), lags[int(np.argmin(lag_rms))])


# ---------------- closed loop ----------------

def closed_loop(lib, gains, mass_ctrl, mass_true, steps, t_end, tau_att=0.03, tau_thrust=0.015,
                latency=0.008, sigma=3e-4, alpha=0.4, beta=0.1, rate=240.0, kd=0.0, seed=0):
    """Point mass + first-order tilt/thrust lag; 240 Hz noisy mocap with latency -> alpha-beta -> velocity loop at 500 Hz."""
    rng = np.random.default_rng(seed)
    dt = 0.0005
    p, v = np.zeros(3), np.zeros(3)
    zb, thrust = np.array([0, 0, 1.0]), mass_true * G
    f, st = AbFilter(), CtlState()
    queue, next_frame, next_ctl = [], 0.0, 0.0
    zb_des, t_cmd = zb.copy(), thrust
    log_t, log_v, log_ref = [], [], []
    for k in range(int(t_end / dt)):
        t = k * dt
        if t >= next_frame:
            queue.append((t + latency, p + rng.normal(0, sigma, 3)))
            next_frame += 1.0 / rate
        while queue and queue[0][0] <= t:
            _, z = queue.pop(0)
            if not f.init:
                lib.abReset(ctypes.byref(f), F3(*z))
            else:
                lib.abUpdate(ctypes.byref(f), F3(*z), 1.0 / rate, alpha, beta)
        if t >= next_ctl:
            next_ctl += 0.002
            v_ref = np.zeros(3)
            for t0, ref in steps:
                if t >= t0:
                    v_ref = np.array(ref, float)
            a = F3()
            lib.velCtlAccel(ctypes.byref(st), ctypes.byref(gains), F3(*v_ref), F3(*f.v), 0.002, True, True, a)
            zd, lift = F3(), ctypes.c_float()
            lib.velCtlAccelToZb(ctypes.byref(gains), a, zd, ctypes.byref(lift))
            zb_des = np.array(zd)
            t_cmd = lib.velCtlThrustN(mass_ctrl, zd, lift, F3(*zb))
            log_t.append(t)
            log_v.append(v.copy())
            log_ref.append(v_ref)
        zb = zb + (zb_des - zb) * dt / tau_att
        zb /= np.linalg.norm(zb)
        thrust += (t_cmd - thrust) * dt / tau_thrust
        acc = thrust * zb / mass_true - np.array([0, 0, G]) - kd * v
        v = v + acc * dt
        p = p + v * dt
    return np.array(log_t), np.array(log_v), np.array(log_ref)


def fit_first_order(t, y, t0, y0, y1):
    """Fit delay + time constant of a step from y0 to y1 at t0 (63% and 10% crossings)."""
    s = (y - y0) / (y1 - y0)
    after = t >= t0
    t10 = t[after][np.argmax(s[after] >= 0.1)] - t0
    t63 = t[after][np.argmax(s[after] >= 0.632)] - t0
    tau = (t63 - t10) / (1 - np.log(1 / 0.9))   # t63 - t10 = tau (1 - ln(10/9)) for delay + first order
    delay = t63 - tau
    return tau, delay, s[after].max() - 1.0


def main() -> int:
    lib = load_lib()
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    print("1. alpha-beta gains (240 Hz, sigma 0.3 mm, 8 ms latency, 25% of packets delayed 0-4 ms, 1% drops)")
    rng = np.random.default_rng(1)
    rate, T = 240.0, 1 / 240.0
    t_fine = np.arange(0, 40, 1e-4)
    p_fine, v_fine = true_trajectory(rng, t_fine)
    frame_t = np.arange(0, 39.9, T)
    p_frames = np.stack([np.interp(frame_t, t_fine, p_fine[:, i]) for i in range(3)], 1)
    z = p_frames + rng.normal(0, 3e-4, p_frames.shape)
    idx, arr = deliver(rng, frame_t, np.ones(len(frame_t), bool), 0.008, 0.25, 0.004, 0.01)
    print("     lambda   alpha   beta   rms[m/s]  still-noise[m/s]  lag[ms]")
    best = None
    for lam in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        al, be = kalata(lam)
        te, ve = run_ab(lib, z, idx, arr, rate, al, be)
        rms, noise, lag = score(te, ve, t_fine, v_fine)
        print(f"     {lam:5.2f}   {al:.3f}  {be:.4f}  {rms:.4f}    {noise:.4f}            {lag * 1e3:.0f}")
        if best is None or rms < best[0]:
            best = (rms, lam, al, be, noise, lag)
    rms_ab, lam, al, be, noise_ab, lag_ab = best
    print(f"     best: lambda {lam} -> alpha {al:.3f}, beta {be:.4f}")

    sent_100 = np.zeros(len(frame_t), bool)
    poll = np.arange(0, 39.9, 0.01)
    sent_100[np.unique(np.searchsorted(frame_t, poll, side="right") - 1)] = True
    idx100, arr100 = deliver(np.random.default_rng(2), frame_t, sent_100, 0.008, 0.25, 0.004, 0.01)
    rms_old, noise_old, lag_old = score(*run_ema(z, idx100, arr100), t_fine, v_fine)
    rms_ema240, noise_ema240, lag_ema240 = score(*run_ema(z, idx, arr), t_fine, v_fine)
    print(f"     current EMA @100 Hz feed:  rms {rms_old:.4f}  noise {noise_old:.4f}  lag {lag_old * 1e3:.0f} ms")
    print(f"     current EMA @240 Hz feed:  rms {rms_ema240:.4f}  noise {noise_ema240:.4f}  lag {lag_ema240 * 1e3:.0f} ms")
    print(f"     alpha-beta @240 Hz:        rms {rms_ab:.4f}  noise {noise_ab:.4f}  lag {lag_ab * 1e3:.0f} ms")
    check("alpha-beta beats the current EMA (100 Hz feed) on rms error", rms_ab < rms_old)
    check("alpha-beta lag lower than current EMA", lag_ab < lag_old)

    run_ab(lib, z, idx, arr, rate, al, be)
    miscount, moving_miscount = run_ab.miscount
    rms_oracle, _, _ = score(*run_ab(lib, z, idx, arr, rate, al, be, oracle=True), t_fine, v_fine)
    ideal = deliver(np.random.default_rng(3), frame_t, np.ones(len(frame_t), bool), 0.008, 0.0, 0.0, 0.0)
    rms_ideal, _, _ = score(*run_ab(lib, z, *ideal, rate, al, be), t_fine, v_fine)
    print(f"     same stream, true frame counts: rms {rms_oracle:.4f};  perfect link: rms {rms_ideal:.4f};  "
          f"frame clock miscounts {miscount}/{len(idx)} ({moving_miscount} above 0.5 m/s)")
    check("frame clock within 5% of true frame counting (rms)", rms_ab < 1.05 * rms_oracle)
    check("miscounts above 0.5 m/s < 0.5% of packets", moving_miscount < 0.005 * len(idx))
    src = (SRC / "estimator_mocap.c").read_text()
    fw = {k: float(src.split(f"static float {k} = ")[1].split("f;")[0]) for k in ("alpha", "beta")}
    check("firmware defaults match the best gains", abs(fw["alpha"] - al) < 0.01 and abs(fw["beta"] - be) < 0.01,
          f"(firmware {fw['alpha']}/{fw['beta']})")

    def counts(gaps, frames=None, v=0.0):
        """Feed arrival gaps [periods]; positions follow the true frame indices `frames` at speed v (x axis)."""
        f, clk = AbFilter(), FrameClock()
        lib.abReset(ctypes.byref(f), F3(0, 0, 0))
        f.v[0] = v
        frames = frames or list(range(1, len(gaps) + 1))
        out = []
        for g, k in zip(gaps, frames):
            zk = F3(v * k * T, 0, 0)
            out.append(lib.abFramesElapsed(ctypes.byref(f), ctypes.byref(clk), zk, g * T, T, 0.7, 0.01, 0.0015))
            lib.abUpdate(ctypes.byref(f), zk, out[-1] * T, 0.5, 0.15)
        return out
    check("late packet (+0.6 T) then on time -> 1, 1", counts([1.6, 0.4, 1.0]) == [1, 1, 1], f"{counts([1.6, 0.4, 1.0])}")
    check("dropped frame -> 2", counts([1.0, 2.0, 1.0]) == [1, 2, 1], f"{counts([1.0, 2.0, 1.0])}")
    check("dropped frame + late (+0.6 T) -> 2, then 1", counts([2.6, 0.4, 1.0]) == [2, 1, 1], f"{counts([2.6, 0.4, 1.0])}")
    check("burst of 3 late packets -> 1, 1, 1, 1", counts([1.65, 0.01, 0.01, 1.33]) == [1, 1, 1, 1],
          f"{counts([1.65, 0.01, 0.01, 1.33])}")
    check("moving 1 m/s: very late packet (+0.9 T) is 1 frame, not a drop", counts([1.9], [1], v=1.0) == [1],
          f"{counts([1.9], [1], v=1.0)}")
    check("moving 1 m/s: early-looking drop (gap 1.6 T after a late one) is 2 frames",
          counts([1.6, 1.6], [1, 3], v=1.0) == [1, 2], f"{counts([1.6, 1.6], [1, 3], v=1.0)}")

    print("2. sign conventions and limits")
    gains = Gains(kv=F3(4, 4, 6), ki=F3(0, 0, 0), iLimit=F3(1, 1, 2), maxTiltRad=np.radians(20), maxAccUp=6, maxAccDown=6)

    def attitude(a, yaw_deg):
        zd, lift = F3(), ctypes.c_float()
        lib.velCtlAccelToZb(ctypes.byref(gains), F3(*a), zd, ctypes.byref(lift))
        r, pch = ctypes.c_float(), ctypes.c_float()
        lib.velCtlZbToRollPitch(zd, np.radians(yaw_deg), ctypes.byref(r), ctypes.byref(pch))
        return np.degrees(r.value), np.degrees(pch.value), np.array(zd), lift.value

    r, pch, _, _ = attitude((1, 0, 0), 0)
    check("+x accel at yaw 0 -> nose down (pitch > 0, legacy pitch < 0), no roll", pch > 0 and abs(r) < 1e-4,
          f"(roll {r:.2f}, pitch {pch:.2f})")
    r, pch, _, _ = attitude((0, 1, 0), 0)
    check("+y (left) accel at yaw 0 -> roll < 0 (left side down)", r < 0 and abs(pch) < 1e-4, f"(roll {r:.2f})")
    r, pch, _, _ = attitude((0, 1, 0), 90)
    check("+y accel at yaw 90 -> nose down, no roll", pch > 0 and abs(r) < 1e-3, f"(roll {r:.2f}, pitch {pch:.2f})")
    for yaw in (0, 37, -120):
        r, pch, zd, _ = attitude((1.3, -2.1, 0.7), yaw)
        cr, sr, cp, sp, cy, sy = (np.cos(np.radians(r)), np.sin(np.radians(r)), np.cos(np.radians(pch)),
                                  np.sin(np.radians(pch)), np.cos(np.radians(yaw)), np.sin(np.radians(yaw)))
        zb = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])
        check(f"roll/pitch round-trip to body z (ZYX, yaw {yaw})", np.allclose(zb, zd, atol=1e-5))
    r, pch, _, _ = attitude((30, 0, 0), 0)
    check("tilt clamp at 20 deg", abs(pch - 20) < 1e-3, f"({pch:.3f})")
    _, _, _, lift = attitude((0, 0, 50), 0)
    check("vertical clamp: lift = g + maxAccUp", abs(lift - (G + 6)) < 1e-4, f"({lift:.3f})")
    _, _, _, lift = attitude((0, 0, -50), 0)
    check("vertical clamp: lift = g - maxAccDown", abs(lift - (G - 6)) < 1e-4, f"({lift:.3f})")
    zd, lift = F3(), ctypes.c_float()
    lib.velCtlAccelToZb(ctypes.byref(gains), F3(0, 0, 0), zd, ctypes.byref(lift))
    hover = lib.velCtlThrustN(0.045, zd, lift, F3(0, 0, 1))
    check("hover thrust = m g", abs(hover - 0.045 * G) < 1e-5, f"({hover:.4f} N)")
    check("thrust drops when not yet tilted (projection on current body z)",
          lib.velCtlThrustN(0.045, F3(0.5, 0, np.sqrt(0.75)), 10.0, F3(0, 0, 1)) < 0.045 * 10.0)

    print(f"3. closed loop (kv 4/4/6, tilt lag 30 ms, thrust lag 15 ms, mocap 8 ms, alpha-beta {al:.3f}/{be:.4f})")
    steps = [(1.0, (0.5, 0, 0)), (4.0, (0, 0, 0)), (7.0, (0, 0, 0.3)), (10.0, (0, 0, 0))]
    t, v, _ = closed_loop(lib, gains, 0.045, 0.045, steps, 12.0, alpha=al, beta=be)
    tau_x, d_x, os_x = fit_first_order(t, v[:, 0], 1.0, 0.0, 0.5)
    tau_z, d_z, os_z = fit_first_order(t, v[:, 2], 7.0, 0.0, 0.3)
    ss_x = v[(t > 3.0) & (t < 4.0), 0].mean()
    print(f"     vx step: tau {tau_x * 1e3:.0f} ms (1/kv = 250), delay {d_x * 1e3:.0f} ms, overshoot {os_x * 100:.1f}%, "
          f"steady {ss_x:.3f}")
    print(f"     vz step: tau {tau_z * 1e3:.0f} ms (1/kv = 167), delay {d_z * 1e3:.0f} ms, overshoot {os_z * 100:.1f}%")
    check("vx response is ~first order: tau within 25% of 1/kv", abs(tau_x - 0.25) < 0.0625)
    check("vx overshoot < 10%", os_x < 0.10)
    check("vz overshoot < 10%", os_z < 0.10)
    check("vx steady state within 2% (exact mass, no drag)", abs(ss_x - 0.5) < 0.01, f"({ss_x:.3f})")

    t, v, _ = closed_loop(lib, gains, 0.045, 0.045 * 1.1, [(0.0, (0, 0, 0))], 4.0, alpha=al, beta=be)
    vz_ss = v[t > 3.0, 2].mean()
    predicted = -G * (1.1 - 1) / 6
    print(f"     10% heavier than velCtl.mass: hover vz {vz_ss:.3f} m/s (predicted -g(m/m_ctl - 1)/kvZ = {predicted:.3f})")
    check("mass error -> constant vz bias = -g(m/m_ctl - 1)/kvZ", abs(vz_ss - predicted) < 0.01)
    gains_i = Gains(kv=F3(4, 4, 6), ki=F3(0, 0, 4), iLimit=F3(1, 1, 2), maxTiltRad=np.radians(20), maxAccUp=6, maxAccDown=6)
    t, v, _ = closed_loop(lib, gains_i, 0.045, 0.045 * 1.1, [(0.0, (0, 0, 0))], 6.0, alpha=al, beta=be)
    vz_i = v[t > 5.0, 2].mean()
    check("kiZ removes the mass bias", abs(vz_i) < 0.01, f"(vz {vz_i:.4f})")

    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
