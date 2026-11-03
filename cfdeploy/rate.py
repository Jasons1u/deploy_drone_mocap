"""CF2.1 BL rate mode: tapes of [thrust / hover, p, q, r] rows at 100 Hz, their legacy setpoints, and the firmware rate loop as flown."""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import butter

PERIOD = 0.01                     # [s] one row per setpoint
MAX_THRUST = 0.8                  # [N] powerDistributionGetMaxThrust() on cf21bl: 4 x THRUST_MAX
ARM = 0.05 / np.sqrt(2)           # [m] ARM_LENGTH projected on body x and y
THRUST2TORQUE = 0.00569           # [m] yaw torque per newton (platform_defaults_cf21bl.h)
# [T, tau_x, tau_y, tau_z] = ALLOC f for motor forces M1..M4 at (+x,-y), (-x,-y), (-x,+y), (+x,+y): inverse of powerDistributionForceTorque
ALLOC = np.array([[1.0, 1.0, 1.0, 1.0],
                  [-ARM, -ARM, ARM, ARM],
                  [-ARM, ARM, ARM, -ARM],
                  [-THRUST2TORQUE, THRUST2TORQUE, -THRUST2TORQUE, THRUST2TORQUE]])

DT = 0.0005                       # [s] model step: gyro filter at 1 kHz, PID at 500 Hz
PWM_N = MAX_THRUST / 4 / 65535    # [N] per-motor thrust per PWM unit
# per body axis, deg/s units (platform_defaults_cf21bl.h): kp, ki, kd on the measurement, integral limit
GAINS = np.array([[200.0, 400.0, 2.5, 33.3], [200.0, 400.0, 2.5, 33.3], [120.0, 16.7, 0.0, 166.7]])
TORQUE = PWM_N * np.array([2 * ARM, 2 * ARM, 4 * THRUST2TORQUE])   # [N m] per unit of PID output through the legacy mixer
OUT_MAX = np.array([32767.0, 32767.0, 0.15 * 65535])                # int16 output; yaw clamped by velCtl.yawLim
MOTOR = (0.02136, MAX_THRUST / 4)                                    # [N] idle and full per-motor thrust
MIX = np.linalg.inv(ALLOC)
LPF = butter(2, 80 / 500)                                            # lpf2p: 80 Hz at 1 kHz


# #### // Setpoints: what the host sends and what the rate PID receives // ####
def setpoint(thrust, w) -> tuple:
    """Thrust [N] and body rates (p, q, r) [rad/s] -> cflib send_setpoint(roll, pitch, yawrate, thrust) with flightmode rate mode."""
    p, q, r = np.degrees(w)
    pwm = int(round(min(max(thrust / MAX_THRUST, 0.0), 1.0) * 65535))
    return float(p), float(q), float(-r), pwm   # cflib negates pitch and the commander negates yaw, so only r flips


def firmware_view(roll, pitch, yawrate, thrust) -> tuple:
    """What the rate PID receives from one send_setpoint: (thrust [N], body rates [rad/s]); cflib -> crtp_commander_rpyt -> controller_velocity."""
    packed = (roll, -pitch, yawrate)                     # cflib packs -pitch
    rate = (packed[0], packed[1], -packed[2])            # commander rate mode: attitudeRate = values, yaw inverted
    body = (rate[0], -rate[1], rate[2])                  # rate PID tracks (gyro.x, -gyro.y, gyro.z)
    return thrust / 65535 * MAX_THRUST, np.radians(body)


# #### // Tapes: mj-nlp flips and sysID doublets // ####
def flip_tape(path, pad: int = 10) -> tuple:
    """mj-nlp crazyflie_flip .npz -> (rows, tape hover [N], plan): the maneuver between its pinned hover ends, pad hover rows each side."""
    d = np.load(path, allow_pickle=True)
    if not {"state_rollout", "inertia", "motor_lag"} <= set(d.files):
        raise ValueError(f"{path}: no planned rates / inertia / motor lag; re-solve with the current mj-nlp crazyflie_flip.py")
    U, dt = np.asarray(d["rate_tape"] if "rate_tape" in d.files else d["input"], float), float(d["time"][1] - d["time"][0])
    step = int(round(PERIOD / dt))
    rows = U[::step]
    if not np.allclose(U, np.repeat(rows, step, axis=0)[:len(U)]):
        raise ValueError(f"{path}: input is not held over {PERIOD * 1e3:.0f} ms")
    hover = rows[0]
    if np.any(hover[1:]) or not np.array_equal(rows[-1], hover):
        raise ValueError(f"{path}: tape does not start and end at hover (solved before crazyflie_flip.HOLD?)")
    moving = np.flatnonzero(np.any(rows != hover, axis=1))
    if moving[0] < pad or len(rows) - 1 - moving[-1] < pad:
        raise ValueError(f"{path}: fewer than {pad} hover rows at an end")
    i0, i1 = moving[0] - pad, moving[-1] + pad + 1
    rows = rows[i0:i1].copy()
    rows[:, 0] /= hover[0]
    X = np.asarray(d["state_rollout"], float)[i0 * step:i1 * step + 1]
    plan = dict(t=dt * np.arange(len(X)), x=X, w=X[:, 10:13], J=np.asarray(d["inertia"], float), tau_m=float(d["motor_lag"]),
                lagged="lag_in_plan" in d.files and bool(d["lag_in_plan"]))
    return rows, float(hover[0]), plan


def doublet(col: int, amp: float, dur: float) -> np.ndarray:
    """+amp then -amp on one column (0: thrust as a fraction of hover, 1-3: p q r [rad/s]) for dur [s] each, then a hover row."""
    n = int(round(dur / PERIOD))
    rows = np.zeros((2 * n + 1, 4))
    rows[:, 0] = 1.0
    rows[:n, col] += amp
    rows[n:2 * n, col] -= amp
    return rows


# #### // Firmware loop: 500 Hz rate PID on the 80 Hz-filtered gyro, legacy mixer with cap, first-order motors // ####
class RateLoop:
    """n firmware rate loops in parallel: step(w, w_cmd, T_cmd) every DT -> (T, tau_x, tau_y, tau_z) the motors deliver, (n, 4)."""

    def __init__(self, n: int, tau_m: float, T0: float, w0=None):
        w0 = np.zeros((n, 3)) if w0 is None else np.asarray(w0, float)
        self.k, self.alpha = 0, 1.0 - np.exp(-DT / tau_m)
        self.wf, self.prev, self.integ = w0.copy(), w0.copy(), np.zeros((n, 3))
        self.lpf = np.stack([w0, w0, w0, w0])   # x[n-1], x[n-2], y[n-1], y[n-2]
        self.f_cmd = self.f = np.full((n, 4), T0 / 4)

    def step(self, w, w_cmd, T_cmd) -> np.ndarray:
        (b, a), (x1, x2, y1, y2) = LPF, self.lpf
        if self.k % 2 == 0:
            self.wf = b[0] * w + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            self.lpf = np.stack([w, x1, self.wf, y1])
        if self.k % 4 == 0:
            e = np.degrees(w_cmd - self.wf)
            self.integ = np.clip(self.integ + e * 4 * DT, -GAINS[:, 3], GAINS[:, 3])
            out = GAINS[:, 0] * e + GAINS[:, 1] * self.integ - GAINS[:, 2] * np.degrees(self.wf - self.prev) / (4 * DT)
            self.prev = self.wf
            f = np.column_stack([np.broadcast_to(T_cmd, len(e)), TORQUE * np.clip(out, -OUT_MAX, OUT_MAX)]) @ MIX.T
            self.f_cmd = np.maximum(f - np.maximum(f.max(axis=1, keepdims=True) - MOTOR[1], 0.0), MOTOR[0])   # powerDistributionCap
        self.k += 1
        self.f = self.f + (self.f_cmd - self.f) * self.alpha
        return self.f @ ALLOC.T


def simulate(cmd, J, tau_m: float, T: float = 0.43, w0=None) -> tuple:
    """Decoupled rigid axes (J (3,) [kg m^2]) under per-step setpoints cmd (steps, n, 3): (body rates, filtered gyro), each like cmd."""
    loop = RateLoop(cmd.shape[1], tau_m, T, w0)
    w, rates, gyro = loop.wf.copy(), np.empty_like(cmd), np.empty_like(cmd)
    for k in range(len(cmd)):
        w = w + loop.step(w, cmd[k], T)[:, 1:] / J * DT
        rates[k], gyro[k] = w, loop.wf
    return rates, gyro


def compensate(rows, plan: dict, hover: float, pad: int = 10, angle: float = 30.0, smooth: float = 0.02) -> np.ndarray:
    """Rows the firmware flies as planned: p, q onto plan["w"] and its angle, thrust through the motor lag unless the plan already was."""
    m, P = len(rows) + pad, int(round(PERIOD / DT))
    out = np.vstack([rows, np.tile([1.0, 0.0, 0.0, 0.0], (pad, 1))])
    t, pulses = np.arange(m * P) * DT, np.repeat(np.eye(m), P, axis=0)   # pulses: row i held over its PERIOD
    R = smooth * np.diff(np.eye(m), axis=0)
    cmd = np.zeros((m * P, 2 * m, 3))
    cmd[:, :m, 0], cmd[:, m:, 1] = pulses, pulses
    rates = simulate(cmd, plan["J"], plan["tau_m"], hover)[0]
    for j in (0, 1):   # r is left as is: the yaw loop is torque-limited and the flip holds r = 0
        H, target = rates[:, j * m:(j + 1) * m, j], np.interp(t, plan["t"], plan["w"][:, j], right=0.0)
        A = np.vstack([H, angle * DT * np.cumsum(H, axis=0), R])
        out[:, 1 + j] = lsq_linear(A, np.r_[target, angle * DT * np.cumsum(target), np.zeros(m - 1)], bounds=(-35.0, 35.0)).x
    if plan.get("lagged"):   # planned through the motor lag: its thrust is already the command
        return out
    a, L, y = 1.0 - np.exp(-DT / plan["tau_m"]), np.empty_like(pulses), np.zeros(m)
    for k in range(m * P):
        y = y + (pulses[k] - y) * a
        L[k] = y
    lo, hi = 4 * MOTOR[0] / hover - 1.0, MAX_THRUST / hover - 1.0
    out[:, 0] = 1.0 + lsq_linear(np.vstack([L, R]), np.r_[pulses @ (out[:, 0] - 1.0), np.zeros(m - 1)], bounds=(lo, hi)).x
    return out
