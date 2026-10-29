"""Rate-mode tapes: rows [thrust / hover, p, q, r rad/s] at 100 Hz, their legacy setpoints, and the cf21bl mixer geometry."""

from __future__ import annotations

import numpy as np

PERIOD = 0.01                     # [s] one row per setpoint
MAX_THRUST = 0.8                  # [N] powerDistributionGetMaxThrust() on cf21bl: 4 x THRUST_MAX
ARM = 0.05 / np.sqrt(2)           # [m] ARM_LENGTH projected on body x and y
THRUST2TORQUE = 0.00569           # [m] yaw torque per newton (platform_defaults_cf21bl.h)
# [T, tau_x, tau_y, tau_z] = ALLOC f for motor forces M1..M4 at (+x,-y), (-x,-y), (-x,+y), (+x,+y): inverse of powerDistributionForceTorque
ALLOC = np.array([[1.0, 1.0, 1.0, 1.0],
                  [-ARM, -ARM, ARM, ARM],
                  [-ARM, ARM, ARM, -ARM],
                  [-THRUST2TORQUE, THRUST2TORQUE, -THRUST2TORQUE, THRUST2TORQUE]])


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


def flip_tape(path, pad: int = 10) -> tuple:
    """mj-nlp crazyflie_flip .npz -> (rows, tape hover [N], plan): the maneuver between its pinned hover ends, pad hover rows each side."""
    d = np.load(path, allow_pickle=True)
    if not {"state_rollout", "inertia", "motor_lag"} <= set(d.files):
        raise ValueError(f"{path}: no planned rates / inertia / motor lag; re-solve with the current mj-nlp crazyflie_flip.py")
    U, dt = np.asarray(d["input"], float), float(d["time"][1] - d["time"][0])
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
    plan = dict(t=dt * np.arange(len(X)), x=X, w=X[:, 10:13], J=np.asarray(d["inertia"], float), tau_m=float(d["motor_lag"]))
    return rows, float(hover[0]), plan


def doublet(col: int, amp: float, dur: float) -> np.ndarray:
    """+amp then -amp on one column (0: thrust as a fraction of hover, 1-3: p q r [rad/s]) for dur [s] each, then a hover row."""
    n = int(round(dur / PERIOD))
    rows = np.zeros((2 * n + 1, 4))
    rows[:, 0] = 1.0
    rows[:n, col] += amp
    rows[n:2 * n, col] -= amp
    return rows

