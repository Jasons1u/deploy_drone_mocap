"""The hover policy: observation, ONNX inference, action map. No ROS, so it is tested directly.

Observation (Isaac_Hover_CLF, setpoint frame = target position + target yaw, upright):
    pos_g     = R_ref^T (pos_enu - target_enu)          3
    lin_vel_g = R_ref^T vel_enu                         3
    euler_g   = euler_xyz(q_ref^-1 * q_drone)           3
    ang_vel_w = R_drone ang_vel_b                       3   (world ENU)
    ang_vel_b = body FLU gyro                           3
    ground_prox = 1 / (1 + max(z - gp_z0, 0) / gp_h0)   1   only if the ONNX has state_dim 16
    last_action history, oldest -> newest              4N  UNCLIPPED raw outputs
The ONNX metadata (state_dim, act_hist_len, gp_*, hover_cmd, min_motor_cmd) defines the layout,
so one class runs both the 36-dim (with ground_prox) and 35-dim policies.

ground_prox uses ABSOLUTE height, so world z = 0 must be the floor. With mocap that holds when the
Motive ground plane is on the floor (and, for state_source=px4, when EKF2 uses vision height).

Action map (the trim ships inside the ONNX so it cannot drift from the weights):
    a         = clip(raw, -1, 1)
    motor_cmd = clip(hover_cmd + a * (1 - hover_cmd if a >= 0 else hover_cmd), min_motor_cmd, 1)
motor_cmd is PX4 ActuatorMotors.control for non-reversible motors: 0 = idle .. 1 = full.
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import dataclass

import numpy as np

from . import frames

ACTION_DIM = 4


@dataclass
class DroneState:
    pos_enu: np.ndarray       # (3,) [m]
    vel_enu: np.ndarray       # (3,) [m/s] world
    quat_flu_enu: np.ndarray  # (4,) [w, x, y, z]
    ang_vel_flu: np.ndarray   # (3,) [rad/s] body


@dataclass(frozen=True)
class PolicyOutput:
    raw: np.ndarray        # (4,) unclipped network output
    motor_cmd: np.ndarray  # (4,) [0, 1]
    inf_ms: float


class HoverPolicy:
    def __init__(self, onnx_path: str, providers=("CPUExecutionProvider",), threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=list(providers))
        inp = self.session.get_inputs()[0]
        self._input_name = inp.name
        self.obs_dim = int(inp.shape[1])

        meta = self.session.get_modelmeta().custom_metadata_map
        self.hover_cmd = np.array(json.loads(meta["hover_cmd"]), dtype=np.float64)
        self.min_motor_cmd = float(meta["min_motor_cmd"])
        self.state_dim = int(meta.get("state_dim", 16))
        self.act_hist_len = int(meta.get("act_hist_len", (self.obs_dim - self.state_dim) // 4))
        self.gp_z0 = float(meta.get("gp_z0", 0.4))
        self.gp_h0 = float(meta.get("gp_h0", 0.5))
        if self.state_dim not in (15, 16):
            raise ValueError(f"{onnx_path}: unsupported state_dim {self.state_dim}")
        if self.state_dim + ACTION_DIM * self.act_hist_len != self.obs_dim:
            raise ValueError(
                f"{onnx_path}: obs_dim {self.obs_dim} != state_dim {self.state_dim} + "
                f"4 * act_hist_len {self.act_hist_len} — observation layout unknown")
        self._history = collections.deque(maxlen=self.act_hist_len)
        self.reset()

    def reset(self) -> None:
        """Zero the action history (raw 0 == hover trim). Call on every (re-)engage."""
        self._history.extend(np.zeros(ACTION_DIM) for _ in range(self.act_hist_len))

    def build_obs(self, s: DroneState, target_enu, target_yaw: float) -> np.ndarray:
        q_ref = frames.quat_from_yaw(target_yaw)
        parts = [
            frames.quat_apply_inverse(q_ref, np.asarray(s.pos_enu) - np.asarray(target_enu)),
            frames.quat_apply_inverse(q_ref, s.vel_enu),
            frames.euler_xyz_from_quat(frames.quat_mul(frames.quat_conj(q_ref), s.quat_flu_enu)),
            frames.quat_apply(s.quat_flu_enu, s.ang_vel_flu),
            np.asarray(s.ang_vel_flu, dtype=np.float64),
        ]
        if self.state_dim == 16:
            height = float(s.pos_enu[2])
            parts.append([1.0 / (1.0 + max(height - self.gp_z0, 0.0) / self.gp_h0)])
        parts.extend(self._history)
        return np.concatenate(parts).astype(np.float32)

    def act(self, obs: np.ndarray) -> PolicyOutput:
        """Run the network, push the raw output into the history, map it to motor commands."""
        t0 = time.perf_counter()
        raw = np.asarray(self.session.run(None, {self._input_name: obs[np.newaxis, :]})[0][0],
                         dtype=np.float64)
        inf_ms = (time.perf_counter() - t0) * 1e3
        self._history.append(raw.copy())
        return PolicyOutput(raw, self.motor_map(raw), inf_ms)

    def motor_map(self, raw: np.ndarray) -> np.ndarray:
        a = np.clip(raw, -1.0, 1.0)
        scale = np.where(a >= 0.0, 1.0 - self.hover_cmd, self.hover_cmd)
        return np.clip(self.hover_cmd + a * scale, self.min_motor_cmd, 1.0)
