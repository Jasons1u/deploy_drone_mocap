"""Closed-loop velocity model of the CF2.1 Brushless + mocap_velctl firmware (the sysID fit, as Isaac's VelCtlAction)."""

from __future__ import annotations

import math

import numpy as np

from cfdeploy.safe_cf_obs import V_MAX

G = 9.81


# #### // 3-axis model stepped like Isaac: integer-step delays, used by the sim // ####
class VelocityModel:
    """Isaac VelCtlAction for one env: a_cmd = kv*(cmd(t-d_cmd) - v(t-d_fb)), capped, lagged by tau_a; v' = a - drag*v."""

    BUF = 32

    def __init__(self, model_cfg, dt: float):
        self.cfg, self.dt = model_cfg, dt
        self.kv_nom = np.array(model_cfg.kv, float)

    def reset(self, p0, rng=None) -> "VelocityModel":
        """Nominal sysID values, or (rng) one draw from the training ranges with drift and kicks."""
        c = self.cfg
        d = {k: float(rng.uniform(*r)) for k, r in c.ranges.items()} if rng is not None else dict(c.nominal)
        steps = lambda s: int(round(s / self.dt))
        self.params, self.rng = d, rng
        self.kv = self.kv_nom * d["kv_scale"]
        self.tau = np.array([d["tau_xy"], d["tau_xy"], d["tau_z"]])
        self.a_xy, self.drag = d["a_xy"], np.array([0.0, 0.0, d["drag_z"]])
        self.d_cmd = np.array([steps(d["d_cmd_xy"]), steps(d["d_cmd_xy"]), steps(d["d_cmd_z"])])
        self.d_fb = steps(d["d_fb"])
        sigma, clip = np.array(c.drift_sigma, float), np.array(c.drift_clip, float)
        self.bias = np.zeros(3) if rng is None else np.clip(rng.normal(0.0, sigma), -clip, clip)
        self.cmd_buf, self.v_buf, self.i = np.zeros((self.BUF, 3)), np.zeros((self.BUF, 3)), 0
        self.p, self.v, self.a = np.array(p0, float), np.zeros(3), np.zeros(3)
        return self

    def step(self, cmd) -> None:
        c, dt, i = self.cfg, self.dt, self.i
        self.cmd_buf[i], self.v_buf[i] = cmd, self.v
        delayed = self.cmd_buf[(i - self.d_cmd) % self.BUF, np.arange(3)]
        v_fb = self.v_buf[(i - self.d_fb) % self.BUF]
        self.i = (i + 1) % self.BUF
        a_cmd = self.kv * (delayed - v_fb)
        h = math.hypot(a_cmd[0], a_cmd[1])
        if h > self.a_xy:
            a_cmd[:2] *= self.a_xy / h
        a_cmd[2] = min(max(a_cmd[2], -min(c.a_z_down, G)), c.a_z_up)
        self.a = self.a + (a_cmd - self.a) * (dt / self.tau)
        self.v = np.clip(self.v + (self.a - self.drag * self.v) * dt, -V_MAX, V_MAX)
        if self.rng is not None and self.rng.random() < c.kick_prob:
            self.v = np.clip(self.v + self.rng.normal(0.0, np.array(c.kick_std, float)), -V_MAX, V_MAX)
        self.p = self.p + (self.v + self.bias) * dt


# #### // 1-axis step response with a continuous delay, used to fit the model (fit_vsteps.py) // ####
def step_response(t_rel, v0, a0, v_from, v_to, kv, tau_a, d_cmd, a_max, drag, d_fb, dt=0.0025):
    """Same cascade on one axis; t=0 at the step, the command switch interpolated within a step so d_cmd is continuous."""
    n_pre, n = int(round(0.3 / dt)), int(round(t_rel[-1] / dt)) + 2
    hist = int(round(d_fb / dt)) + 1
    vs, v, a, out = [v0] * hist, v0, a0, np.empty(n)
    for k in range(-n_pre, n):
        frac = min(max((k * dt - d_cmd) / dt + 1.0, 0.0), 1.0)
        cmd = v_from + (v_to - v_from) * frac
        a_cmd = np.clip(kv * (cmd - vs[-hist]), -a_max, a_max)
        a += (a_cmd - a) * dt / tau_a
        v += (a - drag * v) * dt
        if k < 0:
            v, a = v0, a0
        vs.append(v)
        if k >= 0:
            out[k] = v
    return np.interp(t_rel, np.arange(n) * dt, out)
