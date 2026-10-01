"""Deterministic sysID flight controller: mocap pose in, (start_flag, mu) out; no ROS or radio imports."""

from __future__ import annotations

import math
import threading

import numpy as np

from config import Config
from plans import Segment

LOG_HEADER = [
    "t", "phase", "seg", "seg_name", "kind", "settled", "cmd_x", "cmd_y", "cmd_z",
    "r_x", "r_y", "r_z", "v_x", "v_y", "v_z", "vr_x", "vr_y", "vr_z",
    "uv_x", "uv_y", "uv_z", "mu_x", "mu_y", "mu_z", "vbat", "start_flag",
]


def smoothstep(x: float) -> float:
    s = min(max(x, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class SysIdController:
    """wait_pose -> takeoff -> settle -> test segments -> land -> done, with the deploy script's safety guards."""

    def __init__(self, cfg: Config, segments: list[Segment], hover_z: float, dry_run: bool = False):
        self.cfg, self.segments, self.hover_z, self.dry_run = cfg, segments, hover_z, dry_run
        self.kpv = np.array(cfg.kpv, dtype=float)
        self.mu_max = np.array(cfg.mu_max, dtype=float)
        self.mu_slew = np.array(cfg.mu_slew, dtype=float)
        self.lock = threading.Lock()

        self.r = np.zeros(3)
        self.v = np.zeros(3)
        self.v_raw = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.uv = np.zeros(3)
        self.mu_prev = np.zeros(3)
        self.center = np.array([np.nan, np.nan, hover_z])
        self.land_from = np.zeros(3)
        self.vbat = float("nan")
        self.batt_low_since = None

        self.phase = "wait_pose"
        self.phase_t0 = None
        self.seg_idx = 0
        self.seg_t0 = None
        self.seg_wait_t0 = None
        self.settled_flag = 0
        self.start_flag = 0
        self.stopped = False
        self.stop_reason = ""

        self.initialized = False
        self.first_pose_t = self.last_pose_t = self.last_ctrl_t = self.last_change_t = None
        self.last_raw_r = np.zeros(3)
        self.flight_t0 = None
        self.rows: list[list] = []
        self.events: list[str] = []

    # --- public API ---
    @property
    def done(self) -> bool:
        return self.stopped or self.phase == "done"

    def request_stop(self, reason: str) -> None:
        with self.lock:
            if not self.stopped:
                self.stopped, self.stop_reason, self.start_flag = True, reason, 0
                self._event(f"STOP: {reason}")

    def check_watchdog(self, now: float) -> None:
        """Call at ~20 Hz: stale mocap stops (motors off), frozen mocap lands."""
        with self.lock:
            if self.stopped or self.last_pose_t is None or self.phase in ("wait_pose", "done"):
                return
            if now - self.last_pose_t > self.cfg.pose_stale_timeout:
                stale = now - self.last_pose_t
            elif self.last_change_t is not None and now - self.last_change_t > self.cfg.pose_frozen_timeout and self.phase != "land":
                self._event(f"frozen mocap {now - self.last_change_t:.2f}s -> land")
                self._begin("land", now)
                return
            else:
                return
        self.request_stop(f"stale mocap {stale:.3f}s")

    def on_pose(self, now: float, pos, quat) -> tuple[int, np.ndarray]:
        """Process one mocap pose; returns (start_flag, mu) to send."""
        with self.lock:
            return self._on_pose(now, np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))

    # --- internals ---
    def _event(self, msg: str) -> None:
        self.events.append(msg)
        print(f"[sysid] {msg}", flush=True)

    def _begin(self, phase: str, now: float) -> None:
        self.phase, self.phase_t0 = phase, now
        if phase == "land":
            self.land_from = self.r.copy()
        self._event(f"phase -> {phase}")

    def _stop_locked(self, reason: str) -> None:
        if not self.stopped:
            self.stopped, self.stop_reason, self.start_flag = True, reason, 0
            self._event(f"STOP: {reason}")

    def _update_state(self, now: float, pos: np.ndarray, quat: np.ndarray) -> tuple[bool, float]:
        c = self.cfg
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            self._stop_locked("non-finite pose")
            return False, 0.0
        if not self.initialized:
            self.r[:], self.q[:] = pos, quat
            self.first_pose_t = self.last_pose_t = self.last_ctrl_t = self.last_change_t = now
            self.last_raw_r[:] = pos
            self.initialized = True
            return True, 0.0
        dt = now - self.last_ctrl_t
        self.last_ctrl_t = self.last_pose_t = now
        # a tracked drone always jitters; a bit-identical repeat means Motive lost the body
        if np.linalg.norm(pos - self.last_raw_r) > c.pose_frozen_eps:
            self.last_change_t = now
            self.last_raw_r[:] = pos
        disp = pos - self.r
        if np.linalg.norm(disp) > c.max_pose_jump:
            self._stop_locked(f"pose jump {np.linalg.norm(disp):.2f} m")
            return False, dt
        self.r[:], self.q[:] = pos, quat
        if c.dt_min <= dt <= c.dt_max:
            self.v_raw[:] = disp / dt
            self.v += (1.0 - math.exp(-2.0 * math.pi * c.vel_cutoff_hz * dt)) * (self.v_raw - self.v)
        return True, dt

    def _geofence_ok(self) -> bool:
        if not np.all(np.isfinite(self.center)):
            return True
        horiz = float(np.linalg.norm(self.r[:2] - self.center[:2]))
        # on the ground mocap z can read slightly below zero (calibration, noise)
        z_lo = -0.05 if self.phase in ("takeoff", "land") else self.cfg.z_min
        return horiz <= self.cfg.r_max and z_lo <= self.r[2] <= self.cfg.z_max

    def _battery_guard(self, now: float) -> None:
        if self.phase not in ("takeoff", "settle", "test"):
            self.batt_low_since = None
            return
        if math.isfinite(self.vbat) and self.vbat < self.cfg.batt_min_fly:
            self.batt_low_since = now if self.batt_low_since is None else self.batt_low_since
            if now - self.batt_low_since >= self.cfg.batt_low_hold:
                self._event(f"battery {self.vbat:.2f} V -> land")
                self._begin("land", now)
        else:
            self.batt_low_since = None

    def _hold(self, target: np.ndarray) -> np.ndarray:
        c = self.cfg
        err = target - self.r
        cmd = np.array([c.kpr_xy * err[0], c.kpr_xy * err[1], c.kpr_z * err[2]])
        cmd[:2] = np.clip(cmd[:2], -c.hold_uv_max_xy, c.hold_uv_max_xy)
        cmd[2] = float(np.clip(cmd[2], -c.hold_uv_max_z, c.hold_uv_max_z))
        return cmd

    def _settled(self) -> bool:
        return (np.linalg.norm(self.v) < self.cfg.settled_speed
                and np.linalg.norm(self.r - self.center) < self.cfg.settled_dist)

    def _schedule(self, now: float) -> tuple[np.ndarray, np.ndarray | None, Segment | None, int]:
        """Return (uv, open-loop mu or None, active segment or None, its index)."""
        c = self.cfg
        if self.phase_t0 is None:
            self._begin(self.phase, now)
        elapsed = now - self.phase_t0

        if self.phase == "wait_pose":
            self.start_flag = 0
            if now - self.first_pose_t >= c.pose_warmup:
                if now - self.last_change_t > c.pose_frozen_timeout:
                    self._stop_locked("mocap pose frozen before arming -- check the rigid body in Motive")
                elif not self.dry_run and not math.isfinite(self.vbat):
                    self._stop_locked("no battery reading -- refusing to arm")
                elif not self.dry_run and self.vbat < c.batt_min_arm:
                    self._stop_locked(f"battery {self.vbat:.2f} V below arm limit {c.batt_min_arm:.2f} V")
                else:
                    self.center[:2], self.center[2] = self.r[:2], self.hover_z
                    self.flight_t0, self.start_flag = now, 1
                    self._event(f"armed, center {np.round(self.center, 3).tolist()}, vbat {self.vbat:.2f}")
                    self._begin("takeoff", now)
            return np.zeros(3), None, None, -1

        if self.phase == "takeoff":
            self.start_flag = 1
            tgt = np.array([self.center[0], self.center[1],
                            c.land_z + (self.hover_z - c.land_z) * smoothstep(elapsed / c.takeoff_time)])
            if elapsed >= c.takeoff_time:
                self._begin("settle", now)
            return self._hold(tgt), None, None, -1

        if self.phase == "settle":
            if elapsed >= c.settle_time:
                self.seg_idx, self.seg_t0, self.seg_wait_t0 = 0, None, now
                self._begin("test", now)
            return self._hold(self.center), None, None, -1

        if self.phase == "test":
            seg_i = self.seg_idx
            seg = self.segments[seg_i]
            if self.seg_t0 is None:
                ok = not seg.settled_entry or self._settled()
                if ok or now - self.seg_wait_t0 >= c.settle_wait_max:
                    self.seg_t0, self.settled_flag = now, int(ok)
                    if not ok:
                        self._event(f"{seg.name}: not settled after {c.settle_wait_max:.0f}s, starting anyway")
                else:
                    return self._hold(self.center), None, None, -1
            seg_elapsed = now - self.seg_t0
            uv, open_mu = self._hold(self.center), None
            if seg.kind == "brake":
                uv = np.zeros(3)
            elif seg.kind == "vel":
                for ax in seg.axes:
                    uv[ax] = seg.cmd[ax]
            elif seg.kind == "mu_open":
                runaway = (abs(self.v[2]) > c.open_vz_abort or self.r[2] > c.z_max - c.open_z_margin
                           or self.r[2] < c.z_min + c.open_z_margin)
                if runaway:
                    self._event(f"{seg.name}: open-loop step ended early (z {self.r[2]:.2f}, vz {self.v[2]:.2f})")
                    seg_elapsed = seg.duration
                else:
                    open_mu = np.array(seg.cmd, dtype=float)
            if seg_elapsed >= seg.duration:
                self.seg_idx += 1
                self.seg_t0, self.seg_wait_t0 = None, now
                if self.seg_idx >= len(self.segments):
                    self._begin("land", now)
                return uv, None, seg, seg_i
            return uv, open_mu, seg, seg_i

        if self.phase == "land":
            tgt = np.array([self.land_from[0], self.land_from[1],
                            self.land_from[2] + (c.land_z - self.land_from[2]) * smoothstep(elapsed / c.land_time)])
            if elapsed >= c.land_time:
                self.start_flag = 0
                self._begin("done", now)
            return self._hold(tgt), None, None, -1

        self.start_flag = 0
        return np.zeros(3), None, None, -1

    def _on_pose(self, now: float, pos: np.ndarray, quat: np.ndarray) -> tuple[int, np.ndarray]:
        c = self.cfg
        valid, dt = self._update_state(now, pos, quat)
        if not valid or self.stopped:
            return 0, np.zeros(3)
        if self.phase not in ("wait_pose", "done", "land") and not self._geofence_ok():
            if self.dry_run:
                # motors cannot arm and the drone sits on the bench, so only note it
                if not getattr(self, "_fence_noted", False):
                    self._fence_noted = True
                    self._event(f"(dry run) geofence would land at {np.round(self.r, 2).tolist()}")
            else:
                self._event(f"geofence breach at {np.round(self.r, 2).tolist()} -> land")
                self._begin("land", now)
        self._battery_guard(now)

        uv, open_mu, seg, seg_i = self._schedule(now)
        if self.stopped:
            return 0, np.zeros(3)
        self.uv[:] = uv
        if open_mu is None:
            desired = self.kpv * (uv - self.v)
            desired[2] += c.mu_z_hover
        else:
            desired = open_mu.copy()
        desired = np.clip(desired, -self.mu_max, self.mu_max)
        step = self.mu_slew * (dt if c.dt_min <= dt <= c.dt_max else 0.01)
        mu = np.clip(self.mu_prev + np.clip(desired - self.mu_prev, -step, step), -self.mu_max, self.mu_max)
        self.mu_prev[:] = mu
        flag = 0 if self.dry_run else int(self.start_flag)

        t_log = 0.0 if self.flight_t0 is None else now - self.flight_t0
        cmd = seg.cmd if seg is not None else (0.0, 0.0, 0.0)
        self.rows.append([
            round(t_log, 5), self.phase, seg_i,
            seg.name if seg is not None else "", seg.kind if seg is not None else "", self.settled_flag,
            *cmd, *self.r, *self.v, *self.v_raw, *self.uv, *mu, self.vbat, flag,
        ])
        return flag, mu
