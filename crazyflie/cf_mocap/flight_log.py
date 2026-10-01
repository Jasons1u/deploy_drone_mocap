"""One CSV row per control tick: what was commanded, what mocap saw, what the drone estimated.

mocap_* and est_* columns are the same quantities in the same frame (mocap world, right-handed
ZYX angles, pitch > 0 = nose down), so est - mocap is directly the onboard filter's error.
m1..m4 are motor PWM (0..65535). cmd_* is the host controller's attitude/thrust output (mocap-state
mode; NaN with --kalman, where the drone computes its own). est_* is NaN in mocap-state mode (no
onboard position estimate). Setpoint columns are NaN when nothing is commanded (filter test).
Next to each CSV, a .yaml of the exact config it ran with (including --set overrides).
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import yaml

from mocap_natnet.frames import rpy_from_quat

COLUMNS = (
    "t", "phase",
    "sp_x", "sp_y", "sp_z", "sp_yaw_deg",
    "mocap_x", "mocap_y", "mocap_z", "mocap_vx", "mocap_vy", "mocap_vz",
    "mocap_roll_deg", "mocap_pitch_deg", "mocap_yaw_deg", "mocap_age_ms",
    "est_x", "est_y", "est_z", "est_vx", "est_vy", "est_vz",
    "est_roll_deg", "est_pitch_deg", "est_yaw_deg",
    "cmd_roll_deg", "cmd_pitch_deg", "cmd_yawrate_deg_s", "cmd_thrust",
    "m1", "m2", "m3", "m4",
    "battery_v",
)


def _or_nan(values, n):
    return tuple(values) if values is not None else (math.nan,) * n


class FlightLog:
    def __init__(self, log_dir: Path, prefix: str = "hover", config=None):
        """config: the Config that ran, written next to the CSV as <name>.yaml."""
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        if config is not None:
            self.path.with_suffix(".yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
        self._file = None
        self._writer = None
        self._t0 = None

    def __enter__(self) -> "FlightLog":
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        return self

    def __exit__(self, *exc):
        self._file.close()

    def write(self, t: float, phase: str, sp, pose, mocap_age: float, telemetry, cmd=None) -> None:
        """sp: flight_plan.Setpoint or None; pose: mocap_natnet.Pose or None;
        cmd: mocap_control.AttitudeCommand or None."""
        if self._t0 is None:
            self._t0 = t
        setpoint = (sp.x, sp.y, sp.z, sp.yaw_deg) if sp is not None else (math.nan,) * 4
        if pose is not None:
            mocap = (*pose.position, *pose.velocity,
                     *(math.degrees(a) for a in rpy_from_quat(pose.quat)))
        else:
            mocap = (math.nan,) * 9
        self._writer.writerow((
            f"{t - self._t0:.4f}", phase,
            *setpoint,
            *mocap, mocap_age * 1e3,
            *_or_nan(telemetry.position, 3), *_or_nan(telemetry.velocity, 3),
            *_or_nan(telemetry.rpy_deg, 3),
            *((cmd.roll_deg, cmd.pitch_deg, cmd.yawrate_deg_s, cmd.thrust) if cmd is not None
              else (math.nan,) * 4),
            *_or_nan(telemetry.motors, 4),
            telemetry.battery_v if telemetry.battery_v is not None else math.nan,
        ))
