"""Takeoff -> hover -> land as a pure function of time. No I/O, so it is unit-tested directly.

The plan is anchored at the takeoff pose captured from mocap (the same "handover" idea as the PX4
hover policy): XY and yaw are held where the drone sits, so nothing moves sideways or yaws at
takeoff. Only z ramps, with a smoothstep so velocity is continuous at both ends.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Phase(enum.Enum):
    TAKEOFF = "takeoff"
    HOVER = "hover"
    LAND = "land"
    DONE = "done"


@dataclass(frozen=True)
class Setpoint:
    x: float
    y: float
    z: float
    yaw_deg: float
    vz: float = 0.0   # [m/s] feed-forward: rate of the z ramp


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def smoothstep_rate(s: float) -> float:
    """d smoothstep / ds."""
    return 6.0 * s * (1.0 - s) if 0.0 < s < 1.0 else 0.0


class HoverPlan:
    def __init__(self, origin, yaw_deg: float, hover_height: float, takeoff_time: float,
                 land_time: float, hover_duration: float):
        """origin: takeoff position (x, y, z) in the mocap world. hover_duration 0 = until land()."""
        self._x, self._y, self._z0 = (float(v) for v in origin)
        self._yaw = float(yaw_deg)
        self._height = hover_height
        self._takeoff_time = takeoff_time
        self._land_time = land_time
        self._hover_duration = hover_duration
        self._t_start = None
        self._t_land = None
        self._z_land_start = None

    def start(self, t: float) -> None:
        self._t_start = t

    @property
    def landing(self) -> bool:
        return self._t_land is not None

    def land(self, t: float) -> None:
        """Begin landing from wherever the z setpoint is now (works mid-takeoff too)."""
        if self._t_land is None:
            self._z_land_start = self._z_commanded(t)
            self._t_land = t

    def step(self, t: float) -> tuple[Phase, Setpoint]:
        if self._t_start is None:
            raise RuntimeError("HoverPlan.start() not called")
        auto_land_at = self._t_start + self._takeoff_time + self._hover_duration
        if self._hover_duration > 0 and self._t_land is None and t >= auto_land_at:
            self.land(auto_land_at)
        phase = self._phase(t)
        return phase, Setpoint(self._x, self._y, self._z_for(phase, t), self._yaw,
                               self._vz_for(phase, t))

    def _phase(self, t: float) -> Phase:
        if self._t_land is not None:
            return Phase.DONE if t - self._t_land >= self._land_time else Phase.LAND
        return Phase.TAKEOFF if t - self._t_start < self._takeoff_time else Phase.HOVER

    def _z_commanded(self, t: float) -> float:
        return self._z0 + self._height * smoothstep((t - self._t_start) / self._takeoff_time)

    def _vz_for(self, phase: Phase, t: float) -> float:
        if phase in (Phase.LAND, Phase.DONE):
            s = (t - self._t_land) / self._land_time
            return (self._z0 - self._z_land_start) * smoothstep_rate(s) / self._land_time
        s = (t - self._t_start) / self._takeoff_time
        return self._height * smoothstep_rate(s) / self._takeoff_time

    def _z_for(self, phase: Phase, t: float) -> float:
        if phase in (Phase.LAND, Phase.DONE):
            frac = smoothstep((t - self._t_land) / self._land_time)
            return self._z_land_start + (self._z0 - self._z_land_start) * frac
        return self._z_commanded(t)
