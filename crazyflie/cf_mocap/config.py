"""Typed view of config/hover.yaml. Unknown or missing keys fail loudly at load time."""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mocap_natnet.frames import quat_from_rpy, quat_mul

PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "hover.yaml"


@dataclass
class RadioConfig:
    uri: str
    cache_dir: str


@dataclass
class MocapConfig:
    server_ip: str | None
    local_ip: str
    rigid_body_id: int
    up_axis: str
    natnet_version: str | None
    extpose_hz: float
    ext_pos_std_dev: float
    ext_quat_std_dev: float
    body_yaw_offset_deg: float          # rigid body x axis -> drone nose (--calibrate-yaw)
    body_tilt_offset_deg: list[float]   # [roll, pitch]: rigid body axes -> IMU axes (--calibrate-tilt)

    def body_rotation(self):
        """The fixed body-frame correction applied to every mocap orientation:
        q_drone = q_motive * yaw_offset * tilt_offset (the tilt is expressed in the yaw-corrected
        body frame)."""
        roll, pitch = (math.radians(a) for a in self.body_tilt_offset_deg)
        return quat_mul(quat_from_rpy(0.0, 0.0, math.radians(self.body_yaw_offset_deg)),
                        quat_from_rpy(roll, pitch, 0.0))


@dataclass
class FlightConfig:
    state_source: str
    controller: str
    hover_height: float
    hover_duration: float
    takeoff_time: float
    land_time: float
    setpoint_hz: float


@dataclass
class HostControlConfig:
    hover_thrust: float
    thrust_min: float
    thrust_max: float
    kp_xy: float
    kp_z: float
    kv_xy: float
    kv_z: float
    ki_xy: float
    ki_z: float
    i_limit_xy: float
    i_limit_z: float
    max_vel_xy: float
    max_vel_z: float
    max_accel_z: float
    max_tilt_deg: float
    kp_yaw: float
    max_yawrate_deg_s: float


@dataclass
class Geofence:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def __post_init__(self):
        for name in ("x", "y", "z"):
            lo, hi = getattr(self, name)
            if lo >= hi:
                raise ValueError(f"geofence.{name}: need [min, max], got {[lo, hi]}")
            setattr(self, name, (float(lo), float(hi)))

    def contains(self, p) -> bool:
        return all(lo <= v <= hi for v, (lo, hi) in zip(p, (self.x, self.y, self.z)))


@dataclass
class SafetyConfig:
    mocap_timeout: float
    preflight_max_error: float
    max_estimate_error: float
    max_preflight_tilt_deg: float
    max_attitude_mismatch_deg: float
    max_flight_tilt_deg: float
    min_battery_v: float
    estimator_timeout: float
    geofence: Geofence


@dataclass
class GamepadConfig:
    deadman_axis: int
    deadman_threshold: float
    land_button: int


@dataclass
class Config:
    radio: RadioConfig
    mocap: MocapConfig
    flight: FlightConfig
    host_control: HostControlConfig
    safety: SafetyConfig
    gamepad: GamepadConfig
    log_dir: str
    source: Path = field(default=DEFAULT_CONFIG, compare=False)

    @property
    def onboard_control(self) -> bool:
        """The drone's own controller flies it on its onboard state (mocap passthrough or Kalman);
        otherwise the host controller does (state_source host)."""
        return self.flight.state_source in ("mocap", "kalman")

    def to_dict(self) -> dict:
        """Plain YAML-safe dict of every setting (what a log's sidecar records)."""
        def plain(v):
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            return list(v) if isinstance(v, tuple) else v
        d = dataclasses.asdict(self)
        d.pop("source")
        return plain(d)

    def resolve(self, path: str) -> Path:
        """Paths in the config are relative to the package directory."""
        p = Path(path)
        return p if p.is_absolute() else PACKAGE_DIR / p


CONTROLLERS = {"pid": 1, "mellinger": 2, "indi": 3, "brescianini": 4, "lee": 5}   # stabilizer.controller
STATE_SOURCES = ("mocap", "kalman", "host")


def _build(cls, data: dict, where: str):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init and f.name != "source"}
    unknown, missing = set(data) - names, names - set(data)
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    kwargs = {}
    for name in names:
        value = data[name]
        if dataclasses.is_dataclass(hints[name]):
            value = _build(hints[name], value, f"{where}.{name}")
        kwargs[name] = value
    return cls(**kwargs)


def _apply_override(data: dict, assignment: str) -> None:
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got {assignment!r}")
    *parents, leaf = key.strip().split(".")
    node = data
    for p in parents:
        node = node[p]
    if leaf not in node:
        raise ValueError(f"unknown config key {key!r}")
    node[leaf] = yaml.safe_load(raw)


def load_config(path=DEFAULT_CONFIG, overrides=()) -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    for o in overrides:
        _apply_override(data, o)
    cfg = _build(Config, data, path.name)
    cfg.source = path
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.flight.state_source not in STATE_SOURCES:
        raise ValueError(f"flight.state_source must be one of {STATE_SOURCES}")
    if not 0 < cfg.host_control.max_accel_z < 9.0:
        raise ValueError("host_control.max_accel_z must be in (0, 9) m/s^2 (keeps thrust positive)")
    if cfg.flight.controller not in CONTROLLERS:
        raise ValueError(f"flight.controller must be one of {sorted(CONTROLLERS)}")
    if not -180.0 <= cfg.mocap.body_yaw_offset_deg <= 180.0:
        raise ValueError("mocap.body_yaw_offset_deg must be in [-180, 180]")
    if len(cfg.mocap.body_tilt_offset_deg) != 2:
        raise ValueError("mocap.body_tilt_offset_deg must be [roll, pitch]")
    if cfg.mocap.up_axis not in ("z", "y"):
        raise ValueError("mocap.up_axis must be 'z' or 'y'")
    for name in ("hover_height", "takeoff_time", "land_time", "setpoint_hz"):
        if getattr(cfg.flight, name) <= 0:
            raise ValueError(f"flight.{name} must be > 0")
    if cfg.flight.hover_duration < 0:
        raise ValueError("flight.hover_duration must be >= 0")
