"""Constants for Crazyflie sysID flights; the deploy-chain block must match crazyflieDeploy.py and the sim VelocityActionCfg."""

from dataclasses import dataclass


@dataclass
class Config:
    uri: str = "radio://0/80/2M/E7E7E7E7E7"

    # deploy chain: mu = KPV * (uv - v) + MU_Z_HOVER, slew-limited and clamped
    kpv: tuple = (4.0, 4.0, 15.0)
    mu_z_hover: float = 3.0
    mu_max: tuple = (7.0, 7.0, 8.0)
    mu_slew: tuple = (40.0, 40.0, 40.0)
    vel_cutoff_hz: float = 6.0

    # position hold used outside the test segments (sysID r3 gains)
    kpr_xy: float = 2.5
    kpr_z: float = 2.5
    hold_uv_max_xy: float = 1.0
    hold_uv_max_z: float = 0.6

    # flight profile
    land_z: float = 0.08
    pose_warmup: float = 2.0
    takeoff_time: float = 4.0
    settle_time: float = 2.0
    land_time: float = 4.0

    # safety
    batt_min_arm: float = 3.70
    batt_min_fly: float = 3.30
    batt_low_hold: float = 0.4
    pose_stale_timeout: float = 0.20
    pose_frozen_timeout: float = 0.30
    pose_frozen_eps: float = 1e-5
    max_pose_jump: float = 0.30
    r_max: float = 1.3
    z_min: float = 0.05
    z_max: float = 1.6
    dt_min: float = 1.0 / 500.0
    dt_max: float = 0.10

    # open-loop mu steps: wait until settled before each, end early if the drone runs away
    settled_speed: float = 0.12
    settled_dist: float = 0.15
    settle_wait_max: float = 6.0
    open_vz_abort: float = 1.2
    open_z_margin: float = 0.25
