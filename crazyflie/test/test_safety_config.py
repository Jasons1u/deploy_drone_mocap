import math
from types import SimpleNamespace

import pytest

from cf_mocap import safety
from cf_mocap.config import load_config

LEVEL = (1.0, 0.0, 0.0, 0.0)


@pytest.fixture
def cfg():
    return load_config().safety


def tilted(deg):
    a = math.radians(deg) / 2
    return (math.cos(a), math.sin(a), 0.0, 0.0)


def tel(**over):
    """Stand-in for link.TelemetrySample: a healthy, fresh, armable drone sitting level."""
    base = dict(position=(0.0, 0.0, 0.03), rpy_deg=(0.0, 0.0, 0.0), battery_v=4.1,
                supervisor_info=safety.SUP_CAN_BE_ARMED, age_s=0.01, onboard_mocap_age_s=None)
    base.update(over)
    return SimpleNamespace(**base)


def check(cfg, onboard=True, mocap_age=0.01, mocap_pos=(0, 0, 0.03), mocap_quat=LEVEL, **tel_over):
    return safety.preflight(cfg, mocap_age, mocap_pos, mocap_quat, tel(**tel_over), onboard)


@pytest.mark.parametrize("onboard", [True, False])
def test_preflight_ok(cfg, onboard):
    assert check(cfg, onboard) is None


@pytest.mark.parametrize("kwargs,needle", [
    ({"mocap_age": 1.0}, "mocap"),
    ({"mocap_pos": (5, 0, 0)}, "geofence"),
    ({"mocap_quat": tilted(30)}, "tilted"),
    ({"age_s": 2.0}, "no telemetry"),
    ({"position": None}, "state estimate"),
    ({"onboard_mocap_age_s": 1.0}, "has not received mocap"),
    ({"position": (0.2, 0, 0.03)}, "cm from mocap"),
    ({"battery_v": 3.5}, "battery"),
    ({"battery_v": None}, "battery"),
    ({"supervisor_info": 0x0000}, "can be armed"),                  # the failed-self-test drone
    ({"supervisor_info": safety.SUP_LOCKED}, "locked"),
    ({"supervisor_info": None}, "supervisor"),
])
def test_preflight_rejects(cfg, kwargs, needle):
    reason = check(cfg, **kwargs)
    assert reason and needle in reason


def test_preflight_host_mode_checks_imu_against_mocap(cfg):
    assert check(cfg, onboard=False, position=None) is None             # no onboard estimate needed
    reason = check(cfg, onboard=False, rpy_deg=(8.0, 0.0, 0.0))         # IMU says 8 deg, mocap 0
    assert reason and "rigid body axes" in reason


def test_supervisor_problem():
    assert safety.supervisor_problem(safety.SUP_CAN_BE_ARMED | safety.SUP_ARMED, safety.SUP_ARMED) is None
    assert "armed" in safety.supervisor_problem(safety.SUP_CAN_BE_ARMED, safety.SUP_ARMED)
    assert "crashed" in safety.supervisor_problem(safety.SUP_CRASHED, safety.SUP_ARMED)


def test_in_flight(cfg):
    assert safety.in_flight(cfg, 0.01, (0, 0, 0.4), (0.05, 0, 0.4)) is None
    assert "lost" in safety.in_flight(cfg, 0.5, None, (0, 0, 0.4))
    assert "geofence" in safety.in_flight(cfg, 0.01, (0, 0, 3.0), (0, 0, 3.0))
    assert "diverged" in safety.in_flight(cfg, 0.01, (0, 0, 0.4), (0.3, 0, 0.4))
    assert "radio link" in safety.in_flight(cfg, 0.01, (0, 0, 0.4), None, LEVEL, telemetry_age=1.0)
    assert "attitude control" in safety.in_flight(cfg, 0.01, (0, 0, 0.4), None, tilted(40))


def test_tilt():
    assert safety.tilt_deg(LEVEL) == pytest.approx(0)
    assert safety.tilt_deg(tilted(25)) == pytest.approx(25)


def test_config_overrides_and_validation():
    cfg = load_config(overrides=["flight.hover_height=0.7", "mocap.server_ip=10.0.0.5"])
    assert cfg.flight.hover_height == 0.7 and cfg.mocap.server_ip == "10.0.0.5"
    with pytest.raises(ValueError):
        load_config(overrides=["flight.hover_hieght=0.7"])
    with pytest.raises(ValueError):
        load_config(overrides=["flight.controller=lqr"])
    with pytest.raises(ValueError):
        load_config(overrides=["safety.geofence.x=[1, -1]"])
