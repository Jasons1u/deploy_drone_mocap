import pytest

from cf_mocap.flight_plan import HoverPlan, Phase, smoothstep

ORIGIN = (0.3, -0.2, 0.03)


def plan(duration=5.0):
    p = HoverPlan(ORIGIN, yaw_deg=40.0, hover_height=0.5, takeoff_time=2.0, land_time=2.0,
                  hover_duration=duration)
    p.start(100.0)
    return p


def test_smoothstep_ends():
    assert smoothstep(-1) == 0 and smoothstep(0) == 0
    assert smoothstep(0.5) == pytest.approx(0.5)
    assert smoothstep(1) == 1 and smoothstep(2) == 1


def test_xy_and_yaw_held_at_takeoff_pose():
    p = plan()
    for t in (100.0, 101.0, 103.0, 107.5):
        _, sp = p.step(t)
        assert (sp.x, sp.y, sp.yaw_deg) == (0.3, -0.2, 40.0)


def test_full_sequence():
    p = plan()
    phase, sp = p.step(100.0)
    assert phase is Phase.TAKEOFF and sp.z == pytest.approx(0.03)
    assert p.step(101.0)[1].z == pytest.approx(0.28)            # half way up
    phase, sp = p.step(103.0)
    assert phase is Phase.HOVER and sp.z == pytest.approx(0.53)
    phase, sp = p.step(107.0)                                   # 2 s takeoff + 5 s hover
    assert phase is Phase.LAND and sp.z == pytest.approx(0.53)
    assert p.step(108.0)[1].z == pytest.approx(0.28)
    phase, sp = p.step(109.0)
    assert phase is Phase.DONE and sp.z == pytest.approx(0.03)


def test_late_tick_still_lands_on_schedule():
    """A control tick arriving after the auto-land time lands from the scheduled instant."""
    p = plan()
    phase, sp = p.step(108.0)
    assert phase is Phase.LAND and sp.z == pytest.approx(0.28)


def test_land_mid_takeoff_starts_from_current_height():
    p = plan()
    z_now = p.step(101.0)[1].z
    p.land(101.0)
    phase, sp = p.step(101.0)
    assert phase is Phase.LAND and sp.z == pytest.approx(z_now)
    assert p.step(103.0)[0] is Phase.DONE


def test_hover_forever_until_land():
    p = plan(duration=0.0)
    assert p.step(10_000.0)[0] is Phase.HOVER
    p.land(10_000.0)
    assert p.step(10_001.0)[0] is Phase.LAND


def test_step_before_start():
    p = HoverPlan(ORIGIN, 0, 0.5, 2, 2, 0)
    with pytest.raises(RuntimeError):
        p.step(0.0)
