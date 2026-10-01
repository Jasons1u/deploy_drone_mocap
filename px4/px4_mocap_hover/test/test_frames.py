import numpy as np
import pytest

from px4_mocap_hover import frames


def rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_vector_maps_are_involutions():
    v = np.array([1.0, 2.0, 3.0])
    assert frames.ned_to_enu(v) == pytest.approx([2, 1, -3])
    assert frames.enu_to_ned(frames.ned_to_enu(v)) == pytest.approx(v)
    assert frames.flu_to_frd(frames.frd_to_flu(v)) == pytest.approx(v)


def test_attitude_conversion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rand_quat(rng)
        back = frames.flu_enu_to_frd_ned(frames.frd_ned_to_flu_enu(q))
        assert np.allclose(back, q) or np.allclose(back, -q)


def test_nose_north_level_is_enu_yaw_90():
    """PX4 identity attitude = level, nose North. North is ENU +Y, i.e. ENU yaw +90 deg."""
    q = frames.frd_ned_to_flu_enu([1.0, 0.0, 0.0, 0.0])
    assert frames.yaw_from_quat(q) == pytest.approx(np.pi / 2)
    assert frames.quat_apply(q, [0, 0, 1]) == pytest.approx([0, 0, 1])   # body up is world up


def test_attitude_conversion_is_consistent_with_vector_maps():
    """R_enu(q') @ v_flu == ned_to_enu(R_ned(q) @ frd_to_flu(v_flu))."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = rand_quat(rng)
        v = rng.normal(size=3)
        lhs = frames.quat_apply(frames.frd_ned_to_flu_enu(q), v)
        rhs = frames.ned_to_enu(frames.quat_apply(q, frames.flu_to_frd(v)))
        assert lhs == pytest.approx(rhs)


def test_quat_apply_inverse():
    rng = np.random.default_rng(3)
    q, v = rand_quat(rng), rng.normal(size=3)
    assert frames.quat_apply_inverse(q, frames.quat_apply(q, v)) == pytest.approx(v)


def test_euler_matches_yaw_quat():
    for yaw in (-3.0, -1.0, 0.0, 0.5, 3.0):
        assert frames.euler_xyz_from_quat(frames.quat_from_yaw(yaw)) == pytest.approx([0, 0, yaw])


def test_wrap_pi():
    assert frames.wrap_pi(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    assert frames.wrap_pi(-0.2) == pytest.approx(-0.2)
