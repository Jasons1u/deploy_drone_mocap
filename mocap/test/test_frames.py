import math

import pytest

from mocap_natnet.frames import (from_z_up, quat_from_rpy, quat_mul, rotate, rpy_from_quat,
                                 shortest_arc, to_z_up, yaw_from_quat)


def yaw_quat(deg):
    a = math.radians(deg) / 2
    return (math.cos(a), 0.0, 0.0, math.sin(a))


def test_y_up_height_becomes_z():
    pos, _ = to_z_up((0.1, 1.2, -0.3), (1, 0, 0, 0), "y")
    assert pos == pytest.approx((0.1, 0.3, 1.2))


def test_y_up_identity_stays_identity():
    """A body created aligned with the Motive axes is level with yaw 0 in either convention."""
    _, q = to_z_up((0, 0, 0), (1, 0, 0, 0), "y")
    assert q == pytest.approx((1, 0, 0, 0))


def test_y_up_yaw_maps_to_z_yaw():
    """Yawing about Motive's +Y (up) by +30 deg is yawing about our +Z by +30 deg."""
    a = math.radians(30) / 2
    q_yup = (math.cos(a), 0.0, math.sin(a), 0.0)
    _, q = to_z_up((0, 0, 0), q_yup, "y")
    assert yaw_from_quat(q) == pytest.approx(math.radians(30))
    # body forward (x) still rotates consistently with the position mapping
    fwd_yup = rotate(q_yup, (1, 0, 0))
    fwd_zup, _ = to_z_up(fwd_yup, (1, 0, 0, 0), "y")
    assert rotate(q, (1, 0, 0)) == pytest.approx(fwd_zup)


@pytest.mark.parametrize("up", ["y", "z"])
def test_round_trip(up):
    pos, q = (0.3, -1.1, 0.7), quat_mul(yaw_quat(40), (math.cos(0.1), math.sin(0.1), 0, 0))
    p2, q2 = to_z_up(*from_z_up(pos, q, up), up)
    assert p2 == pytest.approx(pos)
    assert q2 == pytest.approx(q)


def test_bad_axis():
    with pytest.raises(ValueError):
        to_z_up((0, 0, 0), (1, 0, 0, 0), "x")


def test_rpy_round_trip():
    rpy = (0.3, -0.2, 2.5)
    assert rpy_from_quat(quat_from_rpy(*rpy)) == pytest.approx(rpy)


@pytest.mark.parametrize("u,v", [((0, 0, 1), (0.03, -0.02, 1)), ((1, 0, 0), (0, 1, 0)),
                                 ((0, 0, 1), (0, 0, 1))])
def test_shortest_arc(u, v):
    q = shortest_arc(u, v)
    n = math.sqrt(sum(a * a for a in v))
    assert rotate(q, u) == pytest.approx([a / n for a in v], abs=1e-9)
    if u == (0, 0, 1):
        assert q[3] == pytest.approx(0.0, abs=1e-12)     # tilting z needs no rotation about z
