import math

import pytest

pytest.importorskip("cflib")

from cf_mocap import diagnostics  # noqa: E402
from cf_mocap.config import DEFAULT_CONFIG, load_config  # noqa: E402
from mocap_natnet.frames import quat_conj, quat_from_rpy, quat_mul, rotate, yaw_from_quat  # noqa: E402


def q_deg(r, p, y):
    return quat_from_rpy(*(math.radians(a) for a in (r, p, y)))


# The drone's IMU really sits at this attitude; the rigid body is created slightly crooked.
Q_IMU = q_deg(-2.1, 1.1, 78.0)
MISALIGN = q_deg(1.6, -0.5, 0.0)
Q_MOCAP_RAW = quat_mul(Q_IMU, MISALIGN)
ACC_AT_REST = rotate(quat_conj(Q_IMU), (0.0, 0.0, 1.0))     # gravity reaction, IMU frame [G]


# The offset is stored as [roll, pitch]; dropping the second-order yaw part costs ~1e-6 rad.
TOL = 1e-5


def corrected_up(offset_deg):
    q = quat_mul(Q_MOCAP_RAW, q_deg(*offset_deg, 0.0))
    return rotate(quat_conj(q), (0.0, 0.0, 1.0))


def test_offset_makes_mocap_up_match_accelerometer():
    offset = diagnostics.tilt_offset(ACC_AT_REST, Q_MOCAP_RAW, [0.0, 0.0])
    assert corrected_up(offset) == pytest.approx(ACC_AT_REST, abs=TOL)
    assert max(abs(a) for a in offset) < 2.0


def test_recalibration_from_a_partial_offset_converges_to_the_same_answer():
    full = diagnostics.tilt_offset(ACC_AT_REST, Q_MOCAP_RAW, [0.0, 0.0])
    partial = [0.5, -0.2]
    q_partial = quat_mul(Q_MOCAP_RAW, q_deg(*partial, 0.0))
    again = diagnostics.tilt_offset(ACC_AT_REST, q_partial, partial)
    assert corrected_up(again) == pytest.approx(ACC_AT_REST, abs=TOL)
    assert again == pytest.approx(full, abs=0.01)


def test_write_offset_keeps_comments(tmp_path):
    cfg_file = tmp_path / "hover.yaml"
    cfg_file.write_text(DEFAULT_CONFIG.read_text())
    assert diagnostics._write_offset(cfg_file, 1.234, -0.5)
    text = cfg_file.read_text()
    assert "body_tilt_offset_deg: [1.234, -0.500]   # [roll, pitch]" in text
    assert load_config(cfg_file).mocap.body_tilt_offset_deg == [1.234, -0.5]


def test_zero_offset_is_no_correction_and_offset_is_applied():
    zero = load_config(overrides=["mocap.body_tilt_offset_deg=[0, 0]", "mocap.body_yaw_offset_deg=0"])
    assert zero.mocap.body_rotation() == pytest.approx((1.0, 0.0, 0.0, 0.0))
    tilted = load_config(overrides=["mocap.body_tilt_offset_deg=[-1.9, 0.7]", "mocap.body_yaw_offset_deg=0"])
    assert tilted.mocap.body_rotation() == pytest.approx(q_deg(-1.9, 0.7, 0.0))


def body_z_in_world(q):
    return rotate(q, (0.0, 0.0, 1.0))


def test_yaw_calibration_zeroes_yaw_and_keeps_the_tilt_correction():
    """Rigid body 74.5 deg off the nose (measured 2026-09-29), drone nose along +X, tilt already
    calibrated in the old frame."""
    old_tilt = [-4.172, -1.107]
    drone = q_deg(0.4, -0.3, 0.0)                              # really: nose along +X
    raw = quat_mul(drone, q_deg(0.0, 0.0, -74.5))              # what Motive streams
    before = quat_mul(raw, q_deg(*old_tilt, 0.0))              # old correction (yaw 0)
    new_yaw, new_tilt = diagnostics.yaw_calibration(before, 0.0, old_tilt)
    after = quat_mul(raw, load_config(overrides=[
        f"mocap.body_yaw_offset_deg={new_yaw}",
        f"mocap.body_tilt_offset_deg=[{new_tilt[0]}, {new_tilt[1]}]"]).mocap.body_rotation())
    assert math.degrees(yaw_from_quat(after)) == pytest.approx(0.0, abs=0.05)
    # only turned about the (tilt-corrected) body z axis: that axis still points the same way
    # (to ~1e-4 rad: storing the offsets as roll/pitch/yaw drops a second-order yaw term)
    assert body_z_in_world(after) == pytest.approx(body_z_in_world(before), abs=1e-3)
    assert new_yaw == pytest.approx(74.5, abs=0.5)


def test_config_value_writer_keeps_comments(tmp_path):
    cfg_file = tmp_path / "hover.yaml"
    cfg_file.write_text(DEFAULT_CONFIG.read_text())
    assert diagnostics._write_config_value(cfg_file, "body_yaw_offset_deg", "74.50")
    assert "body_yaw_offset_deg: 74.50          # rigid body x" in cfg_file.read_text()
    assert load_config(cfg_file).mocap.body_yaw_offset_deg == 74.5
