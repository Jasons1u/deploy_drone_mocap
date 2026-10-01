"""Filter-test error math, the CSV it writes, and the handheld loop with fakes."""

import csv
import math
import time

import pytest
import yaml

pytest.importorskip("cflib")

from cf_mocap import diagnostics  # noqa: E402
from cf_mocap.config import load_config  # noqa: E402
from cf_mocap.flight_log import COLUMNS, FlightLog  # noqa: E402
from cf_mocap.flight_plan import Setpoint  # noqa: E402
from cf_mocap.link import TelemetrySample  # noqa: E402
from mocap_natnet import Pose  # noqa: E402


def quat_rpy(roll, pitch, yaw):
    """Right-handed ZYX (deg) -> (w, x, y, z)."""
    r, p, y = (math.radians(a) / 2 for a in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def pose(pos=(0.1, 0.2, 0.3), rpy=(0, 0, 0), vel=(0, 0, 0)):
    return Pose(pos, quat_rpy(*rpy), vel, 1, 0.0, time.monotonic())


def tel(pos=(0.1, 0.2, 0.3), rpy=(0, 0, 0), vel=(0, 0, 0)):
    return TelemetrySample(pos, vel, rpy, (1000, 2000, 3000, 4000), 4.0)


def test_filter_error_zero_when_matching_including_tilt():
    """Telemetry pitch is already converted to right-handed, so a real tilt compares at zero."""
    e = diagnostics.filter_error(pose(rpy=(10, 20, 45)), tel(rpy=(10, 20, 45)))
    assert (e.pos_cm, e.vel_cm_s) == (0, 0)
    assert (e.roll_deg, e.pitch_deg, e.yaw_deg) == pytest.approx((0, 0, 0), abs=1e-6)


def test_filter_error_values_and_yaw_wrap():
    e = diagnostics.filter_error(pose(rpy=(0, 0, -179)), tel(pos=(0.13, 0.24, 0.3), rpy=(0, 0, 179),
                                                             vel=(0.1, 0, 0)))
    assert e.pos_cm == pytest.approx(5.0)
    assert e.vel_cm_s == pytest.approx(10.0)
    assert e.yaw_deg == pytest.approx(-2.0)


def test_filter_error_missing_data():
    assert diagnostics.filter_error(None, tel()) is None
    assert diagnostics.filter_error(pose(), TelemetrySample()) is None


def test_summarize():
    errs = [diagnostics.FilterError(3, 0, 0, 0, 4), diagnostics.FilterError(4, 0, 0, 0, -3)]
    s = diagnostics.summarize(errs)
    assert s["pos_cm"] == pytest.approx((math.sqrt(12.5), 4))
    assert s["yaw_deg"] == pytest.approx((math.sqrt(12.5), 4))


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_flight_log_columns(tmp_path):
    with FlightLog(tmp_path) as flog:
        flog.write(10.0, "hover", Setpoint(1, 2, 3, 45), pose(rpy=(5, 10, 30)), 0.004,
                   tel(rpy=(5, 10, 30)))
        flog.write(10.02, "handheld", None, None, float("inf"), TelemetrySample())
    rows = read(flog.path)
    assert list(rows[0]) == list(COLUMNS)
    assert float(rows[0]["mocap_pitch_deg"]) == pytest.approx(10.0)
    assert float(rows[0]["est_pitch_deg"]) == pytest.approx(10.0)
    assert rows[0]["m4"] == "4000" and float(rows[0]["mocap_age_ms"]) == pytest.approx(4.0)
    assert math.isnan(float(rows[1]["sp_z"])) and math.isnan(float(rows[1]["mocap_x"]))


class FakeMocap:
    def __init__(self, offset_cm=0.0):
        self._offset = offset_cm / 100

    def pose(self, _body):
        return pose(pos=(0.1 + self._offset, 0.2, 0.3))

    def age(self, _body):
        return 0.005


class FakeTelemetry:
    latest = tel()


@pytest.mark.parametrize("offset_cm,code", [(0.0, 0), (10.0, 2)])
def test_filter_test_loop(tmp_path, offset_cm, code):
    cfg = load_config()
    with FlightLog(tmp_path, prefix="filter") as flog:
        result = diagnostics.filter_test(cfg, FakeMocap(offset_cm), FakeTelemetry(), flog,
                                         duration=0.2)
    assert result == code
    rows = read(flog.path)
    assert len(rows) >= 5 and {r["phase"] for r in rows} == {"handheld"}


class BiasedTelemetry:
    """What the 2026-09-28 log showed: drone still, Kalman velocity stuck near 11 cm/s."""
    latest = tel(vel=(0.085, -0.078, 0.0))


def test_filter_test_flags_velocity_bias_while_still(tmp_path, capsys):
    cfg = load_config()
    with FlightLog(tmp_path, prefix="filter") as flog:
        result = diagnostics.filter_test(cfg, FakeMocap(), BiasedTelemetry(), flog, duration=0.2)
    out = capsys.readouterr().out
    assert result == 2
    assert "while held still" in out and "still_velocity" in out


def test_config_sidecar_records_overrides(tmp_path):
    cfg = load_config(overrides=["mocap.ext_quat_std_dev=0.05"])
    with FlightLog(tmp_path, "filter", cfg) as flog:
        pass
    saved = yaml.safe_load(flog.path.with_suffix(".yaml").read_text())
    assert saved["mocap"]["ext_quat_std_dev"] == 0.05
    assert saved["safety"]["geofence"]["x"] == [-1.5, 1.5]
