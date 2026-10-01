"""Drive hover._fly with fakes: every exit path must end with the motors stopped."""

import csv
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("cflib")
pygame = pytest.importorskip("pygame")

from cf_mocap import hover, safety  # noqa: E402
from cf_mocap.config import load_config  # noqa: E402
from cf_mocap.flight_log import COLUMNS, FlightLog  # noqa: E402
from cf_mocap.link import TelemetrySample  # noqa: E402
from mocap_natnet import Pose  # noqa: E402

BODY = 1


class FakeCF:
    def __init__(self):
        self.calls = []
        self.commander = SimpleNamespace(
            send_setpoint=lambda *a: self.calls.append(("setpoint", a)),
            send_position_setpoint=lambda *a: self.calls.append(("position", a)),
            send_stop_setpoint=lambda: self.calls.append(("stop",)),
            send_notify_setpoint_stop=lambda *a: self.calls.append(("notify_stop",)),
        )
        self.supervisor = SimpleNamespace(
            send_arming_request=lambda armed: self.calls.append(("arm", armed)))

    def names(self):
        return [c[0] for c in self.calls]

    def positions(self):
        return [c[1] for c in self.calls if c[0] == "position"]


class FakeMocap:
    def __init__(self, pos=(0.1, 0.2, 0.03)):
        self.pos = pos
        self.lost_after = None
        self._t0 = time.monotonic()

    def pose(self, _body):
        return Pose(self.pos, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0, 0.0, time.monotonic())

    def age(self, _body):
        if self.lost_after is not None and time.monotonic() - self._t0 > self.lost_after:
            return 10.0
        return 0.005


class FakeTelemetry:
    """Fresh 50 Hz telemetry; the supervisor arms when FakeCF received an arming request."""

    def __init__(self, mocap, cf, arms=True, stale_after=None, onboard_mocap_stale_after=None):
        self._mocap, self._cf, self._arms = mocap, cf, arms
        self._stale_after = stale_after
        self._onboard_stale_after = onboard_mocap_stale_after
        self._t0 = time.monotonic()

    @property
    def latest(self):
        armed = self._arms and ("arm", True) in self._cf.calls
        info = safety.SUP_CAN_BE_ARMED | (safety.SUP_ARMED if armed else 0)
        t = time.monotonic() - self._t0
        stale = self._stale_after is not None and t > self._stale_after
        onboard_stale = self._onboard_stale_after is not None and t > self._onboard_stale_after
        return TelemetrySample(position=self._mocap.pos, velocity=(0.0, 0.0, 0.0),
                               rpy_deg=(0.0, 0.0, 0.0), motors=(0, 0, 0, 0), battery_v=4.1,
                               supervisor_info=info, age_s=5.0 if stale else 0.01,
                               onboard_mocap_age_s=1.0 if onboard_stale else 0.01)


class FakeGamepad:
    def __init__(self, release_after=None):
        self._release_after = release_after
        self._t0 = None
        self.land_at = None

    @property
    def held(self):
        if self._t0 is None:
            self._t0 = time.monotonic()
        return self._release_after is None or time.monotonic() - self._t0 < self._release_after

    def take_land_request(self):
        if self.land_at is not None and time.monotonic() - self._t0 > self.land_at:
            self.land_at = None
            return True
        return False


class NullLog:
    def write(self, *a):
        pass


def test_nominal_flight_writes_csv(tmp_path):
    """The real FlightLog, fed by the real flight loop."""
    mocap = FakeMocap()
    with FlightLog(tmp_path) as flog:
        cf = FakeCF()
        outcome = hover._fly(cf, config(), mocap, FakeTelemetry(mocap, cf), FakeGamepad(), flog)
    assert outcome == "landed"
    with open(flog.path) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(COLUMNS)
    assert {r["phase"] for r in rows} == {"takeoff", "hover", "land"}


def config(**flight):
    overrides = ["flight.takeoff_time=0.2", "flight.land_time=0.2", "flight.hover_duration=0.2",
                 "flight.setpoint_hz=200"]
    overrides += [f"flight.{k}={v}" for k, v in flight.items()]
    return load_config(overrides=overrides)


def fly(cfg, mocap, gamepad, **telemetry):
    cf = FakeCF()
    outcome = hover._fly(cf, cfg, mocap, FakeTelemetry(mocap, cf, **telemetry), gamepad, NullLog())
    return outcome, cf


@pytest.mark.parametrize("source", ["mocap", "kalman"])
def test_nominal_flight_onboard_control_sends_position_setpoints(source):
    outcome, cf = fly(config(state_source=source), FakeMocap(), FakeGamepad())
    assert outcome == "landed"
    names = cf.names()
    assert names.index("arm") < names.index("position") and "setpoint" in names   # unlock only
    assert names[-3:] == ["stop", "notify_stop", "arm"] and cf.calls[-1] == ("arm", False)
    zs = [p[2] for p in cf.positions()]
    assert max(zs) == pytest.approx(0.43, abs=0.01)             # 0.03 takeoff z + 0.4 hover
    assert {p[:2] for p in cf.positions()} == {(0.1, 0.2)}      # XY held at the takeoff point


def test_default_is_mocap_passthrough_with_indi():
    cfg = config()
    assert (cfg.flight.state_source, cfg.flight.controller) == ("mocap", "indi")
    assert cfg.onboard_control


def test_nominal_flight_host_control_sends_attitude_setpoints():
    cfg = config(state_source="host")
    outcome, cf = fly(cfg, FakeMocap(), FakeGamepad())
    assert outcome == "landed"
    assert "position" not in cf.names()
    setpoints = [c[1] for c in cf.calls if c[0] == "setpoint"]
    assert setpoints[0] == (0, 0, 0, 0)                          # commander unlock
    thrusts = [sp[3] for sp in setpoints[1:]]
    hc = cfg.host_control
    assert thrusts and all(hc.thrust_min <= th <= hc.thrust_max for th in thrusts)
    assert max(thrusts) > hc.hover_thrust                        # climbing: target above the drone
    assert cf.calls[-1] == ("arm", False)


def test_drone_that_does_not_arm_gets_no_thrust():
    """What happened on 2026-09-29: failed self-test, arming silently ignored."""
    cf, mocap = FakeCF(), FakeMocap()
    with pytest.raises(hover.Abort, match="did not arm"):
        hover._fly(cf, config(), mocap, FakeTelemetry(mocap, cf, arms=False), FakeGamepad(), NullLog())
    thrusts = [c[1][3] for c in cf.calls if c[0] == "setpoint"]
    assert thrusts == [0]                       # only the zero-thrust unlock
    assert cf.calls[-1] == ("arm", False)


def test_drone_losing_mocap_is_estop():
    """Mocap firmware: host still sees mocap, but extpose stopped reaching the drone."""
    outcome, cf = fly(config(hover_duration=0), FakeMocap(), FakeGamepad(), onboard_mocap_stale_after=0.15)
    assert "has not received mocap" in outcome and cf.calls[-1] == ("arm", False)


def test_lost_telemetry_is_estop():
    outcome, cf = fly(config(hover_duration=0), FakeMocap(), FakeGamepad(), stale_after=0.15)
    assert "radio link" in outcome and cf.calls[-1] == ("arm", False)


def test_runaway_tilt_is_estop():
    class TippingMocap(FakeMocap):
        def __init__(self):
            super().__init__()
            self._t0 = time.monotonic()

        def pose(self, body):
            p = super().pose(body)
            if time.monotonic() - self._t0 > 0.15:               # tips over mid-flight
                return Pose(p.position, (0.9, 0.44, 0.0, 0.0), p.velocity, 0, 0.0, p.stamp)
            return p
    outcome, cf = fly(config(hover_duration=0), TippingMocap(), FakeGamepad())
    assert "lost attitude control" in outcome and cf.calls[-1] == ("arm", False)


def test_deadman_release_is_estop():
    outcome, cf = fly(config(hover_duration=0), FakeMocap(), FakeGamepad(release_after=0.15))
    assert outcome.startswith("E-STOP: deadman")
    assert cf.calls[-1] == ("arm", False) and "stop" in cf.names()


def test_mocap_loss_is_estop():
    mocap = FakeMocap()
    mocap.lost_after = 0.15
    outcome, cf = fly(config(hover_duration=0), mocap, FakeGamepad())
    assert "mocap lost" in outcome and cf.calls[-1] == ("arm", False)


def test_land_button():
    gamepad = FakeGamepad()
    gamepad.land_at = 0.25
    outcome, _ = fly(config(hover_duration=0), FakeMocap(), gamepad)
    assert outcome == "landed"


def test_preflight_failure_never_arms():
    cf, mocap = FakeCF(), FakeMocap(pos=(9.0, 0.0, 0.03))   # outside the geofence
    with pytest.raises(hover.Abort):
        hover._fly(cf, config(), mocap, FakeTelemetry(mocap, cf), FakeGamepad(), NullLog())
    assert cf.calls == []


class FakeToc:
    def __init__(self, names):
        self._names = set(names)

    def get_element_by_complete_name(self, name):
        return object() if name in self._names else None


@pytest.mark.parametrize("mocap_fw,source,ok", [
    (True, "mocap", True), (False, "mocap", False),
    (False, "kalman", True), (True, "kalman", False),
    (True, "host", True), (False, "host", True),
])
def test_state_source_must_match_firmware(mocap_fw, source, ok):
    cf = SimpleNamespace(param=SimpleNamespace(toc=FakeToc(["mocapEst.velAlpha"] if mocap_fw else [])))
    cfg = config(state_source=source)
    if ok:
        hover._check_firmware(cf, cfg)
    else:
        with pytest.raises(hover.Abort):
            hover._check_firmware(cf, cfg)
