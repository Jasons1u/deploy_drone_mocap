"""cf_mocap.motor_bench: every way out must zero the motors and drop the override."""

import pytest

from cf_mocap import motor_bench
from cf_mocap.motor_bench import MotorBench, MotorStopError


class FakeParam:
    """Like cflib: the value read back only changes once the drone 'replies' (applies)."""

    def __init__(self, applies=True):
        self.writes = []
        self.values = {}
        self.applies = applies

    def set_value(self, name, value):
        self.writes.append((name, value))
        if self.applies:
            self.values[name] = str(value)

    def get_value(self, name):
        return self.values.get(name, "1")


class FakeCF:
    def __init__(self):
        self.param = FakeParam()


def bench(cap=20000):
    cf = FakeCF()
    return MotorBench(cf, cap), cf.param.writes


def test_sliders_are_capped_and_clamped():
    b, _ = bench(cap=20000)
    b.set_requested(0, 50000)
    b.set_requested(1, -5)
    assert b.requested[:2] == [20000, 0]


def test_nothing_is_sent_until_enabled():
    b, writes = bench()
    b.set_requested(0, 10000)
    b.push(now=100.0)
    assert writes == []


def test_enable_sends_values_before_turning_the_override_on():
    b, writes = bench()
    b.set_requested(2, 12000)
    b.enable()
    assert writes[-1] == ("motorPowerSet.enable", 1)
    assert ("motorPowerSet.m3", 12000) in writes[:-1]


def test_push_sends_only_changes_and_is_rate_limited():
    b, writes = bench()
    b.enable()
    writes.clear()
    b.set_requested(0, 5000)
    b.push(now=100.0)
    b.set_requested(0, 6000)
    b.push(now=100.01)                       # too soon: held back
    assert writes == [("motorPowerSet.m1", 5000)]
    b.push(now=100.2)
    assert writes[-1] == ("motorPowerSet.m1", 6000)


def test_stop_zeroes_motors_then_disables_override():
    b, writes = bench()
    for i in range(4):
        b.set_requested(i, 15000)
    b.enable()
    writes.clear()
    b.stop()
    assert writes == [(f"motorPowerSet.m{i}", 0) for i in range(1, 5)] + [("motorPowerSet.enable", 0)]
    assert not b.enabled


def test_enable_after_stop_starts_from_zero():
    """The 2026-09-29 spin test bug: stop() kept M1/M3 requested, so enable() respun them."""
    b, writes = bench()
    b.set_requested(0, 30000)
    b.set_requested(2, 30000)
    b.enable()
    b.stop()
    writes.clear()
    b.enable()
    assert writes == [(f"motorPowerSet.m{i}", 0) for i in range(1, 5)] + [("motorPowerSet.enable", 1)]


def test_stop_raises_if_the_drone_never_confirms(monkeypatch):
    """The link-closing race: writes queued but never applied must not look like a clean stop."""
    monkeypatch.setattr(motor_bench, "CONFIRM_TIMEOUT_S", 0.05)
    cf = FakeCF()
    cf.param.applies = False
    with pytest.raises(MotorStopError, match="MAY STILL BE RUNNING"):
        MotorBench(cf, 20000).stop()


def test_stop_or_warn_warns_loudly(monkeypatch, capsys):
    monkeypatch.setattr(motor_bench, "CONFIRM_TIMEOUT_S", 0.01)
    cf = FakeCF()
    cf.param.applies = False
    assert motor_bench.stop_or_warn(MotorBench(cf, 20000)) is False
    assert "SWITCH THE DRONE OFF" in capsys.readouterr().out
