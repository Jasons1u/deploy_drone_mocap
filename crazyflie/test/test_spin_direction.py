"""The yaw-authority verdict (scripts/spin_direction.py) and the firmware sign it encodes."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("cflib")

_spec = importlib.util.spec_from_file_location(
    "spin_direction", Path(__file__).resolve().parent.parent / "scripts" / "spin_direction.py")
sd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sd          # @dataclass looks its module up here
_spec.loader.exec_module(sd)


@pytest.mark.parametrize("rates,kind", [
    ((2.0, -40.0, 45.0), "ok"),              # M1+M3 up -> CW, M2+M4 up -> CCW
    ((2.0, 45.0, -40.0), "reversed"),        # the other way round
    ((80.0, 82.0, 79.0), "no_authority"),    # differential does nothing; big drift
])
def test_verdict(rates, kind):
    got, text = sd.verdict(sd.Result(*rates))
    assert got == kind
    if rates[0] > sd.MAX_DRIFT_DEG_S:
        assert "do not cancel" in text


def test_phases_never_exceed_the_limits():
    """Default base/diff: total thrust stays below hover and no motor above 40000."""
    base, diff = 24000, 6000
    assert 4 * base <= sd.MAX_TOTAL_PWM and base + diff <= 40000
