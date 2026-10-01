import pytest

pytest.importorskip("cflib")

from cf_mocap.diagnostics import _battery_verdict  # noqa: E402


@pytest.mark.parametrize("lo,hi,code,needle", [
    (4.10, 4.15, 0, "steady"),
    (0.01, 4.18, 2, "JUMPED"),      # flapping reading: USB charging or loose connector
    (0.01, 0.05, 2, "USB power"),
    (3.60, 3.65, 2, "charge it"),
])
def test_battery_verdict(capsys, lo, hi, code, needle):
    assert _battery_verdict(lo, hi, min_v=3.8) == code
    assert needle in capsys.readouterr().out
