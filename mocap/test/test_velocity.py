"""Mocap velocity must survive frames arriving in network bursts."""

import pytest

from mocap_natnet.client import MIN_RATE_WINDOW_S, _VelocityFilter

RATE = 120.0


def bursty_stream(speed, seconds=2.0, burst=3):
    """(frame, position, arrival time): 120 Hz frames delivered `burst` at a time."""
    for f in range(int(seconds * RATE)):
        sent = f / RATE
        arrival = (f // burst) * burst / RATE + (f % burst) * 1e-5   # bunched, microseconds apart
        yield f, (speed * sent, 0.0, 1.0), arrival + 0.002


@pytest.mark.parametrize("speed", [0.0, 0.5])
def test_velocity_is_right_despite_bursts(speed):
    filt = _VelocityFilter(alpha=0.3)
    worst = 0.0
    for frame, pos, t in bursty_stream(speed):
        vx = filt.update(frame, pos, t)[0]
        if t > MIN_RATE_WINDOW_S + 0.2:
            worst = max(worst, abs(vx - speed))
    assert worst < 0.01          # < 1 cm/s; arrival-time differencing gave metres per second here


def test_skipped_frames_use_the_real_gap():
    """Frames dropped while untracked: dt comes from the frame numbers, not a fixed period."""
    filt = _VelocityFilter(alpha=1.0)
    for f in range(0, 120):
        filt.update(f, (f / RATE, 0, 0), f / RATE)
    assert filt.update(125, (125 / RATE, 0, 0), 125 / RATE)[0] == pytest.approx(1.0, rel=1e-3)


def test_duplicate_and_restart():
    filt = _VelocityFilter(alpha=1.0)
    for f in range(120):
        filt.update(f, (f / RATE, 0, 0), f / RATE)
    v = filt.update(119, (5.0, 0, 0), 1.0)           # duplicate frame number: ignored
    assert v[0] == pytest.approx(1.0, rel=1e-3)
    assert filt.update(3, (0, 0, 0), 2.0) == (0.0, 0.0, 0.0)   # counter went back: restart
