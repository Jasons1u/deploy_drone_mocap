"""The motorPowerSet override: set each motor's PWM directly. Shared by the motor test scripts.

!! The override acts below every firmware safety net (disarm, commander timeout, emergency stop):
!! if the radio link drops, the motors keep the last values. Every path out must call stop().

cflib's param.set_value() only queues the write; closing the link right after can drop it (that is
how a spin test once left the motors running). stop() therefore waits for the drone to confirm each
write, and raises MotorStopError if it does not.
"""

from __future__ import annotations

import time

MOTORS = ("m1", "m2", "m3", "m4")
PWM_MAX = 65535
SEND_PERIOD_S = 0.05    # at most 20 param writes per motor per second
CONFIRM_TIMEOUT_S = 2.0


class MotorStopError(RuntimeError):
    """The drone did not confirm the motors were stopped: they may still be running."""


class MotorBench:
    """Owns the motorPowerSet override. Every path out goes through stop()."""

    def __init__(self, cf, cap: int):
        self._cf = cf
        self.cap = cap
        self.enabled = False
        self.requested = [0, 0, 0, 0]
        self._sent = [None, None, None, None]
        self._last_send = 0.0

    def set_requested(self, i: int, pwm: float) -> None:
        self.requested[i] = int(max(0, min(self.cap, pwm)))

    def enable(self) -> None:
        for i, m in enumerate(MOTORS):              # start from whatever the sliders say
            self._write(f"motorPowerSet.{m}", self.requested[i])
            self._sent[i] = self.requested[i]
        self._cf.param.set_value("motorPowerSet.enable", 1)
        self.enabled = True

    def push(self, now: float) -> None:
        """Send slider changes, rate-limited (a dragged slider fires far faster than needed)."""
        if not self.enabled or now - self._last_send < SEND_PERIOD_S:
            return
        for i, m in enumerate(MOTORS):
            if self.requested[i] != self._sent[i]:
                self._write(f"motorPowerSet.{m}", self.requested[i])
                self._sent[i] = self.requested[i]
        self._last_send = now

    def stop(self) -> None:
        """Motors to zero, then hand control back to the firmware, each write confirmed by the
        drone. Safe to call repeatedly. Forgets the requested values, so the next enable() starts
        from zero. Raises MotorStopError if the drone does not confirm."""
        self.requested = [0, 0, 0, 0]
        self._sent = [0, 0, 0, 0]
        self.enabled = False
        for m in MOTORS:
            self._write(f"motorPowerSet.{m}", 0, confirm=True)
        self._write("motorPowerSet.enable", 0, confirm=True)

    def _write(self, name: str, value: int, confirm: bool = False) -> None:
        self._cf.param.set_value(name, int(value))
        if not confirm:
            return
        deadline = time.monotonic() + CONFIRM_TIMEOUT_S
        while time.monotonic() < deadline:     # the cache only changes when the drone replies
            if int(float(self._cf.param.get_value(name))) == int(value):
                return
            time.sleep(0.01)
        raise MotorStopError(f"the drone did not confirm {name} = {value}: MOTORS MAY STILL BE "
                             f"RUNNING — switch the drone off")


def stop_or_warn(bench: MotorBench, attempts: int = 3) -> bool:
    """stop(), retried; on failure print an unmissable warning instead of a traceback."""
    for _ in range(attempts):
        try:
            bench.stop()
            return True
        except MotorStopError:
            continue
    print("\n" + "!" * 78 + "\n!! The drone did NOT confirm the motors stopped. SWITCH THE DRONE OFF NOW.\n"
          + "!" * 78, flush=True)
    return False
