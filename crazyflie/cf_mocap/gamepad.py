"""Gamepad deadman: hold the trigger to fly, press a button to land.

`held` is True only while the trigger is past the threshold AND the poll thread is alive and
recent. A stalled thread or an unplugged controller therefore reads as "released" (fail-safe),
unlike a plain flag that would freeze at its last value.

Find your axis/button indices with:  cf-hover --gamepad-test
"""

from __future__ import annotations

import os
import threading
import time

from .config import GamepadConfig

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402  (must follow the env var)

POLL_HZ = 100.0
STALE_S = 0.1


class Gamepad:
    def __init__(self, cfg: GamepadConfig):
        self._cfg = cfg
        self._js = None
        self._trigger = -1.0
        self._last_poll = 0.0
        self._connected = False
        self._land_pressed = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.name = ""

    def start(self) -> "Gamepad":
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("no gamepad detected — plug in the controller (it is the deadman)")
        self._js = pygame.joystick.Joystick(0)
        self._js.init()
        self.name = self._js.get_name()
        if self._cfg.deadman_axis >= self._js.get_numaxes():
            raise RuntimeError(f"{self.name} has no axis {self._cfg.deadman_axis}")
        self._connected = True
        self._thread = threading.Thread(target=self._poll, name="gamepad", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        pygame.quit()

    @property
    def held(self) -> bool:
        fresh = time.monotonic() - self._last_poll < STALE_S
        return self._connected and fresh and self._trigger > self._cfg.deadman_threshold

    def take_land_request(self) -> bool:
        """True once per press of the land button."""
        if self._land_pressed.is_set():
            self._land_pressed.clear()
            return True
        return False

    def _poll(self) -> None:
        while not self._stop.is_set():
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEREMOVED:
                    self._connected = False
                elif event.type == pygame.JOYBUTTONDOWN and event.button == self._cfg.land_button:
                    self._land_pressed.set()
            if self._connected:
                self._trigger = self._js.get_axis(self._cfg.deadman_axis)
            self._last_poll = time.monotonic()
            time.sleep(1.0 / POLL_HZ)


def print_inputs() -> None:
    """Live dump of every axis and button, to find the right indices for the config."""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("no gamepad detected")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"{js.get_name()}: {js.get_numaxes()} axes, {js.get_numbuttons()} buttons (Ctrl+C to stop)")
    try:
        while True:
            pygame.event.pump()
            axes = " ".join(f"a{i}={js.get_axis(i):+.2f}" for i in range(js.get_numaxes()))
            pressed = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
            print(f"\r{axes}  buttons={pressed}      ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        pygame.quit()
