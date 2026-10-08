"""Pilot inputs: Linux joystick (deadman, buttons, sticks), terminal kill keys, and stick -> velocity mapping."""

from __future__ import annotations

import math
import os
import select
import struct
import sys
import termios
import threading
import tty

import numpy as np


# #### // Gamepad (Linux js API) // ####
class Gamepad:
    """Joystick reader thread; a read error (controller gone) counts as every input released."""

    EVENT = struct.Struct("<IhBB")

    def __init__(self, device: str):
        self.fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        self.buttons, self.axes, self.alive = {}, {}, True
        self.stop_evt = threading.Event()
        threading.Thread(target=self.read_loop, daemon=True).start()

    def read_loop(self):
        while not self.stop_evt.is_set() and self.alive:
            try:
                if not select.select([self.fd], [], [], 0.05)[0]:
                    continue
                data = os.read(self.fd, self.EVENT.size * 64)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if not data:
                self.alive = False
                break
            for i in range(0, len(data) - self.EVENT.size + 1, self.EVENT.size):
                _, value, typ, num = self.EVENT.unpack_from(data, i)
                if typ & 0x01:
                    self.buttons[num] = bool(value)
                elif typ & 0x02:
                    self.axes[num] = value / 32767.0

    def held(self, button: int) -> bool:
        return self.alive and self.buttons.get(button, False)

    def axis(self, num: int) -> float | None:
        """[-1, 1], None if never reported or the controller is gone."""
        return self.axes.get(num) if self.alive else None

    def trigger_held(self, num: int, threshold: float) -> bool:
        a = self.axis(num)
        return a is not None and a > threshold

    def close(self):
        self.stop_evt.set()
        os.close(self.fd)


class Edge:
    """Rising-edge detector for buttons."""

    def __init__(self):
        self.prev = False

    def __call__(self, now: bool) -> bool:
        rising = now and not self.prev
        self.prev = now
        return rising


# #### // Terminal kill keys // ####
class NoKeyboard:
    kill = land = False

    def close(self):
        pass


class Keyboard:
    """SPACE -> kill, L -> land; needs a TTY on stdin."""

    def __init__(self):
        self.kill = self.land = False
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.stop_evt = threading.Event()
        threading.Thread(target=self.read_loop, daemon=True).start()

    def read_loop(self):
        while not self.stop_evt.is_set():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                c = os.read(self.fd, 1)
                if c == b" ":
                    self.kill = True
                elif c in (b"l", b"L"):
                    self.land = True

    def close(self):
        self.stop_evt.set()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def keyboard():
    return Keyboard() if sys.stdin.isatty() else NoKeyboard()


# #### // Sticks -> velocity command // ####
def sticks(pad, axes, deadband: float) -> np.ndarray:
    """(forward, right, up, yaw_cw) in [-1, 1] with deadband; Linux js axes are + right / + down."""
    raw = [-(pad.axis(axes.ry) or 0.0), pad.axis(axes.rx) or 0.0, -(pad.axis(axes.ly) or 0.0), pad.axis(axes.lx) or 0.0]
    out = np.array([0.0 if abs(x) < deadband else (x - math.copysign(deadband, x)) / (1 - deadband) for x in raw])
    return np.clip(out, -1.0, 1.0)


def stick_velocity(s, yaw_rad, cfg) -> tuple:
    """Sticks -> world velocity (m/s) and CCW yaw rate (deg/s); cfg = the sticks section."""
    fwd, left = s[0] * cfg.max_xy, -s[1] * cfg.max_xy
    if cfg.room_frame:
        vx, vy = fwd, left
    else:
        c, si = math.cos(yaw_rad), math.sin(yaw_rad)
        vx, vy = c * fwd - si * left, si * fwd + c * left
    return np.array([vx, vy, s[2] * cfg.max_z]), -s[3] * cfg.max_yaw_rate
