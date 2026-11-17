#!/usr/bin/env python3
"""Gamepad launcher for repeated flip flights: A or B starts fly_rates.py with that button's tape; the pad is released while it flies."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import load  # noqa: E402
from cfdeploy.gamepad import Edge, Gamepad  # noqa: E402

FLY = Path(__file__).resolve().parent / "fly_rates.py"


def wait_press(device: str, buttons: dict) -> str:
    """Open the pad, block until one of buttons (name -> number) goes down from released, close the pad; returns the name."""
    pad = Gamepad(device)
    try:
        time.sleep(0.2)   # the js driver's initial-state events: a button already held at open is not a press
        edges = {n: Edge() for n in buttons}
        for n, b in buttons.items():
            edges[n](pad.held(b))
        while pad.alive:
            for n, b in buttons.items():
                if edges[n](pad.held(b)):
                    return n
            time.sleep(0.02)
        raise OSError(f"gamepad {device} gone")
    finally:
        pad.close()


def fly(cmd: list) -> int:
    """Run one flight in the foreground; Ctrl+C reaches fly_rates (first lands, second E-STOPs), not this launcher."""
    child = subprocess.Popen(cmd)
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, metavar="TAPE", help="tape flown when A is pressed")
    ap.add_argument("--b", required=True, metavar="TAPE", help="tape flown when B is pressed")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --: fly_rates.py args for every run, e.g. -- --set launch.timeout=2.0")
    opts = ap.parse_args(argv)
    tapes, extra = {"a": opts.a, "b": opts.b}, opts.extra[1:] if opts.extra[:1] == ["--"] else opts.extra
    for name, tape in tapes.items():
        if not Path(tape).is_file():
            ap.error(f"--{name} {tape}: no such file")
    cfg = load()
    buttons, runs = {n: getattr(cfg.gamepad.buttons, n) for n in tapes}, 0
    while True:
        print(f"\nready: A = {Path(opts.a).name}, B = {Path(opts.b).name} (Ctrl+C here quits)", flush=True)
        try:
            name = wait_press(cfg.gamepad.device, buttons)
        except KeyboardInterrupt:
            print(f"\nlauncher done: {runs} runs")
            return 0
        except OSError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 1
        runs += 1
        cmd = [sys.executable, str(FLY), "--plan", "flip", "--tape", tapes[name], *extra]
        print(f"\n[{name.upper()}] run {runs}: fly_rates.py --tape {Path(tapes[name]).name} {' '.join(extra)}", flush=True)
        print(f"[{name.upper()}] run {runs} ended, exit {fly(cmd)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
