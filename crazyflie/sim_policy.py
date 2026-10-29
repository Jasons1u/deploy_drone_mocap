#!/usr/bin/env python3
"""Desktop sim of a policy flight (cfdeploy/sim.py) with RViz: drag the target and obstacles, keys to reset / pause."""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # repo root: cfdeploy
from cfdeploy.config import add_override_args, apply_overrides, load, policy_path  # noqa: E402
from cfdeploy.logs import run_dir  # noqa: E402
from cfdeploy.mlp_policy import load_policy  # noqa: E402
from cfdeploy.obstacles import obstacle_ood  # noqa: E402
from cfdeploy.sim import Sim  # noqa: E402
from cfdeploy.viz_link import VizSocket, launch_rviz, stop_rviz  # noqa: E402

# #### // Terminal keys // ####
class Keys:
    """Single key presses from the terminal, no Enter needed."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def get(self) -> str | None:
        if select.select([self.fd], [], [], 0)[0]:
            return os.read(self.fd, 1).decode(errors="ignore").lower()
        return None

    def close(self) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def interrupt(signum, frame):
    raise KeyboardInterrupt   # SIGTERM / SIGHUP (terminal closed) clean up like Ctrl+C, so RViz is not left behind


# #### // CLI // ####
def parse_args(argv=None) -> tuple:
    ap = argparse.ArgumentParser(description=__doc__, epilog="Starts RViz too (crazyflie/viz/run_rviz.sh); closing it ends the sim. "
                                 "Keys: r reset drone, o reset obstacles, p pause, q quit.")
    ap.add_argument("--policy", help="exported .npz (default policy_flight.policy)")
    ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "Z"), help="drone start and first target (default sim.start)")
    ap.add_argument("--obstacle", nargs=4, type=float, action="append", metavar=("X", "Y", "Z", "R"),
                    help="obstacle start, repeat for more (default sim.obstacles)")
    ap.add_argument("--randomize", action="store_true", help="draw model params, drift and kicks from the training ranges each reset")
    ap.add_argument("--seed", type=int, help="(default sim.seed)")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (0.5 = slow motion)")
    ap.add_argument("--duration", type=float, default=None, help="[s] sim time, then quit (default: until q / Ctrl+C)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--no-rviz", action="store_true", help="don't start RViz (e.g. it is already running)")
    add_override_args(ap)
    opts = ap.parse_args(argv)
    try:
        cfg = apply_overrides(load(), opts.set)
    except ValueError as exc:
        ap.error(str(exc))
    cfg.policy_flight.policy = policy_path(cfg, opts.policy)
    for key, value in (("start", opts.start), ("obstacles", opts.obstacle), ("seed", opts.seed)):
        if value is not None:
            cfg.sim[key] = value
    return cfg, opts


def main(argv=None) -> int:
    cfg, opts = parse_args(argv)
    policy, _, note = load_policy(cfg.policy_flight.policy)
    if note:
        print(note)
    if cfg.sim.obstacle_max_speed is None:
        cfg.sim.obstacle_max_speed = policy.max_obstacle_speed()
    for o in cfg.sim.obstacles:
        why = obstacle_ood(o[:3])
        if why:
            print(f"warning: start {why}")
    try:
        link = VizSocket(cfg.viz)
    except OSError as exc:
        print(f"ABORT: udp:{cfg.viz.cmd_port} busy ({exc}); is fly_policy.py --viz running?", file=sys.stderr)
        return 1
    print(f"policy {Path(cfg.policy_flight.policy).name}; obstacle speed cap {cfg.sim.obstacle_max_speed:g} m/s; "
          f"{'randomized' if opts.randomize else 'sysID-fit'} ROM")
    sim = Sim(cfg, policy, opts.randomize)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, interrupt)
        except ValueError:
            pass   # not the main thread (tests)
    rviz = None if opts.no_rviz else launch_rviz(cfg)
    keys = Keys() if sys.stdin.isatty() else None
    print("keys: r reset drone, o reset obstacles, p pause, q quit")
    period, n = cfg.training.policy_dt / opts.speed, 0
    every = max(1, int(round(1.0 / (cfg.viz.rate * cfg.training.policy_dt))))
    next_t = time.monotonic()
    try:
        while opts.duration is None or sim.t < opts.duration:
            for msg in link.poll():
                sim.handle(msg)
            k = keys.get() if keys else None
            if k == "q":
                break
            if rviz is not None and rviz.poll() is not None:
                print(f"RViz closed (exit {rviz.returncode}): stopping the sim")
                break
            if k == "p":
                sim.paused = not sim.paused
                print("  paused" if sim.paused else "  running")
            elif k == "r":
                sim.reset_drone()
            elif k == "o":
                sim.reset_obstacles()
            if not sim.paused:
                sim.tick()
            if n % every == 0:
                link.send(sim.state())
            n += 1
            next_t += period
            lag = next_t - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.1:
                next_t = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if keys:
            keys.close()
        stop_rviz(rviz)
        link.close()
    print(f"min gap {sim.min_gap * 100:.1f} cm, collisions {sim.collisions} (since the last reset)")
    if not opts.no_log:
        out = run_dir(cfg, "sim")
        sim.save(out)
        print(f"log: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
