"""Smoke test: print one rigid body's pose live. No drone needed.

    python3 -m mocap_natnet --server 192.168.1.10 --body 1

Check before flying: the rate matches Motive's frame rate, z is height above the floor, moving the
drone forward (its nose direction) increases x at yaw ~ 0, and tilting it right gives roll > 0.
"""

from __future__ import annotations

import argparse
import logging
import math
import time

from .client import DEFAULT_MULTICAST_GROUP, NatNetClient
from .frames import UP_AXES, rpy_from_quat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--server", required=True, help="IP of the PC running Motive")
    ap.add_argument("--local", default="0.0.0.0", help="local interface IP (multi-NIC hosts)")
    ap.add_argument("--body", type=int, default=1, help="rigid body ID")
    ap.add_argument("--up-axis", choices=UP_AXES, default="z", help="Motive streaming up axis")
    ap.add_argument("--group", default=DEFAULT_MULTICAST_GROUP)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with NatNetClient(args.server, local_ip=args.local, multicast_group=args.group,
                      up_axis=args.up_axis) as client:
        client.wait_for_body(args.body, timeout=5.0)
        last_frames, last_t = client.frames_received, time.monotonic()
        rate = 0.0
        try:
            while True:
                time.sleep(0.1)
                now = time.monotonic()
                if now - last_t >= 1.0:
                    rate = (client.frames_received - last_frames) / (now - last_t)
                    last_frames, last_t = client.frames_received, now
                p = client.pose(args.body)
                r, pt, y = (math.degrees(a) for a in rpy_from_quat(p.quat))
                print(f"\r{rate:5.0f} Hz  age {client.age(args.body) * 1e3:5.1f} ms  "
                      f"pos {p.position[0]:+6.3f} {p.position[1]:+6.3f} {p.position[2]:+6.3f} m  "
                      f"rpy {r:+6.1f} {pt:+6.1f} {y:+7.1f} deg  "
                      f"vel {p.velocity[0]:+5.2f} {p.velocity[1]:+5.2f} {p.velocity[2]:+5.2f} m/s  "
                      f"err {p.mean_error * 1e3:4.2f} mm", end="", flush=True)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
