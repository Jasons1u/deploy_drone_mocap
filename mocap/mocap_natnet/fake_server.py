"""Fake Motive: answer NAT_CONNECT and multicast synthetic rigid-body frames.

For bench-testing everything downstream of NatNet without OptiTrack hardware:

    python3 -m mocap_natnet.fake_server --body 1 --pos 0 0 0.02            # sitting on the floor
    python3 -m mocap_natnet.fake_server --body 1 --pos 0 0 0.5 --circle 0.3 # moving in a circle
    python3 -m mocap_natnet.fake_server --version 3.1 --up-axis y           # older / Y-up Motive

Then point the client at this machine (server_ip 127.0.0.1 when on the same host).
"""

from __future__ import annotations

import argparse
import math
import socket
import threading
import time

from . import protocol
from .client import DEFAULT_COMMAND_PORT, DEFAULT_DATA_PORT, DEFAULT_MULTICAST_GROUP
from .frames import UP_AXES, from_z_up


def _answer_connects(sock: socket.socket, info: protocol.ServerInfo, stop: threading.Event):
    reply = protocol.build_server_info(info)
    sock.settimeout(0.5)
    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        if protocol.message_id(data) == protocol.NAT_CONNECT:
            sock.sendto(reply, addr)
            print(f"[fake_server] NAT_CONNECT from {addr[0]}:{addr[1]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--body", type=int, default=1, help="rigid body ID")
    ap.add_argument("--pos", type=float, nargs=3, default=(0.0, 0.0, 0.02), metavar=("X", "Y", "Z"),
                    help="Z-up world position [m]")
    ap.add_argument("--yaw", type=float, default=0.0, help="yaw [deg]")
    ap.add_argument("--circle", type=float, default=0.0, help="circle radius [m] (0 = static)")
    ap.add_argument("--period", type=float, default=8.0, help="circle period [s]")
    ap.add_argument("--rate", type=float, default=120.0, help="frame rate [Hz]")
    ap.add_argument("--version", default="4.1", help="NatNet version to emulate")
    ap.add_argument("--up-axis", choices=UP_AXES, default="z", help="Motive up axis to emulate")
    ap.add_argument("--untracked", action="store_true", help="clear the tracking-valid bit")
    ap.add_argument("--group", default=DEFAULT_MULTICAST_GROUP)
    ap.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT)
    ap.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    args = ap.parse_args(argv)

    version = protocol.parse_version(args.version)
    info = protocol.ServerInfo("Motive (fake)", (3, 1, 0, 0), version, True, args.group,
                               args.data_port)

    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    cmd.bind(("", args.command_port))
    data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    data.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    data.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

    stop = threading.Event()
    threading.Thread(target=_answer_connects, args=(cmd, info, stop), daemon=True).start()
    print(f"[fake_server] NatNet {args.version}, up={args.up_axis}, body {args.body} @ "
          f"{args.rate:.0f} Hz -> {args.group}:{args.data_port}  (Ctrl+C to stop)")

    period = 1.0 / args.rate
    t0 = time.monotonic()
    frame = 0
    try:
        while True:
            t = time.monotonic() - t0
            x, y, z = args.pos
            yaw = math.radians(args.yaw)
            if args.circle > 0.0:
                w = 2.0 * math.pi / args.period
                x += args.circle * math.cos(w * t)
                y += args.circle * math.sin(w * t)
            quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            pos_m, quat_m = from_z_up((x, y, z), quat, args.up_axis)
            body = protocol.RigidBody(args.body, pos_m, quat_m, 0.0002, not args.untracked)
            packet = protocol.build_frame(frame, [body], version,
                                          marker_sets=[("drone", [pos_m] * 4)], unlabeled=[pos_m])
            data.sendto(packet, (args.group, args.data_port))
            frame += 1
            time.sleep(max(0.0, t0 + frame * period - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cmd.close()
        data.close()


if __name__ == "__main__":
    main()
