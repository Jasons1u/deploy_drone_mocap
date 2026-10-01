"""Pre-flight mocap gate: is this drone's rigid body streaming, fast enough, and actually moving (not frozen)?

Uses the mocap settings shared with cf-hover (crazyflie/config/hover.yaml), or --config.
"""

import argparse
import sys
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="default: crazyflie/config/hover.yaml")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    from cf_mocap.config import DEFAULT_CONFIG, load_config
    from mocap_natnet import NatNetClient

    m = load_config(args.config or DEFAULT_CONFIG).mocap
    stamps, poses = [], []

    def on_pose(body_id, pose):
        if body_id == m.rigid_body_id:
            stamps.append(pose.stamp)
            poses.append(pose.position)

    client = NatNetClient(m.server_ip, local_ip=m.local_ip, up_axis=m.up_axis, natnet_version=m.natnet_version,
                          body_rotation=m.body_rotation(), on_pose=on_pose)
    try:
        client.start()
    except (TimeoutError, OSError) as exc:
        print(f"NO MOTIVE at {m.server_ip}: {exc}")
        sys.exit(1)
    try:
        time.sleep(args.seconds)
        if len(poses) < 2:
            try:
                client.wait_for_body(m.rigid_body_id, timeout=0.0)
            except TimeoutError as exc:
                print(f"NO DATA: {exc}")
            sys.exit(1)
    finally:
        client.stop()

    p = np.array(poses)
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    moves = np.linalg.norm(np.diff(p, axis=0), axis=1)
    frozen = float((moves < 1e-5).mean())
    print(f"rigid body {m.rigid_body_id} @ {m.server_ip}: {rate:.0f} Hz, position {np.round(p[-1], 3).tolist()}, "
          f"jitter std {p.std(0).max() * 1e3:.2f} mm, identical frames {frozen * 100:.0f}%")
    if rate < 100 or frozen > 0.5:
        print("NOT LIVE -- low rate or frozen pose")
        sys.exit(1)
    print("LIVE")


if __name__ == "__main__":
    main()
