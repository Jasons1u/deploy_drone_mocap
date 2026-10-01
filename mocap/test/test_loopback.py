"""End-to-end over real sockets: fake_server -> NatNetClient on this host."""

import socket
import threading

import pytest

from mocap_natnet import NatNetClient
from mocap_natnet import fake_server
from mocap_natnet.frames import yaw_from_quat

GROUP = "239.255.42.123"   # off the default group so a real Motive on the LAN cannot interfere


def _free_udp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("version,up", [("4.1", "z"), ("3.1", "y")])
def test_handshake_and_stream(version, up):
    cmd_port, data_port = _free_udp_port(), _free_udp_port()
    argv = ["--body", "3", "--pos", "0.5", "-0.25", "1.0", "--yaw", "30", "--rate", "200",
            "--version", version, "--up-axis", up, "--group", GROUP,
            "--command-port", str(cmd_port), "--data-port", str(data_port)]
    threading.Thread(target=fake_server.main, args=(argv,), daemon=True).start()

    client = NatNetClient("127.0.0.1", multicast_group=GROUP, command_port=cmd_port,
                          data_port=data_port, up_axis=up)
    try:
        client.start(handshake_timeout=3.0)
    except OSError as exc:   # sandbox without multicast
        pytest.skip(f"multicast unavailable: {exc}")
    try:
        assert client.natnet_version[:2] == tuple(int(v) for v in version.split("."))
        pose = client.wait_for_body(3, timeout=3.0)
        assert pose.position == pytest.approx((0.5, -0.25, 1.0), abs=1e-5)
        assert yaw_from_quat(pose.quat) == pytest.approx(0.5236, abs=1e-3)
        assert client.age(3) < 0.1
        assert client.age(99) == float("inf")
    finally:
        client.stop()


@pytest.mark.parametrize("args,body,needle", [
    (["--untracked"], 3, "not tracking it"),      # what happened on 2026-09-29
    ([], 99, "not in the stream"),
])
def test_why_no_body(args, body, needle):
    cmd_port, data_port = _free_udp_port(), _free_udp_port()
    argv = ["--body", "3", "--rate", "200", "--group", GROUP, "--command-port", str(cmd_port),
            "--data-port", str(data_port), *args]
    threading.Thread(target=fake_server.main, args=(argv,), daemon=True).start()
    client = NatNetClient("127.0.0.1", multicast_group=GROUP, command_port=cmd_port,
                          data_port=data_port)
    try:
        client.start(handshake_timeout=3.0)
    except OSError as exc:
        pytest.skip(f"multicast unavailable: {exc}")
    try:
        with pytest.raises(TimeoutError, match=needle):
            client.wait_for_body(body, timeout=0.5)
    finally:
        client.stop()
