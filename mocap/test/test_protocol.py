import struct

import pytest

from mocap_natnet import protocol
from mocap_natnet.protocol import RigidBody, ServerInfo

VERSIONS = [(3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0)]

BODIES = [
    RigidBody(1, (0.1, -0.2, 0.3), (1.0, 0.0, 0.0, 0.0), 0.0005, True),
    RigidBody(7, (1.5, 2.5, -0.5), (0.5, 0.5, -0.5, 0.5), 0.001, False),
]


@pytest.mark.parametrize("version", VERSIONS)
def test_frame_round_trip(version):
    packet = protocol.build_frame(
        42, BODIES, version,
        marker_sets=[("drone", [(1, 2, 3)] * 4), ("all", [(4, 5, 6)])],
        unlabeled=[(7, 8, 9), (1, 1, 1)])
    frame = protocol.parse_frame(packet, version)
    assert frame.frame_number == 42
    assert [b.id for b in frame.bodies] == [1, 7]
    for got, want in zip(frame.bodies, BODIES):
        assert got.position == pytest.approx(want.position)
        assert got.quat == pytest.approx(want.quat)
        assert got.tracking_valid == want.tracking_valid


def test_version_mismatch_is_detected_not_silently_misparsed():
    """A 4.1 packet parsed as 4.0 must fail or give different bodies, never the same garbage-free
    result — this is the bug class the old client had."""
    packet = protocol.build_frame(1, BODIES, (4, 1, 0, 0), marker_sets=[("m", [(0, 0, 0)])])
    try:
        frame = protocol.parse_frame(packet, (4, 0, 0, 0))
    except ValueError:
        return
    assert [b.id for b in frame.bodies] != [1, 7]


def test_truncated_frame_raises():
    packet = protocol.build_frame(1, BODIES, (4, 1, 0, 0))
    with pytest.raises(ValueError):
        protocol.parse_frame(packet[:30], (4, 1, 0, 0))


def test_old_version_rejected():
    with pytest.raises(ValueError):
        protocol.parse_frame(b"\x07\x00\x00\x00", (2, 10, 0, 0))


def test_server_info_round_trip():
    info = ServerInfo("Motive", (3, 1, 2, 0), (4, 1, 0, 0), True, "239.255.42.99", 1511)
    assert protocol.parse_server_info(protocol.build_server_info(info)) == info


def test_server_info_without_connection_block():
    info = ServerInfo("Motive", (2, 3, 0, 0), (3, 0, 0, 0))
    assert protocol.parse_server_info(protocol.build_server_info(info)) == info


def test_connect_packet_layout():
    pkt = protocol.build_connect()
    msg_id, size = struct.unpack_from("<HH", pkt)
    assert msg_id == protocol.NAT_CONNECT
    assert size == len(pkt) - 4
    assert pkt[4:8] == b"Ping"
    assert tuple(pkt[4 + 265:4 + 269]) == protocol.CLIENT_VERSION
