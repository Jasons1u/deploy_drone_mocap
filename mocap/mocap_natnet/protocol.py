"""NatNet wire format: build and parse the few messages this project uses.

Only what hover needs: the connect/server-info handshake and the rigid-body section of a
frame-of-data packet. Supports NatNet 3.0 through 4.x. NatNet 4.1 inserted an int32 byte count
after each section count; that is the only layout difference that matters here.

Pure functions, no sockets. `client.py` owns the I/O, and `fake_server.py` + the tests reuse the
builders so everything speaks exactly the same bytes.

Frame layout up to the rigid bodies (little-endian):
    uint16 message_id, uint16 payload_size
    int32  frame_number
    int32  n_marker_sets    [int32 nbytes if >= 4.1]   each: cstring name, int32 n, n * 3f
    int32  n_unlabeled      [int32 nbytes if >= 4.1]   n * 3f
    int32  n_rigid_bodies   [int32 nbytes if >= 4.1]   each: int32 id, 3f pos, 4f quat (x,y,z,w),
                                                             f mean_error, uint16 params
Everything after the rigid bodies (skeletons, labeled markers, timestamps, ...) is ignored.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

NAT_CONNECT = 0
NAT_SERVERINFO = 1
NAT_FRAMEOFDATA = 7

MIN_VERSION = (3, 0)
CLIENT_VERSION = (4, 1, 0, 0)   # what we announce in NAT_CONNECT, as the official SDK does

_HEADER = struct.Struct("<HH")
_RIGID_BODY = struct.Struct("<i3f4ffH")   # id, pos, quat(x,y,z,w), mean_error, params
_CONNECT_PAYLOAD_LEN = 271                # "Ping" padded; client version at bytes 265..268


@dataclass(frozen=True)
class ServerInfo:
    app_name: str
    app_version: tuple[int, int, int, int]
    natnet_version: tuple[int, int, int, int]
    multicast: bool | None = None       # None: server too old to report connection info
    multicast_group: str | None = None
    data_port: int | None = None


@dataclass(frozen=True)
class RigidBody:
    id: int
    position: tuple[float, float, float]           # [m], Motive world frame
    quat: tuple[float, float, float, float]        # (w, x, y, z), body -> Motive world
    mean_error: float                              # [m] mean marker residual
    tracking_valid: bool


@dataclass(frozen=True)
class Frame:
    frame_number: int
    bodies: tuple[RigidBody, ...]


def message_id(packet: bytes) -> int | None:
    if len(packet) < _HEADER.size:
        return None
    return _HEADER.unpack_from(packet)[0]


def _has_section_sizes(version) -> bool:
    return tuple(version[:2]) >= (4, 1)


def parse_version(text: str) -> tuple[int, int, int, int]:
    """'4.1' -> (4, 1, 0, 0)."""
    parts = [int(p) for p in str(text).split(".")]
    return tuple((parts + [0, 0, 0, 0])[:4])


def check_version(version) -> None:
    if tuple(version[:2]) < MIN_VERSION:
        raise ValueError(f"NatNet {version[0]}.{version[1]} unsupported (need >= 3.0)")


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------

def build_connect(client_version=CLIENT_VERSION) -> bytes:
    payload = bytearray(_CONNECT_PAYLOAD_LEN)
    payload[0:4] = b"Ping"
    payload[265:269] = bytes(client_version)
    return _HEADER.pack(NAT_CONNECT, len(payload)) + bytes(payload)


def build_server_info(info: ServerInfo) -> bytes:
    payload = bytearray(info.app_name.encode()[:255].ljust(256, b"\0"))
    payload += bytes(info.app_version) + bytes(info.natnet_version)
    if info.data_port is not None:
        payload += struct.pack("<QH?", 0, info.data_port, bool(info.multicast))
        payload += socket.inet_aton(info.multicast_group or "0.0.0.0")
    return _HEADER.pack(NAT_SERVERINFO, len(payload)) + bytes(payload)


def parse_server_info(packet: bytes) -> ServerInfo:
    msg_id, _ = _HEADER.unpack_from(packet)
    if msg_id != NAT_SERVERINFO:
        raise ValueError(f"expected NAT_SERVERINFO, got message id {msg_id}")
    off = _HEADER.size
    name = packet[off:off + 256].split(b"\0", 1)[0].decode(errors="replace")
    off += 256
    app_version = tuple(packet[off:off + 4])
    natnet_version = tuple(packet[off + 4:off + 8])
    off += 8
    if len(app_version) != 4 or len(natnet_version) != 4:
        raise ValueError("truncated NAT_SERVERINFO")

    # Connection info (NatNet 3.0+): uint64 clock freq, uint16 data port, bool multicast, 4B group.
    if len(packet) >= off + 15:
        _, data_port, multicast = struct.unpack_from("<QH?", packet, off)
        group = socket.inet_ntoa(packet[off + 11:off + 15])
        return ServerInfo(name, app_version, natnet_version, multicast, group, data_port)
    return ServerInfo(name, app_version, natnet_version)


# ---------------------------------------------------------------------------
# Frame of data
# ---------------------------------------------------------------------------

class _Reader:
    def __init__(self, data: bytes, offset: int):
        self.data = data
        self.off = offset

    def i32(self) -> int:
        (v,) = struct.unpack_from("<i", self.data, self.off)
        self.off += 4
        return v

    def skip(self, n: int) -> None:
        if n < 0 or self.off + n > len(self.data):
            raise ValueError("NatNet frame truncated")
        self.off += n

    def cstring(self) -> None:
        end = self.data.index(b"\0", self.off)
        self.off = end + 1


def parse_frame(packet: bytes, version) -> Frame:
    """Parse the rigid bodies out of a NAT_FRAMEOFDATA packet. Raises ValueError if malformed."""
    check_version(version)
    sized = _has_section_sizes(version)
    try:
        r = _Reader(packet, _HEADER.size)
        frame_number = r.i32()

        n_sets = r.i32()
        if sized:
            r.skip(r.i32())
        else:
            for _ in range(n_sets):
                r.cstring()
                r.skip(12 * r.i32())

        n_unlabeled = r.i32()
        if sized:
            r.i32()
        r.skip(12 * n_unlabeled)

        n_bodies = r.i32()
        if sized:
            r.i32()
        bodies = []
        for _ in range(n_bodies):
            rb_id, px, py, pz, qx, qy, qz, qw, err, params = _RIGID_BODY.unpack_from(r.data, r.off)
            r.off += _RIGID_BODY.size
            bodies.append(RigidBody(rb_id, (px, py, pz), (qw, qx, qy, qz), err, bool(params & 0x01)))
    except struct.error as exc:
        raise ValueError(f"NatNet frame truncated: {exc}") from exc
    return Frame(frame_number, tuple(bodies))


def build_frame(frame_number: int, bodies, version, marker_sets=(), unlabeled=()) -> bytes:
    """Inverse of parse_frame, for tests and the fake server.

    marker_sets: iterable of (name, [(x, y, z), ...]); unlabeled: [(x, y, z), ...].
    """
    check_version(version)
    sized = _has_section_sizes(version)

    def section(count: int, body: bytes) -> bytes:
        head = struct.pack("<i", count)
        return head + (struct.pack("<i", len(body)) if sized else b"") + body

    sets = b"".join(
        name.encode() + b"\0" + struct.pack("<i", len(pts)) + b"".join(struct.pack("<3f", *p) for p in pts)
        for name, pts in marker_sets)
    loose = b"".join(struct.pack("<3f", *p) for p in unlabeled)
    rbs = b"".join(
        _RIGID_BODY.pack(b.id, *b.position, b.quat[1], b.quat[2], b.quat[3], b.quat[0],
                         b.mean_error, 0x01 if b.tracking_valid else 0x00)
        for b in bodies)

    payload = (struct.pack("<i", frame_number)
               + section(len(marker_sets), sets)
               + section(len(unlabeled), loose)
               + section(len(bodies), rbs)
               + struct.pack("<i", 0))   # zero skeletons: a real frame continues past the bodies
    return _HEADER.pack(NAT_FRAMEOFDATA, len(payload)) + payload
