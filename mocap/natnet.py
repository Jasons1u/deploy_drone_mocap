"""Minimal NatNet 3.x/4.x client (multicast): rigid-body poses with a per-frame callback."""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass

NAT_CONNECT, NAT_SERVERINFO, NAT_FRAMEOFDATA = 0, 1, 7
_HEADER = struct.Struct("<HH")
_RIGID_BODY = struct.Struct("<i3f4ffH")   # id, pos, quat (x, y, z, w), mean error, params


@dataclass(frozen=True)
class Body:
    id: int
    position: tuple[float, float, float]        # [m] Motive world (Z up, as streamed)
    quat: tuple[float, float, float, float]     # (w, x, y, z), body -> world
    mean_error: float                           # [m]
    valid: bool


@dataclass(frozen=True)
class Frame:
    number: int
    bodies: dict[int, Body]
    arrival: float                              # time.monotonic()


def parse_frame(packet: bytes, version) -> tuple[int, list[Body]]:
    """Frame number and rigid bodies of a NAT_FRAMEOFDATA packet (NatNet >= 3.0)."""
    sized = tuple(version[:2]) >= (4, 1)
    off = _HEADER.size
    (number,) = struct.unpack_from("<i", packet, off)
    off += 4
    (n_sets,) = struct.unpack_from("<i", packet, off)
    off += 4
    if sized:
        off += 4 + struct.unpack_from("<i", packet, off)[0]
    else:
        for _ in range(n_sets):
            off = packet.index(b"\0", off) + 1
            off += 4 + 12 * struct.unpack_from("<i", packet, off)[0]
    (n_loose,) = struct.unpack_from("<i", packet, off)
    off += 8 if sized else 4
    off += 12 * n_loose
    (n_bodies,) = struct.unpack_from("<i", packet, off)
    off += 8 if sized else 4
    bodies = []
    for _ in range(n_bodies):
        bid, px, py, pz, qx, qy, qz, qw, err, params = _RIGID_BODY.unpack_from(packet, off)
        off += _RIGID_BODY.size
        bodies.append(Body(bid, (px, py, pz), (qw, qx, qy, qz), err, bool(params & 0x01)))
    return number, bodies


class NatNetClient:
    def __init__(self, server_ip: str, local_ip: str | None = None, group: str = "239.255.42.99",
                 command_port: int = 1510, data_port: int = 1511, on_frame=None):
        self.server = (server_ip, command_port)
        self.local_ip = local_ip or route_ip(server_ip)
        self.group, self.data_port = group, data_port
        self.on_frame = on_frame
        self.app_version = self.natnet_version = None
        self.frames = 0
        self.last: Frame | None = None
        self._stop = threading.Event()
        self._thread = None
        self._sock = None

    def start(self, timeout: float = 3.0) -> "NatNetClient":
        self._handshake(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", self.data_port))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     socket.inet_aton(self.group) + socket.inet_aton(self.local_ip))
        s.settimeout(0.2)
        self._sock = s
        self._thread = threading.Thread(target=self._run, name="natnet", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()

    def body(self, body_id: int) -> tuple[Body | None, float]:
        """Latest pose of body_id and its age [s] (inf if never seen or not tracked in the latest frame)."""
        f = self.last
        if f is None or body_id not in f.bodies or not f.bodies[body_id].valid:
            return None, float("inf")
        return f.bodies[body_id], time.monotonic() - f.arrival

    def _handshake(self, timeout: float) -> None:
        payload = bytearray(271)
        payload[0:4] = b"Ping"
        payload[265:269] = bytes((4, 1, 0, 0))
        msg = _HEADER.pack(NAT_CONNECT, len(payload)) + bytes(payload)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((self.local_ip, 0))
        s.settimeout(0.5)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                s.sendto(msg, self.server)
                try:
                    data, _ = s.recvfrom(65535)
                except socket.timeout:
                    continue
                if _HEADER.unpack_from(data)[0] == NAT_SERVERINFO:
                    self.app_version = tuple(data[4 + 256:4 + 260])
                    self.natnet_version = tuple(data[4 + 260:4 + 264])
                    if tuple(self.natnet_version[:2]) < (3, 0):
                        raise RuntimeError(f"NatNet {self.natnet_version} unsupported (need >= 3.0)")
                    return
        finally:
            s.close()
        raise TimeoutError(f"no reply from Motive at {self.server[0]} (local {self.local_ip}) within {timeout} s")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            now = time.monotonic()
            if len(data) < 4 or _HEADER.unpack_from(data)[0] != NAT_FRAMEOFDATA:
                continue
            try:
                number, bodies = parse_frame(data, self.natnet_version)
            except (struct.error, ValueError):
                continue
            frame = Frame(number, {b.id: b for b in bodies}, now)
            self.last = frame
            self.frames += 1
            if self.on_frame:
                self.on_frame(frame)


def route_ip(server_ip: str) -> str:
    """Local IP of the interface that routes to server_ip."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((server_ip, 9))
        return s.getsockname()[0]
    finally:
        s.close()
