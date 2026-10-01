"""NatNet client: stream rigid-body poses from Motive in a background thread.

    with NatNetClient(server_ip="192.168.1.10") as mocap:
        pose = mocap.wait_for_body(1, timeout=3.0)
        ...
        pose = mocap.pose(1)          # latest Pose, or None
        age = mocap.age(1)            # seconds since the last valid frame for that body

Multicast only (Motive's default). The client sends NAT_CONNECT to the server's command port and
reads the NatNet version from the NAT_SERVERINFO reply, so the frame parser always matches what
Motive is actually streaming. Pass `natnet_version` to skip the handshake (e.g. when the command
port is firewalled).

Poses are converted to the project convention (Z-up world, FLU body, see frames.py). Velocity is
a finite difference of consecutive frames, low-pass filtered, timed by Motive frame numbers at the
measured frame rate — not by arrival times, which bunch up on the network (see _VelocityFilter).

A body whose `tracking_valid` bit is clear is NOT updated, so it goes stale and `age()` grows —
callers use that as their dropout signal.
"""

from __future__ import annotations

import collections
import logging
import socket
import threading
import time
from dataclasses import dataclass

from . import protocol
from .frames import UP_AXES, quat_mul, to_z_up

log = logging.getLogger(__name__)

DEFAULT_MULTICAST_GROUP = "239.255.42.99"
DEFAULT_COMMAND_PORT = 1510
DEFAULT_DATA_PORT = 1511

VELOCITY_RESET_GAP_S = 0.1   # longer gaps restart the finite difference instead of averaging
RATE_WINDOW_S = 2.0          # arrival times over this window measure Motive's frame rate
MIN_RATE_WINDOW_S = 0.5      # velocity reads zero until the rate has been measured this long


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]         # [m] Z-up world
    quat: tuple[float, float, float, float]      # (w, x, y, z) FLU body -> Z-up world
    velocity: tuple[float, float, float]         # [m/s] Z-up world, filtered finite difference
    frame_number: int
    mean_error: float                            # [m] Motive marker residual
    stamp: float                                 # time.monotonic() when the frame arrived


class _VelocityFilter:
    """Finite-difference velocity timed by frame numbers, low-pass filtered.

    Consecutive arrival times cannot be differenced: frames reach us in network bursts, so two
    frames 8 ms apart in Motive can arrive microseconds apart, and position / tiny-dt spikes to
    metres per second. Frame numbers are exact, so dt = (frame gap) / (frame rate), with the rate
    measured from arrival times over a few seconds, where the jitter averages out.
    """

    def __init__(self, alpha: float):
        self._alpha = alpha
        self._arrivals = collections.deque()   # (frame_number, arrival time)
        self._prev = None                      # (frame_number, position)
        self._vel = (0.0, 0.0, 0.0)

    def _frame_rate(self) -> float | None:
        (f0, t0), (f1, t1) = self._arrivals[0], self._arrivals[-1]
        if t1 - t0 < MIN_RATE_WINDOW_S:
            return None
        return (f1 - f0) / (t1 - t0)

    def update(self, frame: int, pos, t: float):
        if self._arrivals and frame <= self._arrivals[-1][0]:
            if frame == self._arrivals[-1][0]:
                return self._vel                    # duplicate frame
            self._arrivals.clear()                  # frame counter went back: Motive restarted
            self._prev = None
        self._arrivals.append((frame, t))
        while t - self._arrivals[0][1] > RATE_WINDOW_S:
            self._arrivals.popleft()

        rate = self._frame_rate()
        if self._prev is None or rate is None:
            self._vel = (0.0, 0.0, 0.0)
        else:
            dt = (frame - self._prev[0]) / rate
            if dt > VELOCITY_RESET_GAP_S:
                self._vel = (0.0, 0.0, 0.0)
            else:
                a = self._alpha
                self._vel = tuple(a * (p - q) / dt + (1.0 - a) * v
                                  for p, q, v in zip(pos, self._prev[1], self._vel))
        self._prev = (frame, pos)
        return self._vel


class NatNetClient:
    def __init__(self, server_ip: str, local_ip: str = "0.0.0.0",
                 multicast_group: str = DEFAULT_MULTICAST_GROUP,
                 command_port: int = DEFAULT_COMMAND_PORT, data_port: int = DEFAULT_DATA_PORT,
                 up_axis: str = "z", natnet_version: str | None = None,
                 velocity_alpha: float = 0.3, body_rotation=(1.0, 0.0, 0.0, 0.0),
                 on_pose=None):
        """on_pose(body_id, Pose) is called from the receive thread for every valid body update.

        body_rotation: fixed (w, x, y, z) applied in the body frame of every pose, q = q_motive *
        body_rotation. Corrects a rigid body whose axes are slightly off the drone's IMU axes.
        """
        if not server_ip:
            raise ValueError("server_ip is required (IP of the PC running Motive)")
        if up_axis not in UP_AXES:
            raise ValueError(f"up_axis must be one of {UP_AXES}, got {up_axis!r}")
        self._server = (server_ip, command_port)
        self._local_ip = local_ip
        self._group = multicast_group
        self._data_port = data_port
        self._up_axis = up_axis
        self._version = protocol.parse_version(natnet_version) if natnet_version else None
        self._alpha = velocity_alpha
        self._on_pose = on_pose
        self._body_rotation = tuple(body_rotation)

        self.server_info: protocol.ServerInfo | None = None
        self.frames_received = 0
        self.parse_errors = 0

        self._poses: dict[int, Pose] = {}
        self._streamed: set[int] = set()     # every rigid body id in the stream, tracked or not
        self._filters: dict[int, _VelocityFilter] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._cmd_sock = None
        self._data_sock = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, handshake_timeout: float = 3.0) -> "NatNetClient":
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.bind((self._local_ip, 0))
        self._data_sock = self._open_data_socket()

        if self._version is None:
            self.server_info = self._handshake(handshake_timeout)
            self._version = self.server_info.natnet_version
            log.info("Motive %s (%s), NatNet %s", self.server_info.app_name,
                     _fmt(self.server_info.app_version), _fmt(self._version))
        protocol.check_version(self._version)

        self._thread = threading.Thread(target=self._recv_loop, name="natnet-rx", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for s in (self._cmd_sock, self._data_sock):
            if s is not None:
                s.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------------
    # Queries (thread-safe)
    # ------------------------------------------------------------------

    @property
    def natnet_version(self):
        return self._version

    def pose(self, body_id: int) -> Pose | None:
        with self._lock:
            return self._poses.get(body_id)

    def age(self, body_id: int) -> float:
        """Seconds since the last valid frame for body_id (inf if never seen)."""
        p = self.pose(body_id)
        return float("inf") if p is None else time.monotonic() - p.stamp

    def body_ids(self) -> list[int]:
        with self._lock:
            return sorted(self._poses)

    def wait_for_body(self, body_id: int, timeout: float) -> Pose:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            p = self.pose(body_id)
            if p is not None:
                return p
            time.sleep(0.02)
        raise TimeoutError(self._why_no_body(body_id, timeout))

    def _why_no_body(self, body_id: int, timeout: float) -> str:
        if self.frames_received == 0:
            return (f"no mocap frames within {timeout:.1f} s: Motive is not streaming to this "
                    f"machine (check Streaming > Broadcast Frame Data, multicast, the network)")
        with self._lock:
            streamed = sorted(self._streamed)
        if body_id in streamed:
            return (f"rigid body {body_id} is streamed but Motive is not tracking it "
                    f"({self.frames_received} frames, tracked in none). Is the drone in the "
                    f"cameras' view with its markers on? Motive needs 3+ of them visible.")
        return (f"rigid body {body_id} is not in the stream ({self.frames_received} frames; "
                f"rigid bodies streamed: {streamed or 'none'}). Check the Streaming ID in Motive.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_data_socket(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", self._data_port))
        mreq = socket.inet_aton(self._group) + socket.inet_aton(self._local_ip)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        s.settimeout(0.5)
        return s

    def _handshake(self, timeout: float) -> protocol.ServerInfo:
        self._cmd_sock.settimeout(0.5)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._cmd_sock.sendto(protocol.build_connect(), self._server)
            try:
                data, _ = self._cmd_sock.recvfrom(65535)
            except socket.timeout:
                continue
            if protocol.message_id(data) == protocol.NAT_SERVERINFO:
                return protocol.parse_server_info(data)
        raise TimeoutError(
            f"no NAT_SERVERINFO from {self._server[0]}:{self._server[1]} within {timeout:.1f} s. "
            f"Check the Motive IP and that streaming is enabled, or set natnet_version explicitly.")

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._data_sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if protocol.message_id(data) != protocol.NAT_FRAMEOFDATA:
                continue
            now = time.monotonic()
            try:
                frame = protocol.parse_frame(data, self._version)
            except ValueError as exc:
                self.parse_errors += 1
                log.warning("dropping malformed NatNet frame: %s", exc)
                continue
            self.frames_received += 1
            for body in frame.bodies:
                self._streamed.add(body.id)
                if body.tracking_valid:
                    self._update(body, frame.frame_number, now)

    def _update(self, body: protocol.RigidBody, frame_number: int, now: float) -> None:
        pos, quat = to_z_up(body.position, body.quat, self._up_axis)
        quat = quat_mul(quat, self._body_rotation)
        filt = self._filters.setdefault(body.id, _VelocityFilter(self._alpha))
        pose = Pose(pos, quat, filt.update(frame_number, pos, now), frame_number, body.mean_error,
                    now)
        with self._lock:
            self._poses[body.id] = pose
        if self._on_pose is not None:
            self._on_pose(body.id, pose)


def _fmt(version) -> str:
    return ".".join(str(v) for v in version)
