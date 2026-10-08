"""Crazyflie link: connect with a watchdog, firmware params, telemetry log blocks, and the mocap -> extpose relay."""

from __future__ import annotations

import bisect
import sys
import threading
import time

from cfdeploy.config import firmware_params, repo_path

CONTROLLER_VELCTL = 6        # stabilizer.controller of the out-of-tree velocity controller
SUP_CAN_BE_ARMED, SUP_ARMED = 1 << 0, 1 << 1
SUP_LOCKED = 1 << 6          # supervisor.info: locked after an emergency stop, cleared only by a restart
SUP_CRASHED = 1 << 7         # supervisor.info: tumbled/crashed; arming refused until recovered

LOG_BLOCKS = {               # periods come from config telemetry.flight / telemetry.sysid
    "pos": [("stateEstimate.x", "float"), ("stateEstimate.y", "float"), ("stateEstimate.z", "float"),
            ("velCtl.thrustN", "float"), ("mocapEst.ageMs", "uint16_t"), ("supervisor.info", "uint16_t")],
    "vel": [("stateEstimate.vx", "float"), ("stateEstimate.vy", "float"), ("stateEstimate.vz", "float"),
            ("stabilizer.roll", "float"), ("stabilizer.pitch", "float")],
    "ref": [("velCtl.vRefX", "float"), ("velCtl.vRefY", "float"), ("velCtl.vRefZ", "float"),
            ("velCtl.rollDes", "float"), ("velCtl.pitchDes", "float"), ("velCtl.yawCmd", "float")],
    "acc": [("velCtl.aCmdX", "float"), ("velCtl.aCmdY", "float"), ("velCtl.aCmdZ", "float"),
            ("stabilizer.yaw", "float"), ("pm.vbat", "float")],
    "mot": [("motor.m1", "uint16_t"), ("motor.m2", "uint16_t"), ("motor.m3", "uint16_t"), ("motor.m4", "uint16_t")],
    "est": [("mocapEst.poses", "uint32_t"), ("mocapEst.drops", "uint32_t"), ("mocapEst.yawOffset", "float")],
}
CF_MS_WRAP = 1 << 24         # log timestamps are 24-bit milliseconds since boot
OFFSET_CREEP = 1e-4          # [s/s] how fast the drone -> host clock offset may grow (crystal drift), never through a delay spike


# #### // Telemetry: log blocks, each sample mapped to host time from the drone's own timestamp // ####
class Telemetry:
    """Latest value + history of every log block; sample_time() is when the drone sampled it, so downlink delay shows."""

    def __init__(self, cf, periods):
        from cflib.crazyflie.log import LogConfig
        self.latest, self.rows, self.stamp, self.sample, self.delay = {}, {k: [] for k in LOG_BLOCKS}, {}, {}, {}
        self.offset = self.offset_t = self.last_ms = None
        self.wraps = 0
        self.lock = threading.Lock()
        self.configs = []
        for name, variables in LOG_BLOCKS.items():
            lc = LogConfig(name=name, period_in_ms=int(periods[name]))
            for v, t in variables:
                lc.add_variable(v, t)
            lc.data_received_cb.add_callback(lambda ts, d, c, name=name: self.on_data(name, ts, d))
            cf.log.add_config(lc)
            self.configs.append(lc)

    def on_data(self, name, ts, d):
        now = time.monotonic()
        with self.lock:
            if self.last_ms is not None and ts < self.last_ms - CF_MS_WRAP // 2:
                self.wraps += 1
            self.last_ms = ts
            cf_s = (ts + self.wraps * CF_MS_WRAP) / 1000.0
            gap = now - cf_s   # drone -> host clock offset + this packet's downlink delay; the smallest is the offset
            if self.offset is None or gap < self.offset:
                self.offset = gap
            else:
                self.offset = min(gap, self.offset + OFFSET_CREEP * (now - self.offset_t))
            self.offset_t = now
            self.latest.update(d)
            self.stamp[name] = now
            self.sample[name] = cf_s + self.offset
            self.delay[name] = now - self.sample[name]
            self.rows[name].append((now, *d.values(), ts))

    def start(self):
        for lc in self.configs:
            lc.start()

    def stop(self):
        for lc in self.configs:
            lc.stop()

    def get(self, key, default=None):
        with self.lock:
            return self.latest.get(key, default)

    def age(self, block="pos") -> float:
        """Since the last packet of the block arrived."""
        with self.lock:
            return time.monotonic() - self.stamp.get(block, -1e9)

    def sample_time(self, block="pos") -> float | None:
        """Host time at which the drone sampled the block's latest values."""
        with self.lock:
            return self.sample.get(block)

    def sample_age(self, block="pos") -> float:
        """How old the block's latest values are, downlink delay included."""
        with self.lock:
            return time.monotonic() - self.sample.get(block, -1e9)

    def latency(self, block="pos") -> float:
        """Downlink delay of the block's last packet beyond the best seen (a radio backlog shows here)."""
        with self.lock:
            return self.delay.get(block, 0.0)


# #### // Connect + firmware params // ####
def connect(scf, timeout: float) -> str | None:
    """open_link + parameter download with a watchdog (a half-open link otherwise blocks forever)."""
    result = {}

    def run():
        try:
            scf.open_link()
            scf.wait_for_params()
            result["ok"] = True
        except Exception as exc:
            result["error"] = str(exc)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if result.get("ok"):
        return None
    if th.is_alive():
        scf.cf.close_link()
        return f"connect timed out after {timeout:.0f} s"
    return result.get("error", "unknown")


def configure(cf, cfg) -> dict:
    """Check the firmware, select the velocity controller, set mass / integrators / firmware params; returns all params."""
    toc = cf.param.toc.toc
    if "velCtl" not in toc or "alpha" not in toc.get("mocapEst", {}):
        raise RuntimeError("drone is not running the mocap_velctl firmware (crazyflie/firmware/flash.py)")
    if int(cf.param.get_value("system.selftestPassed")) != 1:
        raise RuntimeError("drone failed its power-on self-test; power-cycle it lying flat")
    if "yawLim" not in toc["velCtl"]:
        raise RuntimeError("firmware predates velCtl.yawLim; rebuild + flash crazyflie/firmware")
    cf.param.set_value("stabilizer.controller", CONTROLLER_VELCTL)
    cf.param.set_value("velCtl.mass", cfg.drone.mass)
    for k, v in (("kiX", cfg.drone.ki_xy), ("kiY", cfg.drone.ki_xy), ("kiZ", cfg.drone.ki_z)):
        cf.param.set_value(f"velCtl.{k}", v)
    wanted = firmware_params(cfg)
    for name, value in wanted.items():
        group, key = name.split(".", 1)
        if key not in toc.get(group, {}):
            raise RuntimeError(f"unknown parameter {name}")
        cf.param.set_value(name, float(value))
    time.sleep(0.3)
    params = {f"{g}.{k}": cf.param.get_value(f"{g}.{k}")
              for g in ("velCtl", "mocapEst", "pid_attitude", "pid_rate") for k in toc[g]}
    for name, value in wanted.items():
        if abs(float(params[name]) - float(value)) > 1e-4:
            raise RuntimeError(f"{name} reads back {params[name]}, not {value}")
    if wanted:
        print("params: " + ", ".join(f"{name}={params[name]}" for name in wanted))
    if int(cf.param.get_value("stabilizer.controller")) != CONTROLLER_VELCTL:
        raise RuntimeError("could not select the velocity controller")
    return params


def stop_and_disarm(cf) -> None:
    cf.commander.send_stop_setpoint()
    cf.commander.send_notify_setpoint_stop()
    cf.supervisor.send_arming_request(False)


# #### // Mocap -> drone relay: the NatNet thread never waits for the radio; a sender sends the newest pose // ####
class MocapRelay:
    """NatNet frame callback (obstacles, velocity filter, log) + sender thread: newest pose out, superseded ones skipped."""

    def __init__(self, body_id: int, est=None, obstacles=None):
        self.body_id, self.est, self.obstacles = body_id, est, obstacles
        self.cf, self.sent, self.skipped, self.errors = None, 0, 0, 0
        self.rows, self.times, self.sends = [], [], []   # sends: (t, frame, s blocked in cflib)
        self.pending = None
        self.wake = threading.Condition()
        threading.Thread(target=self.send_loop, name="extpose", daemon=True).start()

    def __call__(self, frame):
        if self.obstacles is not None:
            self.obstacles.update(frame)
        b = frame.bodies.get(self.body_id)
        if b is None or not b.valid:
            return
        if self.est is not None:
            self.est.update(frame.number, b.position)
        self.rows.append((frame.arrival, frame.number, *b.position, *b.quat))
        self.times.append(frame.arrival)
        with self.wake:
            if self.pending is not None and self.cf is not None:
                self.skipped += 1
            self.pending = (frame.number, b)
            self.wake.notify()

    def send_loop(self):
        while True:
            with self.wake:
                while self.pending is None:
                    self.wake.wait()
                number, b = self.pending
                self.pending = None
            cf = self.cf
            if cf is None:
                continue
            w, qx, qy, qz = b.quat
            t0 = time.monotonic()
            try:
                cf.extpos.send_extpose(x=b.position[0], y=b.position[1], z=b.position[2], qx=qx, qy=qy, qz=qz, qw=w)
            except Exception:
                self.errors += 1
                continue
            self.sent += 1
            self.sends.append((t0, number, time.monotonic() - t0))

    def position_at(self, t: float):
        """Drone position from the last frame that arrived by host time t (None before the first)."""
        i = bisect.bisect_right(self.times, t)
        return None if i == 0 else self.rows[i - 1][2:5]


def wait_for_body(mocap, body_id: int, timeout: float) -> bool:
    t0 = time.monotonic()
    while mocap.body(body_id)[0] is None:
        if time.monotonic() - t0 > timeout:
            print(f"ABORT: rigid body {body_id} not tracked ({mocap.frames} frames received)", file=sys.stderr)
            return False
        time.sleep(0.02)
    return True


# #### // Radio link (reopened after a restart over the radio) // ####
class DroneLink:
    """Radio link + configured firmware params + telemetry + link statistics; the relay sends extpose while it is open."""

    def __init__(self, cfg, relay: MocapRelay, periods):
        self.cfg, self.relay, self.periods = cfg, relay, periods
        self.scf = self.cf = self.tel = None
        self.params, self.stats = {}, []   # stats: (t, name, value) from cflib link_statistics

    def open(self) -> bool:
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        uri = self.cfg.radio.uri
        self.scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache=str(repo_path(self.cfg.paths.log_dir) / "cache")))
        err = connect(self.scf, timeout=self.cfg.radio.connect_timeout)
        if err:
            print(f"ABORT: no radio link to {uri} ({err}); is the drone on?", file=sys.stderr)
            self.scf = None
            return False
        self.cf = self.scf.cf
        try:
            self.params = configure(self.cf, self.cfg)
        except RuntimeError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            self.close()
            return False
        self.tel = Telemetry(self.cf, self.periods)
        self.tel.start()
        ls = self.cf.link_statistics
        for name in ("link_quality", "uplink_rssi", "uplink_rate", "downlink_rate", "uplink_congestion", "downlink_congestion"):
            getattr(ls, f"{name}_updated").add_callback(lambda v, name=name: self.stats.append((time.monotonic(), name, v)))
        ls.latency_updated.add_callback(lambda v: self.stats.append((time.monotonic(), "latency_p95_ms", v)))
        self.relay.cf = self.cf
        time.sleep(1.5)
        return True

    def ok(self) -> bool:
        return self.tel is not None and self.tel.age("pos") < 1.0

    def needs_restart(self) -> bool:
        return self.tel is not None and bool(int(self.tel.get("supervisor.info", 0)) & (SUP_LOCKED | SUP_CRASHED))

    def close(self) -> None:
        self.relay.cf = None
        if self.tel:
            self.tel.stop()
            self.tel = None
        if self.scf:
            self.scf.close_link()
            self.scf = None

    def reboot(self) -> bool:
        from cflib.utils.power_switch import PowerSwitch
        self.close()
        try:
            PowerSwitch(self.cfg.radio.uri).stm_power_cycle()
        except Exception as exc:
            print(f"  radio restart failed ({exc})", file=sys.stderr)
        time.sleep(4.0)
        return self.open()
