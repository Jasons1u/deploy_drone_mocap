"""The mocap-state controller, flown in a simple simulation before it flies for real.

The sim is deliberately worse than the real drone: 30 ms command latency (mocap + radio; the logs
show ~10 ms telemetry lag), 80 ms attitude lag, and a true hover thrust 7% off the config's guess.
"""

import collections
import math
from dataclasses import dataclass

import pytest

from cf_mocap.config import load_config
from cf_mocap.flight_plan import HoverPlan, Phase, Setpoint
from cf_mocap.mocap_control import G, MocapPositionController
from mocap_natnet.frames import quat_from_rpy

DT = 0.01
YAW = math.radians(78.0)
TRUE_HOVER_THRUST = 1.07 * load_config().host_control.hover_thrust


@dataclass
class P:   # the bits of mocap_natnet.Pose the controller reads
    position: tuple
    velocity: tuple
    quat: tuple


class Sim:
    """Point mass; tilt follows the command with a first-order lag; thrust along body z."""

    def __init__(self, pos, hover_thrust=TRUE_HOVER_THRUST, latency_steps=3, tau=0.08):
        self.hover_thrust = hover_thrust
        self.p, self.v = list(pos), [0.0, 0.0, 0.0]
        self.roll = self.pitch = 0.0
        self.queue = collections.deque([None] * latency_steps)
        self.tau = tau
        self.ground = pos[2]
        self.max_tilt = 0.0

    def pose(self):
        return P(tuple(self.p), tuple(self.v), quat_from_rpy(self.roll, self.pitch, YAW))

    def step(self, cmd):
        self.queue.append(cmd)
        cmd = self.queue.popleft()
        if cmd is None:
            return
        a = DT / self.tau
        self.roll += a * (math.radians(cmd.roll_deg) - self.roll)
        self.pitch += a * (math.radians(cmd.pitch_deg) - self.pitch)
        self.max_tilt = max(self.max_tilt, math.degrees(math.hypot(self.roll, self.pitch)))
        f = G * cmd.thrust / self.hover_thrust                    # specific thrust [m/s^2]
        # body z in world for ZYX (roll, pitch, yaw):
        cr, sr, cp, sp, cy, sy = (math.cos(self.roll), math.sin(self.roll), math.cos(self.pitch),
                                  math.sin(self.pitch), math.cos(YAW), math.sin(YAW))
        zb = (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr)
        acc = [f * zb[0], f * zb[1], f * zb[2] - G]
        for i in range(3):
            self.v[i] += acc[i] * DT
            self.p[i] += self.v[i] * DT
        if self.p[2] <= self.ground:           # on the ground: no falling through, no sliding
            self.p[2], self.v = self.ground, [0.0, 0.0, 0.0]


def fly(plan, sim, controller, seconds, push=None):
    worst_hover_err = 0.0
    t = 0.0
    plan.start(0.0)
    while t < seconds:
        phase, sp = plan.step(t)
        if push and abs(t - push[0]) < DT / 2:
            sim.p[0] += push[1]
        cmd = controller.step(t, sp, sim.pose())
        assert all(math.isfinite(v) for v in (cmd.roll_deg, cmd.pitch_deg, cmd.yawrate_deg_s))
        sim.step(cmd)
        if phase is Phase.HOVER and (push is None or t < push[0]):
            worst_hover_err = max(worst_hover_err, math.dist(sim.p, (sp.x, sp.y, sp.z)))
        t += DT
    return worst_hover_err


@pytest.fixture
def cfg():
    return load_config()


def make(cfg, start=(0.3, -0.2, 0.03), true_hover_thrust=TRUE_HOVER_THRUST):
    plan = HoverPlan(start, math.degrees(YAW), 0.4, 2.5, 2.5, 0.0)
    return (plan, Sim(start, true_hover_thrust),
            MocapPositionController(cfg.host_control, ground_z=start[2]))


def test_takeoff_and_hover_with_wrong_hover_thrust(cfg):
    plan, sim, ctrl = make(cfg)
    fly(plan, sim, ctrl, 6.0)
    # after takeoff + a few seconds of hover the z integral has found the real hover thrust
    assert math.dist(sim.p, (0.3, -0.2, 0.43)) < 0.02
    assert sim.max_tilt < cfg.host_control.max_tilt_deg + 1.0


def test_survives_a_25_percent_hover_thrust_error(cfg):
    """The 2026-09-28 flight logs lifted off at ~35500 PWM while the old guess was 44000."""
    plan, sim, ctrl = make(cfg, true_hover_thrust=0.8 * cfg.host_control.hover_thrust)
    fly(plan, sim, ctrl, 10.0)
    assert math.dist(sim.p, (0.3, -0.2, 0.43)) < 0.02
    assert sim.max_tilt < cfg.host_control.max_tilt_deg + 1.0


def test_recovers_from_a_20cm_push(cfg):
    plan, sim, ctrl = make(cfg)
    fly(plan, sim, ctrl, 12.0, push=(8.0, 0.2))
    assert math.dist(sim.p, (0.3, -0.2, 0.43)) < 0.02
    assert sim.max_tilt < cfg.host_control.max_tilt_deg + 1.0


def test_lands(cfg):
    plan, sim, ctrl = make(cfg)
    t, landed = 0.0, False
    plan.start(0.0)
    while t < 12.0:
        if abs(t - 6.0) < DT / 2:
            plan.land(t)
        phase, sp = plan.step(t)
        if phase is Phase.DONE:
            landed = True
            break
        sim.step(ctrl.step(t, sp, sim.pose()))
        t += DT
    assert landed and sim.p[2] < 0.05


# --- sign conventions (checked against crtp_commander_rpyt.c + cflib send_setpoint) ---

def still(pos):
    return P(pos, (0.0, 0.0, 0.0), quat_from_rpy(0.0, 0.0, 0.0))    # yaw 0: body x = world x


def cmd_for(cfg, sp, pos):
    return MocapPositionController(cfg.host_control, ground_z=-1.0).step(0.0, sp, still(pos))


def test_target_ahead_pitches_nose_down(cfg):
    c = cmd_for(cfg, Setpoint(1.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert c.pitch_deg > 0 and abs(c.roll_deg) < 1e-9


def test_target_to_the_left_rolls_left_side_down(cfg):
    c = cmd_for(cfg, Setpoint(0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert c.roll_deg < 0 and abs(c.pitch_deg) < 1e-9


def test_target_above_adds_thrust(cfg):
    up = cmd_for(cfg, Setpoint(0.0, 0.0, 1.5, 0.0), (0.0, 0.0, 1.0))
    level = cmd_for(cfg, Setpoint(0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert up.thrust > level.thrust == cfg.host_control.hover_thrust


def test_yaw_target_ccw_gives_positive_ccw_rate(cfg):
    c = cmd_for(cfg, Setpoint(0.0, 0.0, 1.0, 30.0), (0.0, 0.0, 1.0))
    assert c.yawrate_deg_s > 0
