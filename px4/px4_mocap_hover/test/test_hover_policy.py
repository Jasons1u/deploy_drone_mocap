from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from px4_mocap_hover import frames  # noqa: E402
from px4_mocap_hover.hover_policy import DroneState, HoverPolicy  # noqa: E402

POLICY_DIR = Path(__file__).resolve().parent.parent / "policy"
POLICIES = {"hover.onnx": (36, 16), "new_hover.onnx": (35, 15)}


@pytest.fixture(params=sorted(POLICIES))
def policy(request):
    return HoverPolicy(str(POLICY_DIR / request.param)), POLICIES[request.param]


def at_rest(pos=(0.3, -0.4, 1.2), yaw=0.7):
    return DroneState(np.array(pos), np.zeros(3), frames.quat_from_yaw(yaw), np.zeros(3))


def test_layout_from_metadata(policy):
    pol, (obs_dim, state_dim) = policy
    assert (pol.obs_dim, pol.state_dim, pol.act_hist_len) == (obs_dim, state_dim, 5)
    assert pol.build_obs(at_rest(), [0, 0, 1], 0.0).shape == (obs_dim,)


def test_obs_is_zero_at_target(policy):
    """Sitting still exactly on the target pose: every setpoint-frame term is zero."""
    pol, _ = policy
    obs = pol.build_obs(at_rest(), [0.3, -0.4, 1.2], 0.7)
    assert obs[:15] == pytest.approx(np.zeros(15), abs=1e-6)


def test_obs_is_in_the_setpoint_frame(policy):
    """1 m along world +Y with target yaw +90 deg is 1 m along the setpoint frame's +X."""
    pol, _ = policy
    obs = pol.build_obs(at_rest(pos=(0, 1, 1), yaw=np.pi / 2), [0, 0, 1], np.pi / 2)
    assert obs[:3] == pytest.approx([1, 0, 0], abs=1e-6)
    assert obs[6:9] == pytest.approx([0, 0, 0], abs=1e-6)


def test_ground_prox_only_on_16_dim(policy):
    pol, (_, state_dim) = policy
    low = pol.build_obs(at_rest(pos=(0, 0, 0.1)), [0, 0, 1], 0.0)
    if state_dim == 16:
        assert low[15] == pytest.approx(1.0)          # at or below gp_z0 saturates to 1
        high = pol.build_obs(at_rest(pos=(0, 0, 1.4)), [0, 0, 1], 0.0)
        assert high[15] == pytest.approx(1 / (1 + (1.4 - pol.gp_z0) / pol.gp_h0))


def test_action_history_rolls_and_resets(policy):
    pol, (_, state_dim) = policy
    s = at_rest()
    outs = [pol.act(pol.build_obs(s, [0, 0, 1], 0.0)).raw for _ in range(6)]
    hist = pol.build_obs(s, [0, 0, 1], 0.0)[state_dim:].reshape(5, 4)
    assert hist == pytest.approx(np.stack(outs[-5:]), abs=1e-6)   # oldest -> newest
    pol.reset()
    assert np.all(pol.build_obs(s, [0, 0, 1], 0.0)[state_dim:] == 0)


def test_motor_map(policy):
    pol, _ = policy
    assert pol.motor_map(np.zeros(4)) == pytest.approx(pol.hover_cmd)
    assert pol.motor_map(np.full(4, 5.0)) == pytest.approx(np.ones(4))            # clipped to 1
    assert pol.motor_map(np.full(4, -1.0)) == pytest.approx(np.full(4, pol.min_motor_cmd))
    half = pol.motor_map(np.full(4, 0.5))
    assert half == pytest.approx(pol.hover_cmd + 0.5 * (1 - pol.hover_cmd))


def test_near_hover_output_is_sane(policy):
    """At rest on target the trained policy should command roughly hover thrust."""
    pol, _ = policy
    s = at_rest()
    for _ in range(20):
        out = pol.act(pol.build_obs(s, s.pos_enu, 0.7))
    assert np.all(np.isfinite(out.raw))
    assert np.all(np.abs(out.motor_cmd - pol.hover_cmd) < 0.25)
