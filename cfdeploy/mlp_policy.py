"""Numpy forward pass of an exported rsl_rl actor: (x - mean) / (std + eps) -> [Linear, ELU]* -> Linear (deterministic mean)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cfdeploy.safe_cf_obs import SafeCFObs, history_clip_note, obs_layout


class MlpPolicy:
    def __init__(self, path: str | Path):
        d = np.load(path, allow_pickle=False)
        self.mean, self.std, self.eps = d["obs_mean"], d["obs_std"], float(d["obs_eps"])
        n = int(d["n_layers"])
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
        self.obs_dim = self.layers[0][0].shape[1]
        self.act_dim = self.layers[-1][0].shape[0]
        self.meta = {k[5:]: d[k] for k in d.files if k.startswith("meta_")}
        c = float(self.meta["clip_actions"]) if "clip_actions" in self.meta else float("nan")
        self.clip_actions = None if np.isnan(c) else c   # training's RslRlVecEnvWrapper clip: env and action history saw clipped actions
        v = float(self.meta["obstacle_vertical_motion"]) if "obstacle_vertical_motion" in self.meta else float("nan")
        self.obstacle_vertical_motion = None if np.isnan(v) else bool(v)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """obs (obs_dim,) or (N, obs_dim) -> action mean, same leading shape, unclipped."""
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / (self.std + self.eps)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU
        return x

    def max_obstacle_speed(self, default: float = 1.6) -> float:
        """Fastest obstacle speed in training (obstacle_speed_range metadata), else default."""
        try:
            return float(str(self.meta["obstacle_speed_range"]).strip("()[] ").split(",")[1])
        except (KeyError, IndexError, ValueError):
            return default


def obs_for(policy) -> SafeCFObs:
    """Observation builder matching a policy: input size -> layout, clip_actions -> history clip (fakes default to 26 inputs)."""
    return SafeCFObs(**obs_layout(getattr(policy, "obs_dim", 26)), history_clip=getattr(policy, "clip_actions", None))


def load_policy(path) -> tuple:
    """Policy file -> (policy, its observation builder, note on the action-history clip or None)."""
    policy = MlpPolicy(path)
    return policy, obs_for(policy), history_clip_note(policy.clip_actions)
