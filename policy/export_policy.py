#!/usr/bin/env python3
"""Export an rsl_rl ActorCritic checkpoint's actor + obs normalizer to .npz (with training conventions) and verify it."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.mlp_policy import MlpPolicy  # noqa: E402


# #### // Checkpoint -> npz (actor weights, normalizer, training metadata) -> numpy vs torch check // ####
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, epilog="e.g. policy/export_policy.py <run>/model_1499.pt policy/models/<name>.npz "
                                 "--critic-dim 100 --task Isaac_SafeCF_SAFEPPO_CF21BL")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--critic-dim", type=int, required=True, help="critic obs dim of the training env (to rebuild the module)")
    ap.add_argument("--task", default="", help="gym task id, stored as metadata")
    args = ap.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    obs_dim, act_dim = sd[f"actor.{lin[0]}.weight"].shape[1], sd[f"actor.{lin[-1]}.weight"].shape[0]
    hidden = [sd[f"actor.{i}.weight"].shape[0] for i in lin[:-1]]

    dummy = TensorDict({"policy": torch.zeros(1, obs_dim), "critic": torch.zeros(1, args.critic_dim)}, batch_size=[1])
    model = ActorCritic(dummy, {"policy": ["policy"], "critic": ["critic"]}, act_dim, actor_obs_normalization=True,
                        critic_obs_normalization=True, actor_hidden_dims=hidden, critic_hidden_dims=hidden, activation="elu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    norm = model.actor_obs_normalizer
    arrays = {"obs_mean": norm._mean.squeeze(0).numpy().astype(np.float64),
              "obs_std": norm._std.squeeze(0).numpy().astype(np.float64),
              "obs_eps": np.float64(norm.eps), "n_layers": np.int64(len(lin))}
    for j, i in enumerate(lin):
        arrays[f"w{j}"] = sd[f"actor.{i}.weight"].numpy().astype(np.float64)
        arrays[f"b{j}"] = sd[f"actor.{i}.bias"].numpy().astype(np.float64)
    arrays["meta_checkpoint"] = np.array(str(args.checkpoint.resolve()))
    arrays["meta_task"] = np.array(args.task)
    arrays["meta_iter"] = np.int64(ck.get("iter", -1))
    agent_yaml = args.checkpoint.resolve().parent / "params" / "agent.yaml"
    clip = None
    if agent_yaml.is_file():
        import yaml
        clip = (yaml.unsafe_load(agent_yaml.read_text()) or {}).get("clip_actions")
    else:
        print(f"warning: {agent_yaml} not found; assuming clip_actions null (raw action history)", file=sys.stderr)
    arrays["meta_clip_actions"] = np.float64(clip if clip is not None else np.nan)
    env_yaml = agent_yaml.with_name("env.yaml")
    vertical = np.nan
    if env_yaml.is_file():
        import yaml
        hc = ((yaml.unsafe_load(env_yaml.read_text()) or {}).get("commands") or {}).get("hover_clf") or {}
        # pre-attack runs drifted in the plane; runs with attacks but no key are after the key was dropped (always 3-D)
        vertical = float(bool(hc.get("obstacle_vertical_motion", "attack_prob" in hc)))
        for key in ("obstacle_speed_range", "obstacle_turn_interval", "attack_prob", "attack_speed_range"):
            if key in hc:
                arrays[f"meta_{key}"] = np.array(str(hc[key]))
    else:
        print(f"warning: {env_yaml} not found; obstacle vertical motion unknown", file=sys.stderr)
    arrays["meta_obstacle_vertical_motion"] = np.float64(vertical)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)

    pol = MlpPolicy(args.out)
    rng = np.random.default_rng(0)
    m, s = arrays["obs_mean"], arrays["obs_std"]
    obs = np.concatenate([rng.normal(m, s, (2000, obs_dim)), rng.normal(m, 3 * s + 1.0, (2000, obs_dim))])
    with torch.no_grad():
        ref = model.act_inference(TensorDict({"policy": torch.tensor(obs, dtype=torch.float32)}, batch_size=[len(obs)])).numpy()
    out = pol(obs)
    err = np.abs(out - ref).max()
    print(f"clip_actions {clip} (the action history in training saw {'actions clipped to +-' + str(clip) if clip else 'raw actions'}); "
          f"obstacle vertical motion in training: {'unknown' if np.isnan(vertical) else bool(vertical)}")
    print(f"exported {args.checkpoint.name} (iter {arrays['meta_iter']}): obs {obs_dim} -> {hidden} -> {act_dim}, "
          f"normalizer eps {arrays['obs_eps']}; numpy vs act_inference max |diff| {err:.2e} over {len(obs)} obs")
    if err > 1e-4:
        print("FAIL: exported policy does not match", file=sys.stderr)
        return 1
    print(f"PASS -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
