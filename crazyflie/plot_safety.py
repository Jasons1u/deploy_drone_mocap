#!/usr/bin/env python3
"""Safety plots of policy flights: drone and obstacle paths, clearances, the training safety cost r1 and the SafePPO cost critic."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cfdeploy
from cfdeploy.config import load  # noqa: E402

COLORS = {"drone": "#1f4e9c", 25: "#d9661f", 26: "#2a9d4b"}
MODE_SHADE = {"policy": ("#cfe3f7", "policy"), "manual": ("#e6e6e6", "manual"), "launch": ("#f3e2c7", "takeoff / landing"),
              "climb": ("#f3e2c7", None), "land": ("#f3e2c7", None)}
TIME_LEFT = 1.0   # cost-critic input "episode time left" (hardware has no episode clock; 1 = the full 10 s ahead)


# #### // Training constants (the run's env.yaml when it is found, else the values this policy was trained with) // ####
def train_constants(env_yaml: Path | None) -> dict:
    c = dict(beta=0.05, k=6.0, k_arena=10.0, horizon=0.5, floor_height=0.1, arena_radius=6.0 * 0.3048, arena_height=9.5 * 0.3048,
             drone_radius=0.05, unsafe_margin=0.05, boundary_margin=0.10, clearance_range=(-0.3, 2.0), ground_z0=0.4, ground_h0=0.5)
    if env_yaml is not None and env_yaml.is_file():
        import yaml
        e = yaml.unsafe_load(env_yaml.read_text())
        hc = e["commands"]["hover_clf"]
        c.update(beta=hc["r1_beta"], k=hc["r1_k"], k_arena=hc["r1_k_arena"], horizon=hc["r1_predict_horizon"], floor_height=hc["floor_height"],
                 arena_radius=hc["arena_radius"], arena_height=hc["arena_height"], drone_radius=hc["drone_radius"],
                 unsafe_margin=hc["unsafe_margin"], boundary_margin=hc["boundary_unsafe_margin"])
        gp = e["observations"]["critic"]["ground_prox"]["params"]
        c.update(ground_z0=gp["z0"], ground_h0=gp["h0"])
    return c


# #### // Log loading // ####
def read_meta(run: Path) -> dict:
    out = {}
    for line in (run / "meta.txt").read_text().splitlines():
        key, _, val = line.partition(": ")
        if key in ("config", "policy_meta"):
            out[key] = ast.literal_eval(val)
    return out


def load_run(run: Path) -> dict | None:
    """Control rows (100 Hz), obstacle tracks resampled onto them (nan while untracked), and the policy inputs."""
    rows = list(csv.reader(open(run / "control.csv")))
    head, rows = rows[0], rows[1:]
    if not any(r[1] == "policy" for r in rows):
        return None
    t = np.array([float(r[0]) for r in rows])
    mode = np.array([r[1] for r in rows])
    col = {h: i for i, h in enumerate(head)}
    num = lambda names: np.array([[float(r[col[n]]) if len(r) > col[n] else math.nan for n in names] for r in rows])
    d = dict(t=t, mode=mode, p=num(["x", "y", "z"]), v=num(["vx_est", "vy_est", "vz_est"]), target=num(["tx", "ty", "tz"]),
             cmd=num(["cmd_vx", "cmd_vy", "cmd_vz"]), act=num(["a0", "a1", "a2"]), obs=num([f"obs{i}" for i in range(33)]))
    meta = read_meta(run)
    radius = {int(o["id"]): float(o["radius"]) for o in meta["config"]["obstacles"]["tracked"]}
    raw = [r for r in csv.reader(open(run / "obstacles.csv"))][1:]
    d["obstacles"] = {}
    for oid in sorted(radius):
        a = np.array([[float(x) for x in (r[0], *r[3:9])] for r in raw if int(r[2]) == oid])
        idx = np.clip(np.searchsorted(a[:, 0], t), 1, len(a) - 1)
        age = t - a[idx - 1, 0]
        pos = np.stack([np.interp(t, a[:, 0], a[:, k]) for k in (1, 2, 3)], 1)
        vel = np.stack([np.interp(t, a[:, 0], a[:, k]) for k in (4, 5, 6)], 1)
        gap = np.r_[np.diff(a[:, 0]), 0.0][idx - 1] > 0.1   # inside a tracking dropout
        bad = gap | (t < a[0, 0]) | (age > 0.1)
        pos[bad], vel[bad] = math.nan, math.nan
        d["obstacles"][oid] = dict(pos=pos, vel=vel, radius=radius[oid])
    d["meta"], d["t0"] = meta, t[np.argmax(mode == "launch")] if (mode == "launch").any() else t[0]
    return d


# #### // Safety terms, as mdp/obstacle.py computes them in training // ####
def safety_terms(d: dict, c: dict) -> dict:
    """Per row: clearance per obstacle, arena clearances, predicted obstacle clearance, safety clearance and r1."""
    p, v = d["p"], d["v"]
    clear, pred = {}, []
    for oid, o in d["obstacles"].items():
        unsafe = o["radius"] + c["drone_radius"] + c["unsafe_margin"]
        rel, w = o["pos"] - p, o["vel"] - v
        clear[oid] = np.linalg.norm(rel, axis=1) - unsafe
        tc = np.clip(-(rel * w).sum(1) / np.maximum((w * w).sum(1), 1e-9), 0.0, c["horizon"])
        pred.append(np.linalg.norm(rel + w * tc[:, None], axis=1) - unsafe)
    m = c["boundary_margin"]
    arena = np.stack([p[:, 2] - (c["floor_height"] + m), (c["arena_height"] - c["drone_radius"] - m) - p[:, 2],
                      (c["arena_radius"] - c["drone_radius"] - m) - np.hypot(p[:, 0], p[:, 1])], 1)
    c_obs = np.fmin.reduce(list(clear.values()))
    c_pred = np.clip(np.fmin.reduce(pred), 0.0, None) if c["horizon"] > 0 else c_obs
    c_arena = arena.min(1)
    safety = np.fmin(c_obs, c_arena)
    prox = np.fmax(np.exp(-c["k"] * c_pred), np.exp(-c["k_arena"] * c_arena))
    r1 = np.where(safety >= 0.0, c["beta"] * (prox - 1.0), 1.0 - c["beta"])
    r1[np.isnan(safety)] = math.nan
    return dict(clear=clear, arena=arena, c_arena=c_arena, c_pred=c_pred, safety=safety, r1=r1)


# #### // Cost critic V1 on the 100 privileged inputs (unknowable ones at their training values) // ####
def find_checkpoint(path: str | None, meta: dict) -> Path | None:
    ck = Path(path or meta.get("policy_meta", {}).get("checkpoint", ""))
    if ck.is_file():
        return ck
    if ck.name:   # runs get moved between experiment folders
        hits = sorted(ck.parents[2].glob(f"*/{ck.parent.name}/{ck.name}"))
        return hits[0] if hits else None
    return None


def load_cost_critic(ck: Path):
    import torch
    from rsl_rl.networks import MLP, EmpiricalNormalization
    sd = torch.load(ck, map_location="cpu", weights_only=False)["cost_critic_state_dict"]
    n_in = sd["critic.0.weight"].shape[1]
    net = torch.nn.Module()
    net.critic = MLP(n_in, 1, [sd[f"critic.{i}.weight"].shape[0] for i in (0, 2, 4)], "elu")
    net.critic_obs_normalizer = EmpiricalNormalization(n_in)
    net.load_state_dict(sd)
    net.eval()
    return lambda x: net.critic(net.critic_obs_normalizer(torch.as_tensor(x, dtype=torch.float32))).detach().numpy()[:, 0]


def rom_nominal(cfg) -> tuple:
    """(9 actuation inputs, ROM for applied_mu) from the sysID fit, delays rounded to physics steps like training."""
    n, kv, dt = cfg.drone_model.nominal, np.array(cfg.drone_model.kv, float), cfg.training.physics_dt
    step = lambda s: round(s / dt) * dt
    act = [kv[1] * n.kv_scale, kv[2] * n.kv_scale, 1 / n.tau_xy, 1 / n.tau_z, n.a_xy, n.drag_z, step(n.d_cmd_xy), step(n.d_cmd_z), step(n.d_fb)]
    return np.array(act), dict(kv=kv * n.kv_scale, tau=np.array([n.tau_xy, n.tau_xy, n.tau_z]), a_xy=n.a_xy,
                               d_cmd=np.array([n.d_cmd_xy, n.d_cmd_xy, n.d_cmd_z]), d_fb=n.d_fb, a_up=cfg.drone_model.a_z_up,
                               a_down=cfg.drone_model.a_z_down)


def applied_acc(d: dict, rom: dict, seg: np.ndarray) -> np.ndarray:
    """The ROM's applied acceleration over one policy segment, driven by the logged commands and measured velocity."""
    t, cmd, v = d["t"][seg], d["cmd"][seg], d["v"][seg]
    a, out = np.zeros(3), np.zeros((len(t), 3))
    for k in range(len(t)):
        c = np.array([np.interp(t[k] - rom["d_cmd"][i], t, cmd[:, i]) for i in range(3)])
        vf = np.array([np.interp(t[k] - rom["d_fb"], t, v[:, i]) for i in range(3)])
        ac = rom["kv"] * (c - vf)
        h = math.hypot(ac[0], ac[1])
        ac[:2] *= min(1.0, rom["a_xy"] / max(h, 1e-6))
        ac[2] = min(max(ac[2], -min(rom["a_down"], 9.81)), rom["a_up"])
        dt = t[k] - t[k - 1] if k else 0.01
        a = a + (ac - a) * (1.0 - np.exp(-dt / rom["tau"]))
        out[k] = a
    return out


def critic_inputs(d: dict, c: dict, cfg, t_rem: float, clip: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(row indices, inputs) for every policy row, in the run's CriticCfg term order."""
    act9, rom = rom_nominal(cfg)
    idx, X = [], []
    pol = d["mode"] == "policy"
    starts = np.flatnonzero(pol & ~np.r_[False, pol[:-1]])
    for s in starts:
        e = s
        while e + 1 < len(pol) and pol[e + 1]:
            e += 1
        seg = np.arange(s, e + 1)
        acc = applied_acc(d, rom, seg)
        hist = np.repeat(d["obs"][s, 7:10][None], 8, 0)   # the runner fills the history with its engage command
        for j, k in enumerate(seg):
            if j:
                hist = np.roll(hist, -1, 0)
                hist[-1] = np.clip(d["act"][k - 1], -clip, clip)
            o, p = d["obs"][k], d["p"][k]
            obst = [(oid, ob["pos"][k], ob["radius"]) for oid, ob in d["obstacles"].items() if not np.isnan(ob["pos"][k, 0])]
            if not obst:
                continue
            cl = [np.linalg.norm(q - p) - (r + c["drone_radius"] + c["unsafe_margin"]) for _, q, r in obst]
            order = np.argsort(cl, kind="stable")
            _, q, r = obst[order[0]]
            rel = q - p
            dist = np.linalg.norm(rel)
            bvec = rel / max(dist, 1e-6) * (dist - (r + c["drone_radius"] + c["unsafe_margin"]))
            ground = 1.0 / (1.0 + max(o[6] - c["ground_z0"], 0.0) / c["ground_h0"])
            slots = []
            for i in range(4):
                if i < 2:
                    geom, vel = o[19 + 4 * i: 23 + 4 * i], o[27 + 3 * i: 30 + 3 * i]
                    slots += [*geom, obst[order[i]][2] if i < len(obst) else 0.0, *vel]
                else:
                    slots += [0.0, 0.0, 0.0, c["clearance_range"][1], 0.0, 0.0, 0.0, 0.0]
            x = np.concatenate([o[0:3], o[3:6], np.zeros(3), np.zeros(3), np.zeros(3), [ground], hist.reshape(-1), bvec,
                                [np.linalg.norm(bvec)], [t_rem], [r], o[3:6], acc[j], [rom["a_up"]], np.zeros(3), act9, o[16:19], slots])
            idx.append(k)
            X.append(x)
    return np.array(idx), np.array(X)


# #### // Figures // ####
def shade_modes(ax, tt, mode) -> None:
    edges = np.flatnonzero(np.r_[True, mode[1:] != mode[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        color, _ = MODE_SHADE.get(mode[a], ("white", None))
        ax.axvspan(tt[a], tt[min(b, len(tt) - 1)], color=color, lw=0, zorder=0)


def draw_maps(ax_xy, ax_xz, d, s, c, k_min, window=None, zoom=None) -> None:
    from matplotlib.patches import Circle
    tt = d["t"] - d["t0"]
    sel = np.ones(len(tt), bool) if window is None else (tt >= window[0]) & (tt <= window[1])
    air = sel & np.isin(d["mode"], ["manual", "policy"])
    R, Rw = c["arena_radius"], c["arena_radius"] - c["drone_radius"] - c["boundary_margin"]
    ax_xy.add_patch(Circle((0, 0), R, fill=False, color="k", lw=1.2))
    ax_xy.add_patch(Circle((0, 0), Rw, fill=False, color="k", lw=0.8, ls="--"))
    ax_xy.add_patch(Circle((0, 0), 1.4, fill=False, color="#888", lw=0.8, ls=":"))
    ax_xz.axhline(c["floor_height"] + c["boundary_margin"], color="k", lw=0.8, ls="--")
    ax_xz.axhline(c["arena_height"] - c["drone_radius"] - c["boundary_margin"], color="k", lw=0.8, ls="--")
    ax_xz.axvline(-Rw, color="k", lw=0.8, ls="--")
    ax_xz.axvline(Rw, color="k", lw=0.8, ls="--")
    for oid, o in d["obstacles"].items():
        for ax, j in ((ax_xy, 1), (ax_xz, 2)):
            ax.plot(o["pos"][air, 0], o["pos"][air, j], color=COLORS.get(oid, "#a33"), lw=0.8, alpha=0.55, label=f"obstacle {oid}")
    for m, style in (("manual", dict(color="#777", lw=0.8, alpha=0.7)), ("policy", dict(color=COLORS["drone"], lw=1.2))):
        pm = d["p"].copy()
        pm[~(air & (d["mode"] == m))] = math.nan
        for ax, j in ((ax_xy, 1), (ax_xz, 2)):
            ax.plot(pm[:, 0], pm[:, j], label=f"drone ({m})", **style)
    tg = d["target"][air & (d["mode"] == "policy")]
    tg = np.unique(np.round(tg[~np.isnan(tg[:, 0])], 3), axis=0)
    for ax, j in ((ax_xy, 1), (ax_xz, 2)):
        ax.plot(tg[:, 0], tg[:, j], "*", color="#c9a400", ms=13, mec="k", mew=0.6, label="target", zorder=5)
    if k_min is not None:   # the closest approach under the policy: spheres at that instant
        p = d["p"][k_min]
        for ax, j in ((ax_xy, 1), (ax_xz, 2)):
            ax.add_patch(Circle((p[0], p[j]), c["drone_radius"], color=COLORS["drone"], zorder=6))
            for oid, o in d["obstacles"].items():
                q = o["pos"][k_min]
                if np.isnan(q[0]):
                    continue
                col = COLORS.get(oid, "#a33")
                ax.add_patch(Circle((q[0], q[j]), o["radius"], color=col, alpha=0.9, zorder=6))
                ax.add_patch(Circle((q[0], q[j]), o["radius"] + c["drone_radius"] + c["unsafe_margin"], fill=False, color=col, ls="--", zorder=6))
    for ax, ylabel in ((ax_xy, "y [m]"), (ax_xz, "z [m]")):
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    if zoom is None:
        ax_xy.set_xlim(-R - 0.1, R + 0.1)
        ax_xy.set_ylim(-R - 0.1, R + 0.1)
        ax_xz.set_xlim(-R - 0.1, R + 0.1)
        ax_xz.set_ylim(0, c["arena_height"] + 0.05)
    else:   # half-width [m] around the drone at k_min
        p = d["p"][k_min]
        for ax, j in ((ax_xy, 1), (ax_xz, 2)):
            ax.set_xlim(p[0] - zoom, p[0] + zoom)
            ax.set_ylim(p[j] - zoom, p[j] + zoom)


def draw_series(axes, d, s, vc, c, window=None) -> None:
    tt = d["t"] - d["t0"]
    air = np.isin(d["mode"], ["manual", "policy"])
    ax_c, ax_v, ax_r, ax_k = axes
    for ax in axes:
        shade_modes(ax, tt, d["mode"])
        ax.grid(alpha=0.3)
    for oid, cl in s["clear"].items():
        ax_c.plot(tt, cl, color=COLORS.get(oid, "#a33"), lw=1.0, label=f"obstacle {oid}")
    ca = s["c_arena"].copy()
    ca[~air] = math.nan
    ax_c.plot(tt, ca, color="k", lw=0.8, alpha=0.6, label="arena (floor / ceiling / wall)")
    ax_c.axhline(0.0, color="#c00", lw=1.0, ls="--", label="unsafe boundary")
    ax_c.axhline(-c["unsafe_margin"], color="#c00", lw=1.4, label="collision")
    ax_c.set_ylim(-0.12, 1.6)
    ax_c.set_ylabel("clearance [m]")
    ax_v.plot(tt, np.linalg.norm(d["v"], axis=1), color=COLORS["drone"], lw=1.0, label="drone")
    for oid, o in d["obstacles"].items():
        ax_v.plot(tt, np.linalg.norm(o["vel"], axis=1), color=COLORS.get(oid, "#a33"), lw=0.8, label=f"obstacle {oid}")
    ax_v.set_ylabel("speed [m/s]")
    ax_v.set_ylim(0, None)
    r1 = s["r1"].copy()
    r1[~air] = math.nan
    unsafe = r1 > 0
    ax_r.plot(tt, np.where(unsafe, math.nan, r1), color="#7b2d8e", lw=1.0, label="r1 = beta (exp(-k c) - 1) while safe")
    edges = np.flatnonzero(np.diff(np.r_[0, unsafe.astype(int), 0]))
    for j, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
        ax_r.axvspan(tt[a], tt[min(b, len(tt) - 1)] + 0.02, color="#d00", lw=0, alpha=0.8,
                     label=f"unsafe: r1 = 1 - beta = {1 - c['beta']:.2f} (off scale)" if j == 0 else None)
    ax_r.set_ylim(-c["beta"] * 1.1, c["beta"] * 0.15)
    ax_r.axhline(0.0, color="k", lw=0.6)
    ax_r.set_ylabel("safety cost r1")
    if vc is not None:
        idx, v = vc
        line = np.full(len(tt), math.nan)
        line[idx] = v
        ax_k.plot(tt, line, color="#9a6b00", lw=1.0, label=f"cost critic V1 (episode time left {TIME_LEFT:g})")
        ax_k.axhline(0.0, color="k", lw=0.6)
    else:
        ax_k.text(0.5, 0.5, "cost critic: checkpoint not found", transform=ax_k.transAxes, ha="center")
    ax_k.set_ylabel("cost critic V1")
    ax_k.set_xlabel("time since takeoff [s]")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    if window is not None:
        for ax in axes:
            ax.set_xlim(*window)


def mode_legend(fig) -> None:
    from matplotlib.patches import Patch
    handles = [Patch(color=col, label=lab) for col, lab in MODE_SHADE.values() if lab]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8, frameon=False)


def summary(d, s) -> dict:
    pol = d["mode"] == "policy"
    out = dict(policy_s=pol.sum() * 0.01)
    for oid, cl in s["clear"].items():
        x = cl[pol]
        out[oid] = (np.nanmin(x), int(np.sum(np.diff((x < 0).astype(int)) == 1)), int(np.sum(np.diff((x < -0.05).astype(int)) == 1)))
    out["r1_max"] = np.nanmax(s["r1"][pol])
    out["unsafe_s"] = np.nansum(s["safety"][pol] < 0) * 0.01
    return out


def plot_run(run: Path, out_dir: Path, cfg, ck_arg) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load_run(run)
    if d is None:
        print(f"{run.name}: no policy flight (skipped)")
        return []
    ck = find_checkpoint(ck_arg, d["meta"])
    c = train_constants(ck.parent / "params" / "env.yaml" if ck else None)
    s = safety_terms(d, c)
    vc = None
    if ck is not None:
        critic = load_cost_critic(ck)
        idx, X = critic_inputs(d, c, cfg, TIME_LEFT)
        vc = (idx, critic(X))
    pol = d["mode"] == "policy"
    k_min = int(np.nanargmin(np.where(pol, s["safety"], np.inf)))
    sm = summary(d, s)
    tt = d["t"] - d["t0"]
    files = []

    fig = plt.figure(figsize=(16, 15), layout="constrained")
    gs = fig.add_gridspec(5, 2, height_ratios=[4.2, 1, 1, 1, 1])
    draw_maps(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), d, s, c, k_min)
    fig.axes[0].legend(loc="upper left", fontsize=8)
    fig.axes[0].set_title(f"top view; spheres at the closest approach under the policy (t = {tt[k_min]:.1f} s)", fontsize=10)
    fig.axes[1].set_title("side view (x-z); dashed: unsafe bands of the floor, ceiling and wall", fontsize=10)
    axes = [fig.add_subplot(gs[1 + i, :]) for i in range(4)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    draw_series(axes, d, s, vc, c)
    for ax, ncol in zip(axes, (5, 3, 2, 1)):
        ax.legend(loc="upper left", fontsize=8, ncol=ncol)
    per = "; ".join(f"obstacle {oid}: min clearance {v[0] * 100:.0f} cm, {v[1]} unsafe-band entries, {v[2]} collisions"
                    for oid, v in sm.items() if isinstance(oid, int))
    fig.suptitle(f"{run.name}: {sm['policy_s']:.0f} s on the policy, {sm['unsafe_s']:.2f} s inside an unsafe band, max r1 {sm['r1_max']:.3f}\n{per}",
                 fontsize=12)
    mode_legend(fig)
    files.append(out_dir / f"{run.name}_overview.png")
    fig.savefig(files[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    events, used = [], np.zeros(len(tt), bool)   # the three closest approaches under the policy, 5 s apart
    for k in np.argsort(np.where(pol, s["safety"], np.inf)):
        if not pol[k] or np.isnan(s["safety"][k]):
            break
        if not used[k]:
            events.append(k)
            used |= np.abs(tt - tt[k]) < 5.0
        if len(events) == 3:
            break
    fig = plt.figure(figsize=(6 * len(events), 17), layout="constrained")
    gs = fig.add_gridspec(6, len(events), height_ratios=[2.4, 2.4, 1, 1, 1, 1])
    for j, k in enumerate(events):
        w = (tt[k] - 4.0, tt[k] + 4.0)
        draw_maps(fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j]), d, s, c, k, window=(tt[k] - 2.0, tt[k] + 2.0), zoom=0.8)
        fig.axes[-2].set_title(f"t = {tt[k]:.1f} s: clearance {s['safety'][k] * 100:.0f} cm (paths +-2 s)", fontsize=10)
        axes = [fig.add_subplot(gs[2 + i, j]) for i in range(4)]
        draw_series(axes, d, s, vc, c, window=w)
        axes[0].set_ylim(-0.12, 1.0)
        for ax in axes:
            ax.axvline(tt[k], color="k", lw=0.8, ls=":")
        if j == 0:
            for ax in axes:
                ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"{run.name}: closest approaches under the policy", fontsize=12)
    mode_legend(fig)
    files.append(out_dir / f"{run.name}_closest.png")
    fig.savefig(files[-1], dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"{run.name}: policy {sm['policy_s']:.0f} s, unsafe {sm['unsafe_s']:.2f} s, max r1 {sm['r1_max']:.3f}; {per}"
          + ("" if ck else "; cost critic skipped (checkpoint not found)"))
    return files


# #### // CLI // ####
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="policy-flight log folders, or a folder of them")
    ap.add_argument("--out", type=Path, help="output folder (default <first run's parent>/plots)")
    ap.add_argument("--checkpoint", help="SafePPO checkpoint for the cost critic (default: the one in the policy's metadata)")
    args = ap.parse_args(argv)
    runs = []
    for r in args.runs:
        runs += sorted(p for p in r.iterdir() if (p / "control.csv").exists()) if not (r / "control.csv").exists() else [r]
    out = args.out or runs[0].parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load()
    for run in runs:
        for f in plot_run(run, out, cfg, args.checkpoint):
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
