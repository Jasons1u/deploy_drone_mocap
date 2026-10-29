#!/usr/bin/env python3
"""Showcase plots of policy flights: the drone dodging hand-moved obstacles, as one figure per run and a short video per dodge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_safety import find_checkpoint, load_run, train_constants  # noqa: E402

INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DRONE, CONTACT = "#2a78d6", "#d03b3b"
OBSTACLE = ["#eb6834", "#1baf7a", "#eda100"]   # by sorted obstacle id
HORIZON = 1.0                                   # [s] an approach: the obstacle would reach a drone holding still within this
WINDOW = (-1.2, 0.6)                            # [s] around the would-be hit
VIDEO_WINDOW = (-2.0, 1.0)                      # [s] same, for the videos
GAP_LIM = (-15, 70)                             # [cm]
SNAP_SHADE = (0.35, 0.55, 0.78, 1.0)            # snapshot tints, earliest first
STYLE = {"font.size": 11, "axes.edgecolor": LINE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}


# #### // Approaches: moments a drone holding still would have been hit // ####
def gaps(d: dict, rd: float) -> dict:
    return {oid: np.linalg.norm(o["pos"] - d["p"], axis=1) - o["radius"] - rd for oid, o in d["obstacles"].items()}


def approaches(d: dict, rd: float) -> list[dict]:
    """Per approach: freeze row kf, would-be hit row kh, gap had the drone held still at kf, and the actual closest gap."""
    t, p, pol = d["t"], d["p"], d["mode"] == "policy"
    nh = int(round(HORIZON / np.median(np.diff(t))))
    gap, out = gaps(d, rd), []
    for oid, o in d["obstacles"].items():
        qw = np.lib.stride_tricks.sliding_window_view(o["pos"], nh, axis=0)
        held = np.linalg.norm(qw - p[: len(qw), :, None], axis=1).min(1) - o["radius"] - rd
        held[~np.lib.stride_tricks.sliding_window_view(pol, nh).all(1)] = np.nan
        held = np.r_[held, np.full(nh - 1, np.nan)]
        hit = held < 0
        for s, e in zip(np.flatnonzero(hit & ~np.r_[False, hit[:-1]]), np.flatnonzero(hit & ~np.r_[hit[1:], False])):
            kf = s + int(np.argmin(held[s: e + 1]))
            kh = kf + int(np.argmin(np.linalg.norm(o["pos"][kf: kf + nh] - p[kf], axis=1)))
            w = slice(kf, kf + nh)
            out.append(dict(oid=oid, kf=kf, kh=kh, held=held[kf], gap=np.nanmin(gap[oid][w]),
                            others=min((np.nanmin(np.r_[g[w], np.inf]) for j, g in gap.items() if j != oid), default=np.inf),
                            speed=np.nanmax(np.linalg.norm(o["vel"][kf: kh + 1], axis=1))))
    return sorted(out, key=lambda e: e["kh"])


def featured(evs: list[dict], t: np.ndarray, n: int, margin: float) -> list[dict]:
    """The clearest dodges: a real hit had it held still, margin kept, obstacle at >= 1 m/s, others > 30 cm away; most gap gained first."""
    ok = lambda e: e["held"] <= -margin and e["gap"] >= margin and e["speed"] >= 1.0 and e["others"] > 0.3
    out = []
    for e in sorted(filter(ok, evs), key=lambda e: e["held"] - e["gap"]):
        if len(out) < n and all(abs(t[e["kh"]] - t[f["kh"]]) > 3.0 for f in out):
            out.append(e)
    return sorted(out, key=lambda e: e["kh"])


# #### // Drawing helpers // ####
def tint(color: str, a: float) -> tuple:
    from matplotlib.colors import to_rgb
    return tuple(a * x + 1.0 - a for x in to_rgb(color))


def obstacle_color(d: dict, oid: int) -> str:
    i = sorted(d["obstacles"]).index(oid)
    return OBSTACLE[i] if i < len(OBSTACLE) else MUTED


def view_basis(pts: np.ndarray) -> np.ndarray:
    """Rows: screen right and up in the plane that best fits the points; world up points up on screen when it lies in the plane."""
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    n = vt[2] if vt[2, 2] >= 0 else -vt[2]
    up = np.array([0.0, 0.0, 1.0]) - n[2] * n
    up = up if np.linalg.norm(up) > 0.3 else vt[1]   # near-horizontal plane: seen from above
    up = up / np.linalg.norm(up)
    return np.stack([np.cross(up, n), up])


def event_view(d: dict, e: dict, rate: float) -> dict:
    k0, k1 = e["kf"] - int(0.5 * rate), e["kh"] + int(0.4 * rate)
    pts = np.vstack([d["p"][k0:k1], d["obstacles"][e["oid"]]["pos"][k0:k1]])
    pts = pts[~np.isnan(pts[:, 0])]
    B = view_basis(pts)
    xy = pts @ B.T
    rel = d["obstacles"][e["oid"]]["pos"][snap_rows(e, rate)] - d["p"][snap_rows(e, rate)]
    err = float(np.max(np.linalg.norm(rel, axis=1) - np.linalg.norm(rel @ B.T, axis=1)))   # drawn vs 3D distance at the snapshots
    return dict(B=B, k=(k0, k1), center=(xy.min(0) + xy.max(0)) / 2, span=float((xy.max(0) - xy.min(0)).max()), err=err)


def frame_map(ax, B: np.ndarray, center: np.ndarray, half: float) -> None:
    """Limits, scale bar and an 'up' arrow (world up projected on the view plane)."""
    ax.set_aspect("equal")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(LINE)
    x0, y0 = center - half * 0.88
    ax.plot([x0, x0 + 0.2], [y0, y0], color=INK2, lw=2, solid_capstyle="butt")
    ax.text(x0 + 0.1, y0 + half * 0.03, "20 cm", ha="center", va="bottom", color=INK2, fontsize=9)
    up = B[1, 2]
    if up > 0.3:
        xa, ya = center[0] + half * 0.85, center[1] - half * 0.88
        ax.annotate("", xy=(xa, ya + 0.18 * up), xytext=(xa, ya), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
        ax.text(xa - half * 0.04, ya + 0.09 * up, "up", ha="right", va="center", color=INK2, fontsize=9)
    else:
        ax.text(center[0] + half * 0.9, center[1] - half * 0.88, "seen from above", ha="right", va="bottom", color=INK2, fontsize=9)


def snap_rows(e: dict, rate: float) -> list[int]:
    return [e["kf"] - int(0.3 * rate), e["kf"], e["kh"], e["kh"] + int(0.25 * rate)]


def snap_disc(ax, xy: np.ndarray, r: float, color: str, a: float, i: int, z: int, out: np.ndarray | None = None) -> None:
    """A numbered snapshot disc; the number goes just outside along `out` when something covers the centre."""
    from matplotlib.patches import Circle
    ax.add_patch(Circle(xy, r, fc=tint(color, a), ec="white", lw=1.5, zorder=z))
    if out is None:
        ax.text(*xy, str(i), ha="center", va="center", fontsize=8, color="white" if a > 0.6 else INK, zorder=z)
    else:
        ax.text(*(xy + out * (r + 0.035)), str(i), ha="center", va="center", fontsize=8, color=INK2, zorder=6)


def draw_snapshots(ax, d: dict, e: dict, v: dict, rd: float, label: bool) -> None:
    from matplotlib.patches import Circle
    o, col, rate = d["obstacles"][e["oid"]], obstacle_color(d, e["oid"]), 1.0 / np.median(np.diff(d["t"]))
    k0, k1 = v["k"]
    for path, c in ((d["p"], DRONE), (o["pos"], col)):
        xy = path[k0:k1] @ v["B"].T
        ax.plot(*xy.T, color=tint(c, 0.55), lw=1.3, zorder=1)
        ax.annotate("", xy=xy[-1], xytext=xy[-6], arrowprops=dict(arrowstyle="-|>", color=tint(c, 0.55), lw=1.3), zorder=1)
    g = d["p"][e["kf"]] @ v["B"].T
    for i, (a, k) in enumerate(zip(SNAP_SHADE, snap_rows(e, rate))):
        xo, xd = o["pos"][k] @ v["B"].T, d["p"][k] @ v["B"].T
        out = xo - xd if np.linalg.norm(xo - g) < o["radius"] + rd else None   # covered by the held-still drone
        snap_disc(ax, xo, o["radius"], col, a, i + 1, 2, None if out is None else out / max(np.linalg.norm(out), 1e-6))
        snap_disc(ax, xd, rd, DRONE, a, i + 1, 4)
    ax.add_patch(Circle(g, rd * 1.25, fill=False, ec=INK, lw=1.3, ls=(0, (2, 1.5)), zorder=5))
    if label:   # away from the obstacle's path
        away = g - np.nanmean(o["pos"][k0:k1] @ v["B"].T, 0)
        away = away / max(np.linalg.norm(away), 1e-6)
        ax.annotate("held still", xy=g + away * rd * 1.3, xytext=g + away * 0.25, color=INK2, fontsize=10, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), zorder=6)


def gap_curves(d: dict, e: dict, rd: float) -> tuple:
    """Time from the would-be hit [s], actual gap and gap had the drone held still from the freeze on [cm] (= actual before it)."""
    t, o = d["t"], d["obstacles"][e["oid"]]
    g = np.linalg.norm(o["pos"] - d["p"], axis=1) - o["radius"] - rd
    held = np.where(t >= t[e["kf"]], np.linalg.norm(o["pos"] - d["p"][e["kf"]], axis=1) - o["radius"] - rd, g)
    return t - t[e["kh"]], g * 100, held * 100


def draw_gap(ax, d: dict, e: dict, rd: float, lw: float = 2.2, alpha: float = 1.0, snaps: bool = True) -> None:
    tt, g, held = gap_curves(d, e, rd)
    sel = (tt >= WINDOW[0]) & (tt <= WINDOW[1])
    hs = sel & (d["t"] >= d["t"][e["kf"]])
    ax.plot(tt[hs], held[hs], color=tint(INK2, alpha), lw=lw * 0.75, ls=(0, (3, 2)), zorder=2)
    ax.plot(tt[sel], g[sel], color=tint(DRONE, alpha), lw=lw, zorder=3)
    if snaps:
        for i, (a, k) in enumerate(zip(SNAP_SHADE, snap_rows(e, 1.0 / np.median(np.diff(d["t"]))))):
            ax.plot(tt[k], g[k], "o", ms=13, mfc=tint(DRONE, a), mec="white", mew=1.2, zorder=4)
            ax.text(tt[k], g[k], str(i + 1), ha="center", va="center", fontsize=8, color="white" if a > 0.6 else INK, zorder=5)


def frame_gap(ax, ylim: tuple, label_contact: bool, xlim: tuple = WINDOW) -> None:
    ax.axhspan(ylim[0], 0, color=tint(CONTACT, 0.12), lw=0, zorder=0)
    ax.axhline(0, color=CONTACT, lw=1.0, zorder=1)
    ax.axvline(0, color=LINE, lw=1.0, zorder=0)
    if label_contact:
        ax.text(xlim[0] + 0.05, ylim[0] / 2, "contact", va="center", color=INK2, fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_xlabel("time from would-be hit [s]")


# #### // Figure and videos // ####
def plot_figure(d: dict, evs: list[dict], feats: list[dict], rd: float, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    t = d["t"] - d["t0"]
    rate = 1.0 / np.median(np.diff(t))
    n = len(feats)
    views = [event_view(d, e, rate) for e in feats]
    half = max(v["span"] for v in views) / 2 + 0.08
    fig = plt.figure(figsize=(4.3 * n + 4.6, 9.6), layout="constrained")
    gs = fig.add_gridspec(3, n + 1, height_ratios=[0.07, 1.35, 1], width_ratios=[1.12] + [1] * n)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    pol_s = (d["mode"] == "policy").sum() / rate
    clear = [e["gap"] for e in evs if e["gap"] > 0]
    stats = [(f"{pol_s:.0f} s", "flying the policy while the obstacles\nwere moved around it by hand"),
             (f"{len(evs)}", f"approaches: an obstacle would have hit\na drone holding still within {HORIZON:g} s"),
             (f"{len(clear)} of {len(evs)}", f"ended without contact\n(median closest gap {np.median(clear) * 100:.0f} cm)")]
    for i, (big, small) in enumerate(stats):
        y = 0.97 - i * 0.34
        ax.text(0.0, y, big, transform=ax.transAxes, fontsize=30, color=INK, va="top")
        ax.text(0.0, y - 0.15, small, transform=ax.transAxes, fontsize=11, color=INK2, va="top", linespacing=1.3)

    ax_all = fig.add_subplot(gs[2, 0])
    grid = np.arange(WINDOW[0], WINDOW[1] + 1e-9, 0.01)
    both = np.array([[np.interp(grid, c[0], c[i]) for i in (1, 2)] for c in (gap_curves(d, e, rd) for e in evs)])
    for e in evs:
        draw_gap(ax_all, d, e, rd, lw=1.0, alpha=0.25, snaps=False)
    ax_all.plot(grid, np.nanmedian(both[:, 1], 0), color=INK2, lw=2.0, ls=(0, (3, 2)), zorder=4)
    ax_all.plot(grid, np.nanmedian(both[:, 0], 0), color=DRONE, lw=2.8, zorder=5)
    ax_all.text(WINDOW[1] - 0.03, GAP_LIM[1] - 3, "bold: median", ha="right", va="top", color=INK2, fontsize=9)
    frame_gap(ax_all, GAP_LIM, True)
    ax_all.set_ylabel("gap between surfaces [cm]")
    ax_all.set_title(f"All {len(evs)} approaches, aligned", loc="left", fontsize=12)

    for j, (e, v) in enumerate(zip(feats, views)):
        ax_m = fig.add_subplot(gs[1, j + 1])
        draw_snapshots(ax_m, d, e, v, rd, label=j == 0)
        frame_map(ax_m, v["B"], v["center"], half)
        ax_m.set_title(f"Dodge {j + 1}  ·  {t[e['kh']]:.0f} s into the flight\nobstacle {e['oid']} at {e['speed']:.1f} m/s, closest gap {e['gap'] * 100:.0f} cm",
                       loc="left", fontsize=12, linespacing=1.4)
        ax_g = fig.add_subplot(gs[2, j + 1], sharey=ax_all)
        draw_gap(ax_g, d, e, rd)
        frame_gap(ax_g, GAP_LIM, False)
        ax_g.tick_params(labelleft=False)

    dot = lambda c, **k: Line2D([], [], ls="", marker="o", ms=11, mfc=c, mec="white", **k)
    handles = [dot(DRONE, label="drone"), *[dot(obstacle_color(d, oid), label=f"obstacle {oid}") for oid in sorted(d["obstacles"])],
               Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec=INK, label="drone, had it held still"),
               Line2D([], [], color=DRONE, lw=2, label="gap, actual"), Line2D([], [], color=INK2, lw=1.5, ls=(0, (3, 2)), label="gap, had it held still"),
               dot(tint(MUTED, 0.35), label="1 → 4: snapshots in time order")]
    ax_l = fig.add_subplot(gs[0, :])
    ax_l.axis("off")
    ax_l.legend(handles=handles, loc="center left", ncol=len(handles), fontsize=10, handletextpad=0.3, columnspacing=1.6, borderaxespad=0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=16, color=INK)
    fig.text(0.01, -0.01, f"Views: the plane that best fits each dodge (at the snapshots, drawn distances are within {np.ceil(max(v['err'] for v in views) * 100):.0f} cm "
             "of the 3D ones).\nSnapshots: 1 = 0.3 s before the freeze, "
             "2 = the freeze (where the drone would have held still), 3 = the would-be hit, 4 = 0.25 s after. Gap: distance between the drone (5 cm) and obstacle (10 cm) spheres.",
             fontsize=9, color=MUTED, ha="left", va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def animate(d: dict, e: dict, rd: float, title: str, path: Path, slow: float, fps: int = 30) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    t = d["t"] - d["t0"]
    rate = 1.0 / np.median(np.diff(t))
    o, col = d["obstacles"][e["oid"]], obstacle_color(d, e["oid"])
    v = event_view(d, e, rate)
    B, th = v["B"], t[e["kh"]]
    k0, k1 = int(np.searchsorted(t, th + VIDEO_WINDOW[0])), int(np.searchsorted(t, th + VIDEO_WINDOW[1]))
    xy_d, xy_o = d["p"] @ B.T, o["pos"] @ B.T
    g = np.linalg.norm(o["pos"] - d["p"], axis=1) - o["radius"] - rd
    held = np.linalg.norm(o["pos"] - d["p"][e["kf"]], axis=1) - o["radius"] - rd
    span = np.vstack([xy_d[k0:k1], xy_o[k0:k1]])
    span = span[~np.isnan(span[:, 0])]
    center, half = (span.min(0) + span.max(0)) / 2, (span.max(0) - span.min(0)).max() / 2 + 0.15

    fig, (ax_m, ax_g) = plt.subplots(1, 2, figsize=(12, 5.4), width_ratios=[1.1, 1], layout="constrained")
    frame_map(ax_m, B, center, half)
    frame_gap(ax_g, GAP_LIM, True, VIDEO_WINDOW)
    ax_g.set_ylabel("gap between surfaces [cm]")
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, color=INK)
    tr_d, = ax_m.plot([], [], color=tint(DRONE, 0.55), lw=1.5)
    tr_o, = ax_m.plot([], [], color=tint(col, 0.55), lw=1.5)
    c_o = ax_m.add_patch(Circle((0, 0), o["radius"], fc=col, ec="white", lw=1.5, zorder=2))
    c_d = ax_m.add_patch(Circle((0, 0), rd, fc=DRONE, ec="white", lw=1.5, zorder=4))
    c_h = ax_m.add_patch(Circle(xy_d[e["kf"]], rd * 1.25, fill=False, ec=INK, lw=1.3, ls=(0, (2, 1.5)), zorder=5, visible=False))
    away = xy_d[e["kf"]] - np.nanmean(xy_o[e["kf"]: e["kh"] + int(0.4 * rate)], 0)
    away = away / max(np.linalg.norm(away), 1e-6)
    l_h = ax_m.annotate("held still", xy=xy_d[e["kf"]] + away * rd * 1.3, xytext=xy_d[e["kf"]] + away * 0.25, color=INK2, fontsize=10,
                        ha="center", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), visible=False)
    ln_g, = ax_g.plot([], [], color=DRONE, lw=2.2, label="actual")
    ln_h, = ax_g.plot([], [], color=INK2, lw=1.6, ls=(0, (3, 2)), label="had the drone held still")
    cur = ax_g.axvline(WINDOW[0], color=INK2, lw=0.8)
    ax_g.legend(loc="upper right", fontsize=10)
    dot = lambda c, lab: Line2D([], [], ls="", marker="o", ms=10, mfc=c, mec="white", label=lab)
    ax_m.legend(handles=[dot(DRONE, "drone"), dot(col, f"obstacle {e['oid']}")], loc="upper left", fontsize=10)
    clock = ax_m.text(0.98, 0.98, "", transform=ax_m.transAxes, ha="right", va="top", color=INK2, fontsize=10)

    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(path), dpi=120):
        for tf in np.arange(t[k0], t[k1], slow / fps):
            k = int(np.searchsorted(t, tf))
            a = max(k0, k - int(0.6 * rate))
            tr_d.set_data(*xy_d[a: k + 1].T)
            tr_o.set_data(*xy_o[a: k + 1].T)
            c_d.center, c_o.center = xy_d[k], xy_o[k]
            c_o.set_visible(not np.isnan(xy_o[k, 0]))
            c_h.set_visible(k >= e["kf"])
            l_h.set_visible(k >= e["kf"])
            ln_g.set_data(t[k0: k + 1] - th, g[k0: k + 1] * 100)
            ln_h.set_data(t[e["kf"]: k + 1] - th, held[e["kf"]: k + 1] * 100)
            cur.set_xdata([t[k] - th] * 2)
            clock.set_text(f"{t[k] - th:+.2f} s   ·   {slow:g}× speed")
            writer.grab_frame(facecolor="white")
    plt.close(fig)


def plot_run(run: Path, out_dir: Path, ck_arg: str | None, n: int, slow: float, video: bool) -> list[Path]:
    import matplotlib.pyplot as plt
    d = load_run(run)
    if d is None:
        print(f"{run.name}: no policy flight (skipped)")
        return []
    ck = find_checkpoint(ck_arg, d["meta"])
    c = train_constants(ck.parent / "params" / "env.yaml" if ck else None)
    rd, t = c["drone_radius"], d["t"] - d["t0"]
    evs = approaches(d, rd)
    feats = featured(evs, t, n, c["unsafe_margin"])
    if not feats:
        print(f"{run.name}: {len(evs)} approaches, none clean enough to feature (skipped)")
        return []
    files = [out_dir / f"{run.name}_dodges.png"]
    with plt.rc_context(STYLE):
        plot_figure(d, evs, feats, rd, f"The policy dodging hand-moved obstacles  ·  {run.name}", files[0])
        for j, e in enumerate(feats if video else []):
            files.append(out_dir / f"{run.name}_dodge{j + 1}.mp4")
            animate(d, e, rd, f"Dodge {j + 1}  ·  obstacle {e['oid']} at {e['speed']:.1f} m/s, closest gap {e['gap'] * 100:.0f} cm  ·  {run.name}",
                    files[-1], slow)
    n_clear = sum(e["gap"] > 0 for e in evs)
    print(f"{run.name}: {len(evs)} approaches, {n_clear} without contact; featured at t = " + ", ".join(f"{t[e['kh']]:.1f}" for e in feats) + " s")
    return files


# #### // CLI // ####
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="policy-flight log folders, or a folder of them")
    ap.add_argument("--out", type=Path, help="output folder (default <first run's parent>/plots)")
    ap.add_argument("--checkpoint", help="SafePPO checkpoint, for the training constants (default: the one in the policy's metadata)")
    ap.add_argument("--dodges", type=int, default=3, help="dodges to feature per run")
    ap.add_argument("--slow", type=float, default=0.5, help="video playback speed (0.5 = half speed)")
    ap.add_argument("--no-video", action="store_true", help="figure only")
    args = ap.parse_args(argv)
    runs = []
    for r in args.runs:
        runs += sorted(p for p in r.iterdir() if (p / "control.csv").exists()) if not (r / "control.csv").exists() else [r]
    out = args.out or runs[0].parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    for run in runs:
        for f in plot_run(run, out, args.checkpoint, args.dodges, args.slow, not args.no_video):
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
