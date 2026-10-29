#!/usr/bin/env python3
"""Animate a window of a policy flight: drone, policy target and obstacles in plan and side view, with the gaps over time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_dodges import CONTACT, DRONE, INK, INK2, LINE, MUTED, STYLE, obstacle_color, tint  # noqa: E402
from plot_safety import load_run  # noqa: E402

TRAIL = 1.0                                       # [s] of path drawn behind each body
MARGIN = 0.15                                     # [m] around everything in the window
VIEWS = [("from above", 1, 0, True), ("from +x", 1, 2, False)]   # title, horizontal axis, vertical axis, flip vertical
AXIS = "xyz"


def limits(pts: np.ndarray, r: float) -> np.ndarray:
    pts = pts[~np.isnan(pts[:, 0])]
    return np.stack([pts.min(0) - r - MARGIN, pts.max(0) + r + MARGIN], 1)


def animate(d: dict, name: str, start: float, dur: float, speed: float, path: Path, fps: int = 30) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    t = d["t"] - d["t0"]
    rate = 1.0 / np.median(np.diff(t))
    k0, k1 = (int(k) for k in np.searchsorted(t, [start, start + dur]))
    rd = float(d["meta"]["config"]["training"]["drone_radius"])
    bodies = [("drone", d["p"], rd, DRONE)] + [(f"obstacle {i}", o["pos"], o["radius"], obstacle_color(d, i)) for i, o in d["obstacles"].items()]
    tgt = d["target"]
    lim = limits(np.vstack([b[1][k0:k1] for b in bodies] + [tgt[k0:k1]]), max(b[2] for b in bodies))
    gaps = {name: np.linalg.norm(p - d["p"], axis=1) - r - rd for name, p, r, _ in bodies[1:]}
    off = np.linalg.norm(d["p"] - tgt, axis=1)

    fig = plt.figure(figsize=(13, 8.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    fig.suptitle(f"{name}  ·  +{start:.1f} to +{start + dur:.1f} s after launch", x=0.01, ha="left", fontsize=14, color=INK)
    maps = []
    for j, (title, h, v, flip) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, j])
        ax.set_aspect("equal")
        ax.set_xlim(*lim[h])
        ax.set_ylim(*(lim[v][::-1] if flip else lim[v]))
        ax.set_xlabel(f"{AXIS[h]} [m]")
        ax.set_ylabel(f"{AXIS[v]} [m]")
        ax.set_title(title, loc="left", fontsize=11, color=INK2)
        ax.grid(color=LINE, lw=0.6)
        trails = [ax.plot([], [], color=tint(c, 0.5), lw=1.5, zorder=1)[0] for _, _, _, c in bodies]
        discs = [ax.add_patch(Circle((0, 0), r, fc=c, ec="white", lw=1.5, zorder=4 if n == "drone" else 3)) for n, _, r, c in bodies]
        mark, = ax.plot([], [], ls="", marker="P", ms=13, mfc="white", mec=INK, mew=1.6, zorder=5)
        maps.append((ax, h, v, trails, discs, mark))
    dot = lambda c, lab: Line2D([], [], ls="", marker="o", ms=10, mfc=c, mec="white", label=lab)
    maps[0][0].legend(handles=[dot(c, n) for n, _, _, c in bodies] + [Line2D([], [], ls="", marker="P", ms=11, mfc="white", mec=INK, label="target (ref)")],
                      loc="upper left", fontsize=9, ncols=2)
    clock = maps[1][0].text(0.98, 0.98, "", transform=maps[1][0].transAxes, ha="right", va="top", color=INK2, fontsize=10)

    ax_g = fig.add_subplot(gs[1, :])
    tw = t[k0:k1]
    ylim = (-0.1, max(1.0, np.nanmax(np.r_[off[k0:k1], *[g[k0:k1] for g in gaps.values()]]) + 0.1))
    ax_g.axhspan(ylim[0], 0, color=tint(CONTACT, 0.12), lw=0, zorder=0)
    ax_g.axhline(0, color=CONTACT, lw=1.0, zorder=1)
    ax_g.text(tw[0] + 0.05, ylim[0] / 2, "contact", va="center", color=INK2, fontsize=9)
    ax_g.set_xlim(tw[0], tw[-1])
    ax_g.set_ylim(*ylim)
    ax_g.grid(axis="y", color=LINE, lw=0.6)
    ax_g.set_xlabel("time after launch [s]")
    ax_g.set_ylabel("distance [m]")
    curves = [(off, DRONE, "drone to target")] + [(g, c, f"gap to {n}") for (n, g), (_, _, _, c) in zip(gaps.items(), bodies[1:])]
    lines = []
    for y, c, lab in curves:
        ax_g.plot(tw, y[k0:k1], color=tint(c, 0.25), lw=1.4)
        lines.append(ax_g.plot([], [], color=c, lw=2.2, label=lab)[0])
    cur = ax_g.axvline(tw[0], color=INK2, lw=0.8)
    ax_g.legend(loc="upper left", fontsize=9, ncols=3)

    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(path), dpi=110):
        for tf in np.arange(t[k0], t[k1 - 1], speed / fps):
            k = int(np.searchsorted(t, tf))
            a = max(k0, k - int(TRAIL * rate))
            for _, h, v, trails, discs, mark in maps:
                for (_, p, _, _), tr, disc in zip(bodies, trails, discs):
                    tr.set_data(p[a: k + 1, h], p[a: k + 1, v])
                    disc.center = (p[k, h], p[k, v])
                    disc.set_visible(not np.isnan(p[k, 0]))
                mark.set_data([tgt[k, h]], [tgt[k, v]])
            for (y, _, _), ln in zip(curves, lines):
                ln.set_data(t[k0: k + 1], y[k0: k + 1])
            cur.set_xdata([t[k]] * 2)
            clock.set_text(f"+{t[k]:.2f} s  ·  {d['mode'][k]}  ·  {speed:g}× speed")
            writer.grab_frame(facecolor="white")
    plt.close(fig)


def animate_distances(d: dict, name: str, start: float, dur: float, speed: float, path: Path, fps: int = 30) -> None:
    """Square video of the gap to each obstacle and the drone's distance from the reference; the last frame also saved as a PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.lines import Line2D
    t = d["t"] - d["t0"]
    k0, k1 = (int(k) for k in np.searchsorted(t, [start, start + dur]))
    tw = t[k0:k1]
    rd = float(d["meta"]["config"]["training"]["drone_radius"])
    gaps = {i: (np.linalg.norm(o["pos"] - d["p"], axis=1) - o["radius"] - rd)[k0:k1] * 100 for i, o in d["obstacles"].items()}
    off = np.linalg.norm(d["p"] - d["target"], axis=1)[k0:k1] * 100
    bg = "#fcfcfb"

    fig = plt.figure(figsize=(9, 9), facecolor=bg)
    fig.text(0.07, 0.955, "Two obstacles jabbed at the drone", fontsize=21, color=INK, va="center")
    policy = Path(d["meta"]["config"]["policy_flight"]["policy"]).stem
    fig.text(0.07, 0.918, f"{name}  ·  policy {policy}  ·  +{tw[0]:.0f} to +{tw[-1]:.0f} s after launch", fontsize=11, color=INK2, va="center")
    clock = fig.text(0.93, 0.955, "", fontsize=11, color=MUTED, ha="right", va="center")
    ax_o = fig.add_axes((0.10, 0.445, 0.72, 0.395), facecolor=bg)
    ax_r = fig.add_axes((0.10, 0.075, 0.72, 0.27), facecolor=bg, sharex=ax_o)
    hi = np.nanmax(np.r_[*gaps.values()])
    ylim_o = (-20, 20 * np.ceil((hi + 15) / 20))
    ax_o.axhspan(ylim_o[0], 0, color=tint(CONTACT, 0.12), lw=0, zorder=0)
    ax_o.axhline(0, color=CONTACT, lw=1.0, zorder=1)
    ax_o.text(tw[0] + 0.12, ylim_o[0] / 2, "contact", va="center", color=INK2, fontsize=10)
    ax_o.set_ylim(*ylim_o)
    ax_r.set_ylim(0, 20 * np.ceil(np.nanmax(off) * 1.12 / 20))
    ax_o.set_xlim(tw[0], tw[-1])
    for ax, title in ((ax_o, "Gap to each obstacle [cm]"), (ax_r, "Drone distance from the reference [cm]")):
        ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
        ax.grid(axis="y", color=LINE, lw=0.6)
        ax.tick_params(length=0, labelsize=10.5)
    ax_o.tick_params(labelbottom=False)
    ax_r.set_xlabel("time after launch [s]", fontsize=11)

    series = [(ax_o, gaps[i], obstacle_color(d, i), f"obstacle {i}") for i in sorted(gaps)] + [(ax_r, off, DRONE, "drone")]
    lines = [ax.plot([], [], color=c, lw=1.8, solid_capstyle="round", zorder=3)[0] for ax, _, c, _ in series]
    heads = [ax.plot([], [], ls="", marker="o", ms=8, mfc=c, mec=bg, mew=2, zorder=4)[0] for ax, _, c, _ in series]
    labels = [ax.text(0, 0, lab, fontsize=11, color=INK2, va="center", clip_on=False, visible=False) for ax, _, _, lab in series[:-1]]
    ax_o.legend(handles=[Line2D([], [], color=c, lw=2.4, label=lab) for _, _, c, lab in series[:-1]], loc="lower right",
                bbox_to_anchor=(1.0, 1.0), ncols=2, fontsize=10.5, handlelength=1.6, borderaxespad=0.3)
    curs = [ax.axvline(tw[0], color=MUTED, lw=0.8, zorder=2) for ax in (ax_o, ax_r)]

    kg = {i: int(np.nanargmin(g)) for i, g in gaps.items()}
    io = min(kg, key=lambda i: gaps[i][kg[i]])
    ko, kr = kg[io], int(np.nanargmax(off))
    notes = [(ko, ax_o.text(tw[ko] - 0.2, ylim_o[0] / 2, f"closest: {gaps[io][ko]:.1f} cm, obstacle {io}, +{tw[ko]:.1f} s".replace("-", "\u2212"),
                            fontsize=10.5, color=INK, ha="right", va="center", visible=False)),
             (kr, ax_r.annotate(f"{off[kr] / 100:.2f} m away at +{tw[kr]:.1f} s", xy=(tw[kr], off[kr]), xytext=(tw[kr] - 0.45, off[kr]),
                                fontsize=10.5, color=INK, ha="right", va="center",
                                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkB=6), visible=False))]
    sep = 0.07 * (ylim_o[1] - ylim_o[0])

    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(path), dpi=120):
        for tf in np.r_[np.arange(tw[0], tw[-1], speed / fps), tw[-1]]:
            k = min(int(np.searchsorted(tw, tf)), len(tw) - 1)
            for (_, y, _, _), ln, hd in zip(series, lines, heads):
                ln.set_data(tw[: k + 1], y[: k + 1])
                hd.set_data([tw[k]], [y[k]])
            ys = [series[j][1][k] for j in range(len(labels))]
            order = np.argsort(ys)
            for a, b in zip(order[:-1], order[1:]):   # push close labels apart
                if ys[b] - ys[a] < sep:
                    mid = (ys[a] + ys[b]) / 2
                    ys[a], ys[b] = mid - sep / 2, mid + sep / 2
            for lab, y in zip(labels, ys):
                lab.set_position((tw[k] + 0.12, y))
                lab.set_visible(not np.isnan(y))
            for c in curs:
                c.set_xdata([tw[k]] * 2)
            for kn, note in notes:
                note.set_visible(k >= kn)
            clock.set_text(f"+{tw[k]:.2f} s  ·  {speed:g}× speed")
            writer.grab_frame(facecolor=bg)
    clock.set_text("")
    for c in curs:
        c.set_visible(False)
    fig.savefig(path.with_suffix(".png"), dpi=120, facecolor=bg)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="policy-flight log folder")
    ap.add_argument("start", type=float, help="window start [s after launch]")
    ap.add_argument("--duration", type=float, default=10.0, help="window length [s]")
    ap.add_argument("--speed", type=float, default=1.0, help="video playback speed (0.5 = half speed)")
    ap.add_argument("--distances", action="store_true", help="square video of the distances only (plus a PNG of the last frame)")
    ap.add_argument("--out", type=Path, help="output file (default <run's parent>/plots/<run>_<start>s[_distances].mp4)")
    args = ap.parse_args(argv)
    d = load_run(args.run)
    if d is None:
        print(f"{args.run.name}: no policy flight")
        return 1
    out = args.out or args.run.parent / "plots" / f"{args.run.name}_{args.start:g}s{'_distances' if args.distances else ''}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        (animate_distances if args.distances else animate)(d, args.run.name, args.start, args.duration, args.speed, out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
