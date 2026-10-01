"""Test plans: the segments flown after takeoff and settle."""

from dataclasses import dataclass


@dataclass
class Segment:
    name: str
    kind: str                    # hold | brake | vel | mu_open
    duration: float
    cmd: tuple = (0.0, 0.0, 0.0)  # vel: velocity [m/s] on `axes`; mu_open: absolute mu
    axes: tuple = ()
    settled_entry: bool = False


def hover_plan(seconds: float = 30.0) -> list[Segment]:
    """Position hold; gives the hover crossing mu_z and the trim."""
    return [Segment("hover", "hold", seconds)]


def vstep_plan(xy_mags=(0.20, 0.30, 0.40), z_mags=(0.10, 0.18), step=1.5, brake=1.0, recover=2.0) -> list[Segment]:
    """Mirrored single-axis velocity steps through the KPV loop; gives tau, k, trim and delay."""
    segs = []
    for i, mag in enumerate(xy_mags):
        axes = [("x", 0, mag), ("y", 1, mag)] + ([("z", 2, z_mags[i])] if i < len(z_mags) else [])
        for name, ax, m in axes:
            for sign in (1.0, -1.0):
                cmd = [0.0, 0.0, 0.0]
                cmd[ax] = sign * m
                tag = f"{name}{'+' if sign > 0 else '-'}{m:.2f}"
                segs += [
                    Segment(tag, "vel", step, tuple(cmd), (ax,)),
                    Segment(f"{tag}_brake", "brake", brake),
                    Segment(f"{tag}_recover", "hold", recover),
                ]
    return segs


def thrust_plan(mu_z=(2, 3, 4, 5, 6, 7, 8), step=0.4, recover=2.0, passes=1) -> list[Segment]:
    """Open-loop mu_z staircase from a settled hover; gives alpha_z, b_z and the vertical ceiling."""
    segs = []
    for p in range(passes):
        for m in mu_z:
            tag = f"muz{float(m):.1f}_p{p}"
            segs += [
                Segment(tag, "mu_open", step, (0.0, 0.0, float(m)), settled_entry=True),
                Segment(f"{tag}_recover", "hold", recover),
            ]
    return segs


def build_plan(name: str, hover_seconds: float = 30.0, passes: int = 1) -> tuple[list[Segment], float]:
    """Return (segments, hover height [m]) for a plan name."""
    if name == "hover":
        return hover_plan(hover_seconds), 0.8
    if name == "vsteps":
        return vstep_plan(), 0.8
    if name == "thrust":
        return thrust_plan(passes=passes), 0.5
    raise ValueError(f"unknown plan {name!r} (hover | vsteps | thrust)")
