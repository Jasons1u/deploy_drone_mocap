"""Hardware-free check: fly each plan against a plant with known parameters, fit, compare; then fault-inject the safety guards."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

import fit as fitmod
from fly import run_sim
from sim_plant import PlantParams

TRUE = PlantParams(alpha=(0.60, 0.66, 0.55), drag=(0.75, 0.85, 0.0), hover_mu=3.12, trim_acc=(0.06, -0.05),
                   a_z_max=1.10, delay=0.025)
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, bool(ok), detail))


def close(name, got, want, rel=None, abs_=None):
    tol = rel * abs(want) if rel is not None else abs_
    check(name, got is not None and abs(got - want) <= tol, f"got {got:.4f} want {want:.4f} (tol {tol:.4f})")


def recovery(tmp: Path) -> None:
    paths = {plan: run_sim(plan, tmp / f"{plan}.csv", params=TRUE, seed=i)[0]
             for i, plan in enumerate(["hover", "vsteps", "thrust"])}
    data = {k: fitmod.load(p) for k, p in paths.items()}
    allp = list(data.values())
    rp = fitmod.fit_joint(allp)
    for ax, i in (("x", 0), ("y", 1), ("z", 2)):
        close(f"replay alpha_{ax}", rp[ax]["alpha"], TRUE.alpha[i], rel=0.05)
        close(f"replay delay_{ax}", rp[ax]["delay"], TRUE.delay, abs_=0.006)
    for ax, i in (("x", 0), ("y", 1)):
        close(f"replay drag_{ax}", rp[ax]["drag"], TRUE.drag[i], rel=0.10)
        close(f"replay trim_{ax}", rp[ax]["trim_acc"], TRUE.trim_acc[i], abs_=0.02)
    close("replay hover mu_z", rp["z"]["hover_mu"], TRUE.hover_mu, abs_=0.03)
    close("replay ceiling a_z", rp["z"].get("a_z_max", np.nan), TRUE.a_z_max, rel=0.10)

    hov = fitmod.hover_stats(data["hover"])
    close("hover-rows mu_z", hov["hover_mu"], TRUE.hover_mu, abs_=0.03)
    th = fitmod.thrust_fits(data["thrust"])
    close("thrust alpha_z", th["alpha_z"], TRUE.alpha[2], rel=0.08)
    close("thrust hover mu_z", th["hover_mu"], TRUE.hover_mu, abs_=0.06)
    close("thrust ceiling", th["a_z_max"], TRUE.a_z_max, rel=0.10)

    steps = fitmod.pair_summary(fitmod.step_fits(data["vsteps"]))
    for ax, i in (("x", 0), ("y", 1)):
        k_true = TRUE.alpha[i] * fitmod.KPV[i] / (TRUE.alpha[i] * fitmod.KPV[i] + TRUE.drag[i])
        close(f"step-fit k_{ax}", steps[ax]["k"], k_true, rel=0.08)
    for name, p in paths.items():
        meta = p.with_suffix(".json").read_text()
        check(f"{name} flight completed", '"stop_reason": ""' in meta, "no stop reason")


def scenario(tmp: Path, name: str, params=TRUE, inject=None, dry_run=False):
    _, ctrl = run_sim("hover", tmp / f"s_{name}.csv", params=params, hover_seconds=10.0, inject=inject, dry_run=dry_run)
    return ctrl, np.array(ctrl.sent_flags)


def safety(tmp: Path) -> None:
    ctrl, flags = scenario(tmp, "lowbatt", params=replace(TRUE, vbat=3.5))
    check("low battery refuses to arm", "below arm limit" in ctrl.stop_reason and flags.max(initial=0) == 0, ctrl.stop_reason)

    def stale(t, plant, ctrl):
        return None if 9.0 <= t <= 9.5 else plant.pose()
    ctrl, flags = scenario(tmp, "stale", inject=stale)
    check("stale mocap stops motors", ctrl.stop_reason.startswith("stale mocap") and flags[-1] == 0, ctrl.stop_reason)

    frozen = {}

    def freeze(t, plant, ctrl):
        if 9.0 <= t <= 10.0:
            return frozen.setdefault("p", plant.pose())
        return plant.pose()
    ctrl, _ = scenario(tmp, "frozen", inject=freeze)
    check("frozen mocap lands", any("frozen mocap" in e for e in ctrl.events) and ctrl.phase == "done", "; ".join(ctrl.events[-3:]))

    def jump(t, plant, ctrl):
        return plant.pose() + (np.array([0.5, 0.0, 0.0]) if t >= 9.0 else 0.0)
    ctrl, flags = scenario(tmp, "jump", inject=jump)
    check("pose jump stops motors", ctrl.stop_reason.startswith("pose jump") and flags[-1] == 0, ctrl.stop_reason)

    def drift(t, plant, ctrl):
        return plant.pose() + np.array([max(0.0, t - 9.0) * 3.0, 0.0, 0.0])
    ctrl, _ = scenario(tmp, "geofence", inject=drift)
    check("geofence breach lands", any("geofence" in e for e in ctrl.events), "; ".join(e for e in ctrl.events if "geofence" in e)[:80])

    def kill(t, plant, ctrl):
        if t >= 9.0:
            ctrl.request_stop("kill key")
        return plant.pose()
    ctrl, flags = scenario(tmp, "kill", inject=kill)
    check("kill stops motors", ctrl.stop_reason == "kill key" and flags[-1] == 0, ctrl.stop_reason)

    ctrl, flags = scenario(tmp, "dry", dry_run=True)
    check("dry run never arms", flags.max(initial=0) == 0 and any("phase -> test" in e for e in ctrl.events),
          f"max flag {flags.max(initial=0)}, reached test: {any('phase -> test' in e for e in ctrl.events)}")


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        recovery(tmp)
        safety(tmp)
    w = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  {detail}")
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
