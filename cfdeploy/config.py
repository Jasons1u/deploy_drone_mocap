"""Settings from config/deploy.yaml with attribute access; scripts override single values with --set section.key=value."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/deploy.yaml"


class Section(dict):
    """A config mapping that also reads as attributes: cfg.gamepad.buttons.rb."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value


def wrap(value):
    if isinstance(value, dict):
        return Section({k: wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [wrap(v) for v in value]
    return value


def load(path: Path = CONFIG) -> Section:
    return wrap(yaml.safe_load(Path(path).read_text()))


def repo_path(p) -> Path:
    """Config paths are relative to the repo root; absolute paths pass through."""
    return Path(p) if Path(p).is_absolute() else ROOT / p


def add_override_args(ap) -> None:
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config/deploy.yaml value for this run (YAML value), e.g. --set sticks.max_xy=0.3")


def apply_overrides(cfg: Section, items) -> Section:
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set {item}: expected SECTION.KEY=VALUE")
        *path, last = key.split(".")
        node = cfg
        for k in path:
            if not isinstance(node, dict) or k not in node:
                raise ValueError(f"--set {key}: no such setting in {CONFIG.name}")
            node = node[k]
        if not isinstance(node, dict) or last not in node:
            raise ValueError(f"--set {key}: no such setting in {CONFIG.name}")
        node[last] = wrap(yaml.safe_load(raw))
    return cfg


def set_firmware_params(cfg: Section, items) -> Section:
    """--param GROUP.NAME=VALUE (RAM firmware params for this run) into drone.firmware_params."""
    for item in items:
        name, sep, value = item.partition("=")
        group, dot, key = name.partition(".")
        if not sep or not dot:
            raise ValueError(f"--param {item}: expected GROUP.NAME=VALUE")
        cfg.drone.firmware_params.setdefault(group, Section())[key] = float(value)
    return cfg


def policy_path(cfg: Section, cli_value) -> str:
    """--policy as given (relative to the current folder), else policy_flight.policy (relative to the repo root)."""
    return str(Path(cli_value).resolve() if cli_value else repo_path(cfg.policy_flight.policy))


def firmware_params(cfg: Section) -> dict:
    """drone.firmware_params as {"group.name": value}."""
    return {f"{g}.{k}": v for g, group in cfg.drone.firmware_params.items() for k, v in group.items()}
