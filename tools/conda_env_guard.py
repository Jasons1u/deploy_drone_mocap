"""setuptools build backend that only installs into an isolated, non-base conda environment.

Why: the Python dependencies here (latest cflib -> numpy 2.x) must never reach the system Python
that ROS 2 Humble uses — its compiled message packages (px4_msgs, ...) break under numpy 2. Two
ways that happens, both refused here:

  1. pip running on the system Python (or conda base): installs into ~/.local or the base env.
  2. pip running in a conda python 3.10 that can still see ~/.local/lib/python3.10 (the user site
     is per Python version, shared with the system Python). pip treats those packages as
     installed and UNINSTALLS them from ~/.local when it upgrades one (numpy, packaging, ...).
     Safe only if that pip process was started with PYTHONNOUSERSITE set.

pip sets PYTHONNOUSERSITE=1 inside its own isolated build env, so for (2) this hook cannot trust
its own environment; it reads the environment the parent pip process started with instead
(/proc/<ppid>/environ, Linux). The hooks run before pip resolves or installs any dependency, so
refusing here changes nothing.

Supported install:  ./install.sh      (creates the `drone_mocap` env, then pip install -e .)
Everything else is plain setuptools.
"""

import os
import sys

from setuptools import build_meta as _setuptools
from setuptools.build_meta import *  # noqa: F401,F403  (re-export the other PEP 517 hooks)

_HOW = """
    ./install.sh                  # creates (or updates) the `drone_mocap` env and installs this repo
    conda activate drone_mocap    # afterwards, `pip install -e ".[test]"` here is safe too
"""


def _parent_env() -> dict:
    try:
        with open(f"/proc/{os.getppid()}/environ", "rb") as f:
            entries = f.read().split(b"\0")
    except OSError:
        raise SystemExit("cannot verify pip's environment (/proc unavailable) — use ./install.sh")
    return dict(e.decode(errors="replace").split("=", 1) for e in entries if b"=" in e)


def _check_target() -> None:
    # PEP 517 hooks run under the interpreter pip installs into, so sys.prefix is the target.
    is_conda = os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
    is_base = os.path.isdir(os.path.join(sys.prefix, "envs"))
    if not is_conda or is_base:
        where = "the conda base env" if is_conda else f"the non-conda Python at {sys.prefix}"
        raise SystemExit(f"\ndeploy-drone-mocap installs only into its own conda env, not {where}.\n{_HOW}")
    if not _parent_env().get("PYTHONNOUSERSITE"):
        raise SystemExit(
            "\npip was started without PYTHONNOUSERSITE, so it can see (and would modify) the "
            "packages in ~/.local that your system Python / ROS 2 use.\n" + _HOW)


def get_requires_for_build_editable(config_settings=None):
    _check_target()
    return _setuptools.get_requires_for_build_editable(config_settings)


def get_requires_for_build_wheel(config_settings=None):
    _check_target()
    return _setuptools.get_requires_for_build_wheel(config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _check_target()
    return _setuptools.build_editable(wheel_directory, config_settings, metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _check_target()
    return _setuptools.build_wheel(wheel_directory, config_settings, metadata_directory)
