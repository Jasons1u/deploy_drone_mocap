#!/usr/bin/env bash
# Create (or update) the `drone_mocap` conda env and install this repo into it, editable.
#
#     ./install.sh
#     conda activate drone_mocap
#
# The env is isolated from the system Python that ROS 2 uses: it sees neither ~/.local (whose
# packages pip would otherwise modify) nor ROS's PYTHONPATH. See tools/conda_env_guard.py.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_NAME=drone_mocap

CONDA="${CONDA_EXE:-$(command -v conda || echo "$HOME/miniconda3/bin/conda")}"
[[ -x "$CONDA" ]] || { echo "conda not found (looked for $CONDA)" >&2; exit 1; }

export PYTHONNOUSERSITE=1   # pip must not see ~/.local ...
unset PYTHONPATH             # ... nor ROS 2's packages
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Updating conda env '$ENV_NAME' ..."
    "$CONDA" env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "Creating conda env '$ENV_NAME' ..."
    "$CONDA" env create -f environment.yml
fi

PREFIX="$("$CONDA" env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')"
echo "Installing this repo into $PREFIX ..."
"$PREFIX/bin/python" -m pip install -e ".[test]"

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
