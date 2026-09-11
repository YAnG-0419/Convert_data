#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Set PYTHON_BIN to any Python >=3.11 with venv/ensurepip support.
python_bin="${PYTHON_BIN:-/home/user/lpy/Pi05/.venv/bin/python}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$python_bin" -m venv "$project_root/.runtime"
"$project_root/.runtime/bin/python" -m pip install --cache-dir "$project_root/.cache/pip" -r "$project_root/requirements-lock.txt"
# The runner uses src directly, so setup needs no editable-install/build side effects.
