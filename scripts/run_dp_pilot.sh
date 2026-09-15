#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CONVERT_ROOT="$project_root"
export DP_ROOT="${DP_ROOT:-$project_root/../diffusion_policy}"
export DP_DATASET="$project_root/outputs/dp/pilot_high_30hz"
export NUMBA_CACHE_DIR="$project_root/.cache/numba" MPLCONFIGDIR="$project_root/.cache/matplotlib"
export WANDB_CACHE_DIR="$project_root/.cache/wandb" XDG_CACHE_HOME="$project_root/.cache"
export CUDA_CACHE_PATH="$project_root/.cache/cuda" TORCH_HOME="$project_root/.cache/torch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
pilot_python="${DP_PYTHON:-/home/descfly/miniforge3/envs/robodiff5090/bin/python}"
pilot_workers="${DP_WORKERS:-4}"
artifacts="$project_root/outputs/dp_pilot_20260915"
mkdir -p "$artifacts"
if [[ "${1:-}" != "--train-only" ]]; then
  .runtime/bin/python -B scripts/select_dp_pilot.py | tee "$artifacts/selection.log"
  scripts/dp_batch --episode-list lists/dp_pilot_all.txt --workers "$pilot_workers" --cache-source \
    --output "$DP_DATASET" | tee "$artifacts/conversion.log"
else
  shift
fi
# A previous interrupted conversion must not be used for training.
if [[ -f "$DP_DATASET/verification.json" ]]; then
  .runtime/bin/python -B - <<'CHECK'
import json, os
from pathlib import Path
root=Path(os.environ['DP_DATASET'])
c=json.loads((root/'conversion_report.json').read_text())
v=json.loads((root/'verification.json').read_text())
if c['errors'] or not v['ok'] or c['frames']!=v['frames'] or c['bags']!=v['bags']:
    raise SystemExit('Verification does not match the completed conversion')
CHECK
  cp "$DP_DATASET/verification.json" "$artifacts/verification.json"
else
  scripts/dp verify "$DP_DATASET" > "$artifacts/verification.json"
fi
.runtime/bin/python -B scripts/preview_dp_pilot.py --video | tee "$artifacts/preview.log"
if [[ ! -f "$project_root/outputs/dp_training/pilot_high_30hz/training_complete.json" ]]; then
  PYTHONPATH="$project_root/src:$DP_ROOT" "$pilot_python" -B scripts/train_dp.py \
    --config-name train_franka_pilot "$@" |& tee "$artifacts/training.log"
fi
PYTHONPATH="$project_root/src:$DP_ROOT" "$pilot_python" -B scripts/evaluate_dp_pilot.py \
  |& tee "$artifacts/evaluation.log"
.runtime/bin/python -B scripts/render_dp_pilot.py | tee "$artifacts/render.log"
