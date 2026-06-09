#!/usr/bin/env bash
# Re-run official GRAIN split comparison only (GPU 0, preserves existing geom-gcn numbers).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results

export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] Official GRAIN split comparison (official-only)..."
python3 src/experiments/run_official_grain.py \
  --datasets Texas Actor Cora Chameleon \
  --seeds 5 \
  --episodes 60 \
  --official-only \
  2>&1 | tee logs/official_grain_rerun.log

echo "[$(date)] Regenerate RESULTS.md..."
python3 src/experiments/generate_results.py

echo "[$(date)] DONE"
