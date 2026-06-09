#!/usr/bin/env bash
# Run only the NEW analysis stages — does NOT touch benchmark or efficiency CSVs.
# Uses GPU 0 only to avoid conflicting with other projects on GPU 1.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/ablations results/analysis logs

export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] Starting remaining stages (GPU 0 only)..."
echo "[$(date)] NOTE: benchmark.csv and efficiency.csv are NOT modified."

echo "[$(date)] Analysis (k* correlation + graph structure)..."
python3 src/experiments/run_analysis.py \
  --datasets Texas Cornell Wisconsin Actor Chameleon Squirrel Cora Citeseer Pubmed \
  --seeds 3 --skip-td3 \
  2>&1 | tee logs/analysis.log

echo "[$(date)] Ablations on Texas + Wisconsin..."
python3 src/experiments/run_ablations.py --datasets Texas Wisconsin --seeds 10 \
  2>&1 | tee logs/ablations_webkb.log

echo "[$(date)] Official GRAIN split comparison..."
python3 src/experiments/run_official_grain.py \
  --datasets Texas Actor Cora Chameleon --seeds 5 --episodes 60 \
  2>&1 | tee logs/official_grain.log

echo "[$(date)] Generate RESULTS.md..."
python3 src/experiments/generate_results.py

echo "[$(date)] DONE — all remaining stages complete."
