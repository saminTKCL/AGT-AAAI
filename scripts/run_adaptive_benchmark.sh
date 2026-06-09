#!/usr/bin/env bash
# Run AGT-Implicit-Adaptive benchmark (merges into existing benchmark.csv)
# Uses GPU 0. Does not touch any other results.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results

export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] AGT-Implicit-Adaptive benchmark (9 datasets, 10 seeds)..."
python3 src/experiments/run_benchmark.py \
  --datasets Texas Cornell Wisconsin Actor Chameleon Squirrel Cora Citeseer Pubmed \
  --methods AGT-Implicit-Adaptive \
  --seeds 10 \
  --merge-existing \
  --out results/benchmark.csv \
  2>&1 | tee logs/benchmark_adaptive.log

echo "[$(date)] k* vs TD3 correlation (Texas, Chameleon, Cora — 1 seed each)..."
python3 src/experiments/run_analysis.py \
  --datasets Texas Chameleon Cora Wisconsin Squirrel \
  --seeds 1 \
  --td3-datasets Texas Chameleon Cora \
  --td3-episodes 60 \
  2>&1 | tee logs/analysis_kstar_td3.log

echo "[$(date)] Regenerate RESULTS.md..."
python3 src/experiments/generate_results.py

echo "[$(date)] DONE"
