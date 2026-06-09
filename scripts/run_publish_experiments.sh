#!/usr/bin/env bash
# Full publication experiments on REAL geom-gcn splits (Pei et al.)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/ablations logs

echo "[$(date)] Verifying real splits..."
python3 -c "from src.datasets.load import verify_all_splits; m=verify_all_splits(); assert not m, m; print('OK: 90 real geom-gcn splits')"

echo "[$(date)] Benchmark (9 datasets × 10 seeds)..."
python3 src/experiments/run_benchmark.py \
  --datasets Texas Cornell Wisconsin Actor Chameleon Squirrel Cora Citeseer Pubmed \
  --methods AGT AGT-Implicit GRAIN-TD3 GCN H2GCN GPR-GNN ACM-GNN GloGNN FAGCN Ordered-GNN \
  --seeds 10 \
  --out results/benchmark.csv \
  2>&1 | tee logs/benchmark.log

echo "[$(date)] Ablations..."
python3 src/experiments/run_ablations.py --dataset Actor --seeds 10 \
  2>&1 | tee logs/ablations.log

echo "[$(date)] Efficiency..."
python3 src/experiments/run_efficiency.py --datasets Cora Actor Chameleon Squirrel --seeds 10 \
  2>&1 | tee logs/efficiency.log

echo "[$(date)] Generate RESULTS.md..."
python3 src/experiments/generate_results.py

echo "[$(date)] DONE"
