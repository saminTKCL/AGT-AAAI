#!/usr/bin/env bash
# Supplement: AGT-Implicit + H2GCN benchmark, efficiency, regenerate RESULTS.md
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs

echo "[$(date)] Supplement benchmark (AGT-Implicit + H2GCN, merge)..."
python3 src/experiments/run_benchmark.py \
  --datasets Texas Cornell Wisconsin Actor Chameleon Squirrel Cora Citeseer Pubmed \
  --methods AGT-Implicit H2GCN \
  --seeds 10 \
  --merge-existing \
  --out results/benchmark.csv \
  2>&1 | tee logs/benchmark_supplement.log

echo "[$(date)] Efficiency (10 seeds)..."
python3 src/experiments/run_efficiency.py \
  --datasets Cora Actor Chameleon Squirrel \
  --seeds 10 \
  2>&1 | tee logs/efficiency.log

echo "[$(date)] Ablations on Texas + Wisconsin (show implicit branch contribution)..."
python3 src/experiments/run_ablations.py --datasets Texas Wisconsin --seeds 10 \
  2>&1 | tee logs/ablations_webkb.log

echo "[$(date)] Analysis (k* correlation + graph structure + Chameleon/Squirrel)..."
python3 src/experiments/run_analysis.py \
  --datasets Texas Cornell Wisconsin Actor Chameleon Squirrel Cora Citeseer Pubmed \
  --seeds 3 --skip-td3 \
  2>&1 | tee logs/analysis.log

echo "[$(date)] Official GRAIN split comparison (Texas, Actor, Cora)..."
python3 src/experiments/run_official_grain.py \
  --datasets Texas Actor Cora Chameleon --seeds 5 --episodes 60 \
  2>&1 | tee logs/official_grain.log

echo "[$(date)] Generate RESULTS.md..."
python3 src/experiments/generate_results.py

echo "[$(date)] DONE"
