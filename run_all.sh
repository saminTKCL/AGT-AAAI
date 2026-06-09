#!/usr/bin/env bash
# Reproduce all AGT experiments for AAAI-27 submission
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/ablations

echo "=== Unit tests ==="
python3 -m pytest tests/ -q

echo "=== GRAIN-TD3 reproduction check ==="
python3 src/experiments/verify_grain_repro.py --dataset Cora --seed 0 --fast
python3 src/experiments/verify_grain_repro.py --dataset Chameleon --seed 0 --fast

echo "=== Full benchmark (10 seeds) ==="
python3 src/experiments/run_benchmark.py \
  --datasets Texas Cornell Wisconsin Actor Chameleon Squirrel Cora Citeseer Pubmed \
  --methods AGT GRAIN-TD3 GCN H2GCN GPR-GNN ACM-GNN GloGNN FAGCN Ordered-GNN \
  --seeds 10 \
  --out results/benchmark.csv

echo "=== Ablations ==="
python3 src/experiments/run_ablations.py --dataset Actor --seeds 10

echo "=== Efficiency ==="
python3 src/experiments/run_efficiency.py --seeds 10

echo "=== Generate RESULTS.md ==="
python3 src/experiments/generate_results.py

echo "Done."
