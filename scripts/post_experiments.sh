#!/usr/bin/env bash
# Run after background experiments finish — regenerates RESULTS.md
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Post-experiment pipeline ==="

if [[ ! -f results/benchmark_summary.csv ]]; then
  echo "WARN: results/benchmark_summary.csv missing — benchmark may still be running."
fi

python3 src/experiments/generate_results.py

echo "=== Summary ==="
[[ -f results/benchmark_summary.csv ]] && column -t -s, results/benchmark_summary.csv | head -20
[[ -f results/efficiency_summary.csv ]] && echo && column -t -s, results/efficiency_summary.csv

echo "Done. See RESULTS.md"
