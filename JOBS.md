# Background experiment jobs (REAL geom-gcn data)

## Critical fix applied

Previous results used **synthetic fallback splits** for Texas, Cornell, Wisconsin, Cora, Citeseer, Pubmed — not the official Pei et al. geom-gcn masks. That invalidated all prior benchmark numbers.

**Now fixed:**
- 90 official split files in `data/geom_splits/` (downloaded from graphdml-uiuc-jlu/geom-gcn)
- `load_dataset()` **raises** if real split missing — no synthetic fallback
- Training aligned with official GRAIN protocol (100 RL + 100 GNN episodes)

## Running now

```bash
tail -f logs/publish_master.log
ps aux | grep run_publish_experiments
```

Pipeline: `scripts/run_publish_experiments.sh` (benchmark → ablations → efficiency → RESULTS.md)

## Verify real data

```bash
python3 -c "from src.datasets.load import load_dataset; d=load_dataset('Texas',0); print(d.split_source)"
# → data/geom_splits/texas_split_0.6_0.2_0.npz
```

## When done

Ask agent to finalize `paper/main.tex` numbers from `results/benchmark_summary.csv`.
