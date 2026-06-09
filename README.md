# AGT: Analytic Granularity Transform for Heterophilous GNNs

> **Replaces GRAIN's TD3 RL with a theory-grounded closed-form per-node granularity map.**  
> Target venue: **AAAI-27** (abstract Jul 21, 2026; full paper Jul 28, 2026)

---

## Overview

GRAIN (Zhao et al., AAAI 2025) learns per-node aggregation depth via TD3 reinforcement learning. **AGT (Analytic Granularity Transform)** shows this decision is a signal-to-noise tradeoff under a degree-corrected CSBM, with a **closed-form optimal hop depth** $k^*(h_v, d_v)$ — eliminating RL, replay buffers, and critics entirely.

| | GRAIN-TD3 | AGT |
|---|-----------|-----|
| Policy | TD3 actor + twin critics + replay | Closed-form map + optional tiny calibrator |
| Theory | None | CSBM optimality proof |
| Training | RL meta-loop | End-to-end backprop |
| Interpretability | Black-box | $k^*$ vs local homophily |

---

## Project Structure

```
src/
  datasets/          # geom-gcn split loading (9 datasets)
  models/
    agt.py           # Closed-form granularity + calibration
    grain_env.py     # GRAIN fractional-hop environment
    grain_td3.py     # TD3 baseline + ablation policies
    baselines.py     # GCN, H2GCN, GPR-GNN, ACM-GNN, GloGNN, FAGCN, Ordered-GNN
    local_stats.py   # Per-node homophily, degree, SNR
  theory/            # (see paper/theory_appendix.md)
  train/trainer.py   # AGT and GRAIN-TD3 training loops
  experiments/       # Benchmark, ablation, efficiency scripts
paper/
  main.tex           # AAAI paper draft
  theory_appendix.md # CSBM proof
tests/
  test_agt.py        # Unit tests
```

Legacy `extension/` (GRAIN-CTRL distillation) is **deprecated** — see [IDEA.md](IDEA.md).

---

## Installation

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q
```

Requires PyTorch + PyG. Tested on Python 3.13, PyTorch 2.5, CUDA 12.1 (2× RTX 3070).

---

## Reproduce Results

```bash
# Full benchmark (9 datasets × 10 seeds × 8 methods)
python3 src/experiments/run_benchmark.py --seeds 10

# Ablations + granularity–homophily plot
python3 src/experiments/run_ablations.py --dataset Actor --seeds 10

# Efficiency (wall-clock, params, GPU mem)
python3 src/experiments/run_efficiency.py --seeds 10

# Regenerate RESULTS.md
python3 src/experiments/generate_results.py
```

Use `--fast` for smoke tests (reduced epochs).

---

## Key Results

See [RESULTS.md](RESULTS.md) for full tables. Summary after 10-seed runs:

- **AGT matches or exceeds GRAIN-TD3** on heterophily benchmarks
- **2–4× training speedup** vs GRAIN-TD3 (no RL overhead)
- **Granularity–homophily correlation r > 0.6** validates CSBM theory
- Ablation: closed-form map > fixed-k > random; calibration adds ~1%

---

## Theory (Summary)

Under DC-CSBM, per-node SNR after $k$-hop aggregation is unimodal in $k$. The maximizer satisfies:

$$k^*(v) = k_{\min} + (k_{\max}-k_{\min})\,\tilde{h}_v^{\alpha}(1+\beta\log(1+d_v))\,\widetilde{\mathrm{SNR}}_v^{\gamma}$$

Full proof: [paper/theory_appendix.md](paper/theory_appendix.md)

---

## Citation

```bibtex
@inproceedings{zhao2025grain,
  title={GRAIN: Multi-Granular and Implicit Information Aggregation Graph Neural Network for Heterophilous Graphs},
  author={Zhao, Songwei and Jiang, Yuan and Zhang, Zijing and Yu, Yang and Chen, Hechang},
  booktitle={AAAI},
  year={2025}
}
```

---

## Publication Checklist

- [x] Clean `src/` implementation (AGT + GRAIN-TD3 + baselines)
- [x] CSBM theory appendix
- [x] 10-seed benchmark harness
- [x] Ablation + efficiency analysis
- [x] Paper draft (`paper/main.tex`)
- [x] Unit tests
- [ ] Camera-ready polish for AAAI-27 submission
