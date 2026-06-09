# Supplementary Material: AGT Code and Reproducibility

## Environment

```
Python 3.13, PyTorch 2.5.1+cu121, PyG 2.x
2× NVIDIA RTX 3070
pip install -r requirements.txt
```

## File Map

| File | Purpose |
|------|---------|
| `src/models/agt.py` | Closed-form $k^*$ and calibration head |
| `src/models/local_stats.py` | Per-node homophily, degree, SNR |
| `src/models/grain_env.py` | Fractional-hop GRAIN aggregation |
| `src/models/grain_td3.py` | GRAIN-TD3 baseline |
| `src/models/baselines.py` | 8 GNN baselines |
| `src/experiments/run_benchmark.py` | Main 10-seed table |
| `src/experiments/run_ablations.py` | Component ablations + plot |
| `src/experiments/run_efficiency.py` | Speed / memory comparison |
| `paper/theory_appendix.md` | Full CSBM proof |

## Hyperparameters

- Classifier: 64-d hidden, dropout 0.5, Adam lr=0.01, wd=5e-4
- AGT: $\alpha=1.5, \beta=0.35, \gamma=0.25, k_{\min}=0.2, k_{\max}=5$
- Meta-training: 50 epochs (25 warm + 25 main), calibration 80 steps
- GRAIN-TD3: 80 RL steps + 50 meta epochs
- Baselines: 200 epochs, patience 50

## Expected Runtime

Full reproduction (`run_all.sh`): ~2–4 hours on 2× RTX 3070.
