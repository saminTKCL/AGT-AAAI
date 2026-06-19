#!/usr/bin/env python3
"""Ablation study for AGT components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.load import load_dataset
from src.models.grain_env import GrainEnvironment
from src.models.grain_td3 import FixedGranularityPolicy, RandomGranularityPolicy, TD3Policy
from src.models.agt import AnalyticGranularityPolicy
from src.train.trainer import train_agt, train_agt_implicit, train_grain_td3, train_with_policy


def run_ablations(dataset: str, seed: int, device: torch.device, fast: bool):
    data = load_dataset(dataset, seed=seed, device=device)
    env = GrainEnvironment(data, device=device, train_only=True)
    meta = 40 if fast else 100
    rows = []

    r = train_agt(env, use_calibration=False, gnn_episodes=meta)
    rows.append({"variant": "closed_form", **r.__dict__, "dataset": dataset, "seed": seed})

    env.reset_model()
    r = train_agt(env, use_calibration=True, gnn_episodes=meta)
    rows.append({"variant": "closed+calib", **r.__dict__, "dataset": dataset, "seed": seed})

    env.reset_model()
    r = train_agt_implicit(env, gnn_episodes=meta)
    rows.append({"variant": "agt_implicit", **r.__dict__, "dataset": dataset, "seed": seed})

    # Leaky variant for leakage audit comparison
    leaky_env = GrainEnvironment(data, device=device, train_only=False)
    r = train_agt_implicit(leaky_env, gnn_episodes=meta)
    rows.append({"variant": "agt_implicit_leaky", **r.__dict__, "dataset": dataset, "seed": seed})

    env.reset_model()
    r = train_with_policy(env, RandomGranularityPolicy(seed=seed), gnn_episodes=meta)
    rows.append({"variant": "random", **r.__dict__, "dataset": dataset, "seed": seed})

    env.reset_model()
    r = train_with_policy(env, FixedGranularityPolicy(k=2.0), gnn_episodes=meta)
    rows.append({"variant": "fixed_k2", **r.__dict__, "dataset": dataset, "seed": seed})

    env.reset_model()
    r = train_grain_td3(env, rl_episodes=meta, gnn_episodes=meta)
    rows.append({"variant": "td3", **r.__dict__, "dataset": dataset, "seed": seed})

    return rows


def plot_granularity_homophily(dataset: str, seed: int, device: torch.device, out_dir: Path):
    data = load_dataset(dataset, seed=seed, device=device)
    env = GrainEnvironment(data, device=device)
    policy = env.make_agt_policy(use_calibration=False)
    k = policy.granularity.cpu().numpy()
    h = policy.stats["homophily"].cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.scatter(h, k, alpha=0.5, s=8)
    plt.xlabel("Local homophily $h_v$")
    plt.ylabel("Analytic granularity $k^*_v$")
    plt.title(f"{dataset} (seed {seed})")
    plt.tight_layout()
    plt.savefig(out_dir / f"granularity_homophily_{dataset}_seed{seed}.png", dpi=150)
    plt.close()
    corr = float(np.corrcoef(h, k)[0, 1])
    return corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Single dataset (legacy)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="One or more datasets (overrides --dataset)")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    # Support both --dataset and --datasets
    datasets = args.datasets if args.datasets else ([args.dataset] if args.dataset else ["Actor"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Ablation: {dataset}")
        print('='*50)
        all_rows = []
        corrs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            all_rows.extend(run_ablations(dataset, seed, device, args.fast))
            corrs.append(plot_granularity_homophily(dataset, seed, device, out_dir))

        df = pd.DataFrame(all_rows)
        df.to_csv(out_dir / f"ablation_{dataset}.csv", index=False)
        summary = df.groupby("variant").agg(
            test_mean=("test_acc", "mean"),
            test_std=("test_acc", "std"),
            time_mean=("train_time", "mean"),
        )
        summary.to_csv(out_dir / f"ablation_{dataset}_summary.csv")
        (out_dir / f"granularity_homophily_corr_{dataset}.json").write_text(
            json.dumps({"mean_corr": float(np.mean(corrs)), "corrs": corrs}, indent=2)
        )
        print(summary)


if __name__ == "__main__":
    main()
