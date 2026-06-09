#!/usr/bin/env python3
"""Efficiency comparison: AGT vs GRAIN-TD3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.load import DATASETS, load_dataset
from src.models.grain_env import GrainEnvironment
from src.train.trainer import train_agt, train_agt_implicit, train_grain_td3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Cora", "Actor", "Chameleon", "Squirrel"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    meta = 40 if args.fast else 100
    for ds in args.datasets:
        for seed in range(args.seeds):
            data = load_dataset(ds, seed=seed, device=device)
            env = GrainEnvironment(data, device=device)
            agt = train_agt(env, gnn_episodes=meta)
            env.reset_model()
            implicit = train_agt_implicit(env, gnn_episodes=meta)
            env.reset_model()
            td3 = train_grain_td3(env, rl_episodes=meta, gnn_episodes=meta)
            agt_speedup = td3.train_time / max(agt.train_time, 1e-6)
            implicit_speedup = td3.train_time / max(implicit.train_time, 1e-6)
            rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "agt_time": agt.train_time,
                    "implicit_time": implicit.train_time,
                    "td3_time": td3.train_time,
                    "agt_speedup": agt_speedup,
                    "implicit_speedup": implicit_speedup,
                    "agt_params": agt.params,
                    "implicit_params": implicit.params,
                    "td3_params": td3.params,
                    "agt_mem_mb": agt.peak_mem_mb,
                    "implicit_mem_mb": implicit.peak_mem_mb,
                    "td3_mem_mb": td3.peak_mem_mb,
                }
            )
            print(
                f"{ds} seed={seed}: AGT {agt_speedup:.2f}x, "
                f"AGT-Implicit {implicit_speedup:.2f}x vs TD3"
            )

    out = ROOT / "results" / "efficiency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summary = df.groupby("dataset").mean(numeric_only=True)
    summary.to_csv(out.with_name("efficiency_summary.csv"))
    (out.with_name("efficiency_summary.json")).write_text(summary.to_json(indent=2))
    print(summary[["agt_speedup", "implicit_speedup", "agt_time", "implicit_time", "td3_time"]])


if __name__ == "__main__":
    main()
