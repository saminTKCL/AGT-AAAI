#!/usr/bin/env python3
"""Go/no-go: AGT vs AGT-Implicit vs GRAIN-TD3 on Texas + Actor (real geom-gcn splits)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.load import GRAIN_PAPER_RESULTS, load_dataset
from src.models.grain_env import GrainEnvironment
from src.train.trainer import MethodResult, train_agt, train_agt_implicit, train_grain_td3


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_method(method: str, dataset: str, seed: int, device: torch.device, episodes: int) -> MethodResult:
    seed_all(seed)
    data = load_dataset(dataset, seed=seed, device=device)
    env = GrainEnvironment(data, device=device)
    if method == "AGT":
        r = train_agt(env, use_calibration=True, gnn_episodes=episodes)
    elif method == "AGT-Implicit":
        r = train_agt_implicit(env, gnn_episodes=episodes, implicit_k=10)
    elif method == "GRAIN-TD3":
        r = train_grain_td3(env, rl_episodes=episodes, gnn_episodes=episodes)
    else:
        raise ValueError(method)
    r.method = method
    r.dataset = dataset
    r.seed = seed
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Texas", "Actor"])
    parser.add_argument("--methods", nargs="+", default=["AGT", "AGT-Implicit", "GRAIN-TD3"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=40, help="RL+GNN episodes (use 100 for full run)")
    parser.add_argument("--out", default=str(ROOT / "results" / "gonogo.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[MethodResult] = []
    for ds in args.datasets:
        for method in args.methods:
            for seed in range(args.seeds):
                print(f"=== {method} | {ds} | seed={seed} ===", flush=True)
                r = run_method(method, ds, seed, device, args.episodes)
                rows.append(r)
                paper = GRAIN_PAPER_RESULTS.get(ds, 0)
                print(
                    f"  test={r.test_acc*100:.2f}% val={r.val_acc*100:.2f}% "
                    f"time={r.train_time:.1f}s paper={paper:.1f}%"
                )

    df = pd.DataFrame([r.__dict__ for r in rows])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = df.groupby(["dataset", "method"]).agg(
        test_mean=("test_acc", lambda x: x.mean() * 100),
        test_std=("test_acc", "std"),
        time_mean=("train_time", "mean"),
    )
    print("\n=== Summary (test accuracy %) ===")
    print(summary.to_string())
    summary.to_csv(out.with_name("gonogo_summary.csv"))

    # Decision hint
    for ds in args.datasets:
        sub = summary.loc[ds] if ds in summary.index.get_level_values(0) else None
        if sub is None:
            continue
        if "AGT-Implicit" in sub.index and "GRAIN-TD3" in sub.index:
            imp = sub.loc["AGT-Implicit", "test_mean"]
            td3 = sub.loc["GRAIN-TD3", "test_mean"]
            delta = imp - td3
            verdict = "GO" if delta >= 1.0 else ("MAYBE" if delta >= 0 else "NO-GO")
            print(f"\n{ds}: AGT-Implicit {imp:.2f}% vs TD3 {td3:.2f}% (Δ {delta:+.2f}%) → {verdict}")

    (out.with_name("gonogo_summary.json")).write_text(summary.to_json(indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
