#!/usr/bin/env python3
"""Multi-seed benchmark: AGT vs GRAIN-TD3 vs baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.load import DATASETS, GRAIN_PAPER_RESULTS, load_dataset
from src.models.baselines import BASELINE_REGISTRY, train_baseline
from src.models.grain_env import GrainEnvironment
from src.train.trainer import MethodResult, train_agt, train_agt_implicit, train_agt_implicit_adaptive, train_grain_td3


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single(dataset: str, method: str, seed: int, device: torch.device, fast: bool) -> MethodResult:
    seed_all(seed)
    data = load_dataset(dataset, seed=seed, device=device)
    if method in BASELINE_REGISTRY:
        r = train_baseline(data, method, epochs=120 if fast else 200, device=device)
        return MethodResult(method, dataset, seed, r.test_acc, r.best_val, r.train_time, r.params)
    env = GrainEnvironment(data, device=device)
    meta = 40 if fast else 100
    if method == "AGT":
        r = train_agt(env, use_calibration=True, gnn_episodes=meta)
    elif method == "AGT-closed":
        r = train_agt(env, use_calibration=False, gnn_episodes=meta)
    elif method == "AGT-Implicit":
        r = train_agt_implicit(env, gnn_episodes=meta)
    elif method == "AGT-Implicit-Adaptive":
        r = train_agt_implicit_adaptive(env, gnn_episodes=meta)
    elif method == "GRAIN-TD3":
        r = train_grain_td3(env, rl_episodes=meta, gnn_episodes=meta)
    else:
        raise ValueError(method)
    r.dataset = dataset
    r.seed = seed
    r.method = method
    return r


def aggregate(rows: list[MethodResult]) -> pd.DataFrame:
    recs = [
        {
            "method": r.method,
            "dataset": r.dataset,
            "seed": r.seed,
            "test_acc": r.test_acc,
            "val_acc": r.val_acc,
            "train_time": r.train_time,
            "params": r.params,
            "peak_mem_mb": r.peak_mem_mb,
        }
        for r in rows
    ]
    return pd.DataFrame(recs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Cora", "Actor", "Chameleon"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["AGT", "GRAIN-TD3", "GCN", "H2GCN", "GPR-GNN"],
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--fast", action="store_true", help="Reduced epochs for smoke tests")
    parser.add_argument("--out", default=str(ROOT / "results" / "benchmark.csv"))
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge with existing CSV, replacing rows for methods in --methods",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[MethodResult] = []
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"Skip unknown dataset {ds}")
            continue
        for method in args.methods:
            for seed in range(args.seeds):
                print(f"Running {method} on {ds} seed={seed} ...", flush=True)
                try:
                    r = run_single(ds, method, seed, device, args.fast)
                    rows.append(r)
                    print(f"  test_acc={r.test_acc:.4f} time={r.train_time:.2f}s")
                except Exception as exc:
                    print(f"  FAILED: {exc}")

    df = aggregate(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.merge_existing and out.exists():
        prev = pd.read_csv(out)
        prev = prev[~prev["method"].isin(args.methods)]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["dataset", "method"])
        .agg(
            test_mean=("test_acc", "mean"),
            test_std=("test_acc", "std"),
            time_mean=("train_time", "mean"),
            params_mean=("params", "mean"),
        )
        .reset_index()
    )
    summary_path = out.with_name("benchmark_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Wilcoxon paired tests vs GRAIN-TD3 per dataset
    sig_rows = []
    for ds in sorted(df["dataset"].unique()):
        td3 = df[(df.dataset == ds) & (df.method == "GRAIN-TD3")]["test_acc"].values
        entry = {"dataset": ds, "td3_mean": td3.mean() if len(td3) else None}
        for method in ("AGT", "AGT-Implicit"):
            a = df[(df.dataset == ds) & (df.method == method)]["test_acc"].values
            if len(a) >= 3 and len(a) == len(td3):
                _, p = wilcoxon(a, td3)
                entry[f"{method}_mean"] = float(a.mean())
                entry[f"{method}_wilcoxon_p"] = float(p)
        if len(entry) > 2:
            sig_rows.append(entry)
    sig_path = out.with_name("significance.json")
    sig_path.write_text(json.dumps(sig_rows, indent=2))

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved {out}, {summary_path}, {sig_path}")


if __name__ == "__main__":
    main()
