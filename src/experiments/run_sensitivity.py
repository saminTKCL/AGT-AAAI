#!/usr/bin/env python3
"""Hyperparameter sensitivity grid search for AGT closed-form formula.

Sweeps (alpha_h, beta_d, gamma_snr) and reports test accuracy.
Validates that the default parameters (1.5, 0.35, 0.25) are near-optimal
and the method is not sensitive to small perturbations.

Usage:
    python3 src/experiments/run_sensitivity.py --dataset Texas --seeds 3 [--fast]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.load import load_dataset
from src.models.agt import CSBMParams
from src.models.grain_env import GrainEnvironment
from src.train.trainer import train_agt_implicit

# ── Hyperparameter grid ────────────────────────────────────────────────────────
GRID = {
    "alpha_h":   [0.5, 1.0, 1.5, 2.0, 2.5],
    "beta_d":    [0.1, 0.25, 0.35, 0.5, 0.7],
    "gamma_snr": [0.0, 0.15, 0.25, 0.4, 0.6],
}

# Default (paper) values
DEFAULTS = {"alpha_h": 1.5, "beta_d": 0.35, "gamma_snr": 0.25}


def run_one(dataset: str, seed: int, device: torch.device,
            alpha_h: float, beta_d: float, gamma_snr: float,
            fast: bool) -> float:
    data = load_dataset(dataset, seed=seed, device=device)
    env  = GrainEnvironment(data, device=device, train_only=True)
    # Inject custom CSBM params by patching the stats cache
    params = CSBMParams(alpha_h=alpha_h, beta_d=beta_d, gamma_snr=gamma_snr)
    env._csbm_params_override = params  # picked up by make_agt_policy
    episodes = 40 if fast else 100
    r = train_agt_implicit(env, gnn_episodes=episodes)
    return r.test_acc


def sensitivity_1d(dataset: str, seeds: int, device: torch.device,
                   param_name: str, values: list, fast: bool) -> list[dict]:
    rows = []
    for v in values:
        kwargs = dict(DEFAULTS)
        kwargs[param_name] = v
        accs = []
        for seed in range(seeds):
            acc = run_one(dataset, seed, device, fast=fast, **kwargs)
            accs.append(acc)
        mean_acc = float(np.mean(accs)) * 100
        std_acc  = float(np.std(accs)) * 100
        is_default = (abs(v - DEFAULTS[param_name]) < 1e-6)
        rows.append({
            "param": param_name,
            "value": v,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "is_default": is_default,
        })
        tag = " ← DEFAULT" if is_default else ""
        print(f"    {param_name}={v:.2f}:  {mean_acc:.2f} ± {std_acc:.2f}{tag}",
              flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",  nargs="+", default=["Texas", "Wisconsin"])
    parser.add_argument("--seeds",     type=int, default=3)
    parser.add_argument("--fast",      action="store_true")
    parser.add_argument("--full-grid", action="store_true",
                        help="Run full 3D grid (slow) instead of 1D sweeps")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        if args.full_grid:
            # ── Full 3D grid ─────────────────────────────────────────────
            for alpha_h, beta_d, gamma_snr in product(
                    GRID["alpha_h"], GRID["beta_d"], GRID["gamma_snr"]):
                accs = []
                for seed in range(args.seeds):
                    acc = run_one(dataset, seed, device,
                                  alpha_h, beta_d, gamma_snr, args.fast)
                    accs.append(acc)
                is_def = (abs(alpha_h - 1.5) < 1e-6 and
                          abs(beta_d - 0.35) < 1e-6 and
                          abs(gamma_snr - 0.25) < 1e-6)
                row = {
                    "dataset": dataset,
                    "alpha_h": alpha_h, "beta_d": beta_d, "gamma_snr": gamma_snr,
                    "mean_acc": float(np.mean(accs)) * 100,
                    "std_acc":  float(np.std(accs))  * 100,
                    "is_default": is_def,
                }
                all_rows.append(row)
                tag = " ← DEFAULT" if is_def else ""
                print(f"  α={alpha_h} β={beta_d} γ={gamma_snr}: "
                      f"{row['mean_acc']:.2f} ± {row['std_acc']:.2f}{tag}",
                      flush=True)
        else:
            # ── 1D sweeps per parameter (much faster) ────────────────────
            for param_name in ["alpha_h", "beta_d", "gamma_snr"]:
                print(f"\n  Sweep {param_name}:")
                rows = sensitivity_1d(dataset, args.seeds, device,
                                      param_name, GRID[param_name], args.fast)
                for r in rows:
                    r["dataset"] = dataset
                all_rows.extend(rows)

    # ── Save ──────────────────────────────────────────────────────────────
    out_json = out_dir / "sensitivity.json"
    out_csv  = out_dir / "sensitivity.csv"
    out_json.write_text(json.dumps(all_rows, indent=2))
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── Summary table ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    if "param" in df.columns:
        print("\n=== Sensitivity Summary (1D sweeps) ===")
        for ds in args.datasets:
            print(f"\n{ds}:")
            for param in ["alpha_h", "beta_d", "gamma_snr"]:
                sub = df[(df.dataset == ds) & (df.param == param)]
                if len(sub) == 0:
                    continue
                best_val  = sub.loc[sub.mean_acc.idxmax(), "value"]
                def_acc   = sub[sub.is_default]["mean_acc"].values[0]
                range_acc = sub.mean_acc.max() - sub.mean_acc.min()
                print(f"  {param:12}: default={def_acc:.2f}%  "
                      f"best={best_val:.2f} ({sub.mean_acc.max():.2f}%)  "
                      f"range={range_acc:.2f}%")


if __name__ == "__main__":
    main()
