#!/usr/bin/env python3
"""Leakage audit: compare leaky (oracle homophily) vs leakage-free (pseudo-homophily).

Runs AGT-Implicit with train_only=True vs train_only=False on three datasets
and reports accuracy difference and k* correlation.

Usage:
    python3 src/experiments/verify_leakage.py --datasets Texas Wisconsin Actor --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.load import load_dataset
from src.models.grain_env import GrainEnvironment
from src.models.local_stats import local_homophily, estimate_pseudo_homophily
from src.train.trainer import train_agt_implicit


def verify_one(dataset: str, seed: int, device: torch.device, fast: bool) -> dict:
    episodes = 40 if fast else 100
    data = load_dataset(dataset, seed=seed, device=device)

    # ── Leakage-free (default) ─────────────────────────────────────────────
    env_free = GrainEnvironment(data, device=device, train_only=True)
    r_free = train_agt_implicit(env_free, gnn_episodes=episodes)

    # ── Leaky (oracle labels for homophily) ────────────────────────────────
    env_free.reset_model()
    env_leaky = GrainEnvironment(data, device=device, train_only=False)
    r_leaky = train_agt_implicit(env_leaky, gnn_episodes=episodes)

    # ── k* correlation between leaky and leakage-free ─────────────────────
    with torch.no_grad():
        k_free  = env_free.make_agt_policy().granularity.cpu().numpy()
        k_leaky = env_leaky.make_agt_policy().granularity.cpu().numpy()
        corr = float(np.corrcoef(k_free, k_leaky)[0, 1])

    # ── Homophily estimation accuracy ─────────────────────────────────────
    hom_oracle = local_homophily(data).cpu().numpy()
    hom_pseudo = estimate_pseudo_homophily(data).cpu().numpy()
    hom_corr = float(np.corrcoef(hom_oracle, hom_pseudo)[0, 1])
    hom_mae  = float(np.mean(np.abs(hom_oracle - hom_pseudo)))

    return {
        "dataset": dataset,
        "seed": seed,
        "free_acc":   r_free.test_acc,
        "leaky_acc":  r_leaky.test_acc,
        "delta_acc":  r_free.test_acc - r_leaky.test_acc,
        "kstar_corr": corr,
        "hom_corr":   hom_corr,
        "hom_mae":    hom_mae,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Texas", "Wisconsin", "Actor"])
    parser.add_argument("--seeds",    type=int, default=3)
    parser.add_argument("--fast",     action="store_true", help="40 episodes instead of 100")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results" / "leakage"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    print(f"\n{'Dataset':12} {'Seed':4} {'Free':7} {'Leaky':7} {'Delta':7} "
          f"{'k*-corr':8} {'h-corr':8} {'h-MAE':7}")
    print("-" * 70)

    for ds in args.datasets:
        ds_rows = []
        for seed in range(args.seeds):
            print(f"  {ds} seed={seed}...", flush=True)
            row = verify_one(ds, seed, device, args.fast)
            ds_rows.append(row)
            all_rows.append(row)
            print(
                f"  {row['dataset']:12} {row['seed']:4d} "
                f"  {row['free_acc']*100:5.2f}  {row['leaky_acc']*100:5.2f}  "
                f"{row['delta_acc']*100:+5.2f}  "
                f"{row['kstar_corr']:7.3f}  {row['hom_corr']:7.3f}  "
                f"{row['hom_mae']:6.3f}",
                flush=True,
            )

        # ── Per-dataset summary ────────────────────────────────────────────
        free_mean  = np.mean([r["free_acc"]   for r in ds_rows]) * 100
        leaky_mean = np.mean([r["leaky_acc"]  for r in ds_rows]) * 100
        delta_mean = np.mean([r["delta_acc"]  for r in ds_rows]) * 100
        kc_mean    = np.mean([r["kstar_corr"] for r in ds_rows])
        hc_mean    = np.mean([r["hom_corr"]   for r in ds_rows])
        hm_mean    = np.mean([r["hom_mae"]    for r in ds_rows])
        print(f"  {'MEAN':12}      "
              f"  {free_mean:5.2f}  {leaky_mean:5.2f}  {delta_mean:+5.2f}  "
              f"{kc_mean:7.3f}  {hc_mean:7.3f}  {hm_mean:6.3f}")
        print()

    (out_dir / "leakage_audit.json").write_text(json.dumps(all_rows, indent=2))
    print(f"Saved to {out_dir / 'leakage_audit.json'}")

    # ── Final verdict ──────────────────────────────────────────────────────
    global_delta = np.mean([r["delta_acc"] for r in all_rows]) * 100
    global_kc    = np.mean([r["kstar_corr"] for r in all_rows])
    global_hc    = np.mean([r["hom_corr"]   for r in all_rows])
    print("\n=== LEAKAGE AUDIT VERDICT ===")
    print(f"  Mean Δ acc (free − leaky): {global_delta:+.2f}%")
    print(f"  Mean k* correlation:       {global_kc:.3f}")
    print(f"  Mean homophily estimation corr: {global_hc:.3f}")
    if abs(global_delta) < 1.0 and global_kc > 0.9:
        print("  ✓ PASS: leakage-free and leaky are equivalent (Δ < 1%, r > 0.9)")
    elif abs(global_delta) < 2.0:
        print("  ~ MARGINAL: small delta — leakage-free is safe to use")
    else:
        print("  ✗ WARNING: notable accuracy gap — investigate further")


if __name__ == "__main__":
    main()
