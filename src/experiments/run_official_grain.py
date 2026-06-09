#!/usr/bin/env python3
"""Compare GRAIN-TD3 on geom-gcn splits vs official GRAIN splits.

Official GRAIN (gcn_env) uses:
  num_per_class = int(N * 0.4 / n_classes)
  num_development = int(N * 0.2)
  split seed = 0 (hardcoded in released code)

Our benchmark uses official geom-gcn 60/20/20 splits — harder but standard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
GRAIN_CODE = ROOT / "code" / "GRAIN"
sys.path.insert(0, str(ROOT))


def _load_grain_split_fn():
    """Import set_train_val_test_split from released GRAIN utils without package conflicts."""
    path = GRAIN_CODE / "utils.py"
    spec = importlib.util.spec_from_file_location("grain_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.set_train_val_test_split


def apply_official_grain_splits(data, split_seed: int = 0):
    """Match gcn_env.__init__ split protocol exactly."""
    set_train_val_test_split = _load_grain_split_fn()
    n_class = int(data.y.max().item()) + 1
    n = data.num_nodes
    num_per_class = int((n * 0.4) / n_class)
    num_development = int(n * 0.2)
    return set_train_val_test_split(split_seed, data, num_development, num_per_class)


from src.datasets.load import GRAIN_PAPER_RESULTS, load_dataset
from src.models.grain_env import GrainEnvironment
from src.train.trainer import train_grain_td3


def run_td3_on_data(data, device: torch.device, episodes: int) -> float:
    env = GrainEnvironment(data, device=device)
    r = train_grain_td3(env, rl_episodes=episodes, gnn_episodes=episodes)
    return r.test_acc


def run_geom_gcn(dataset: str, seed: int, device: torch.device, episodes: int) -> float:
    data = load_dataset(dataset, seed=seed, device=device)
    return run_td3_on_data(data, device, episodes)


def run_official_splits(dataset: str, seed: int, device: torch.device, episodes: int):
    """GRAIN-TD3 with official released-code split logic."""
    data = load_dataset(dataset, seed=0, device="cpu")  # graph only; split is independent
    data = data.cpu()
    # Official code hardcodes split seed=0; training seed affects RL init only
    data = apply_official_grain_splits(data, split_seed=0)
    data = data.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        acc = run_td3_on_data(data, device, episodes)
        return acc, None
    except Exception as exc:
        return None, str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Texas", "Actor", "Cora", "Chameleon"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Only run official-split arm; merge geom-gcn from existing CSV",
    )
    parser.add_argument("--out", default=str(ROOT / "results" / "official_grain_comparison.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)

    # Load existing geom-gcn results if official-only
    existing: dict[tuple[str, int], float] = {}
    if args.official_only and out.exists():
        prev = pd.read_csv(out)
        for _, row in prev.iterrows():
            if pd.notna(row.get("geomgcn_acc")):
                existing[(row["dataset"], int(row["seed"]))] = float(row["geomgcn_acc"])

    rows = []
    for ds in args.datasets:
        paper_num = GRAIN_PAPER_RESULTS.get(ds)
        geom_accs, official_accs = [], []

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"\n=== {ds} seed={seed} ===", flush=True)

            if args.official_only and (ds, seed) in existing:
                acc_geom = existing[(ds, seed)]
                print(f"  GRAIN-TD3 (geom-gcn splits):   {acc_geom*100:.2f}%  [cached]")
            else:
                acc_geom = run_geom_gcn(ds, seed, device, args.episodes)
                print(f"  GRAIN-TD3 (geom-gcn splits):   {acc_geom*100:.2f}%")
            geom_accs.append(acc_geom)

            acc_off, err = run_official_splits(ds, seed, device, args.episodes)
            if acc_off is not None:
                official_accs.append(acc_off)
                print(f"  GRAIN-TD3 (official splits):   {acc_off*100:.2f}%")
            else:
                print(f"  GRAIN-TD3 (official splits):   FAILED ({err})")

            rows.append({
                "dataset": ds,
                "seed": seed,
                "geomgcn_acc": acc_geom,
                "official_acc": acc_off,
            })

        print(f"\n--- {ds} summary ---")
        print(f"  geom-gcn mean:  {np.mean(geom_accs)*100:.2f}%")
        if official_accs:
            print(f"  official mean:  {np.mean(official_accs)*100:.2f}%")
            print(f"  split gap:      {(np.mean(official_accs)-np.mean(geom_accs))*100:+.2f}%")
        if paper_num:
            print(f"  paper GRAIN:    {paper_num:.2f}%")

    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    summary = {}
    for ds in args.datasets:
        sub = df[df.dataset == ds]
        off_mean = float(sub.official_acc.dropna().mean()) if sub.official_acc.notna().any() else None
        geom_mean = float(sub.geomgcn_acc.mean())
        summary[ds] = {
            "geomgcn_mean": geom_mean,
            "official_mean": off_mean,
            "paper_grain": GRAIN_PAPER_RESULTS.get(ds),
            "split_gap": (off_mean - geom_mean) if off_mean is not None else None,
        }

    out_json = out.with_name("official_grain_summary.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}, {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
