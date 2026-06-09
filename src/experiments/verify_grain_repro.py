#!/usr/bin/env python3
"""Verify GRAIN-TD3 reproduction against published AAAI 2025 numbers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.load import GRAIN_PAPER_RESULTS, load_dataset
from src.models.grain_env import GrainEnvironment
from src.train.trainer import train_grain_td3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    data = load_dataset(args.dataset, seed=args.seed, device=device)
    env = GrainEnvironment(data, device=device)
    meta = 40 if args.fast else 100
    rl = 40 if args.fast else 100
    result = train_grain_td3(env, rl_episodes=rl, gnn_episodes=meta)
    paper = GRAIN_PAPER_RESULTS.get(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"GRAIN-TD3 test accuracy: {result.test_acc*100:.2f}%")
    if paper:
        print(f"Paper GRAIN:            {paper:.2f}%")
        print(f"Gap:                    {(result.test_acc*100 - paper):+.2f}%")
    print("PASS" if result.test_acc > 0 else "FAIL")


if __name__ == "__main__":
    main()
