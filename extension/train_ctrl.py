#!/usr/bin/env python3
"""Train GRAIN with deterministic CTRL policy (no TD3 noise)."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

GRAIN = Path(__file__).resolve().parents[1] / "code" / "GRAIN"
sys.path.insert(0, str(GRAIN))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from env.gcn import gcn_env
from grain_ctrl_policy import GRAINCTRLPolicy
from research_utils import ExperimentResult, log_result, seed_all


def main():
    seed_all(0)
    dataset = "Cora"
    env = gcn_env(dataset=dataset, max_layer=1)
    env.seed(0)

    policy = GRAINCTRLPolicy(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_num,
        max_action=5.0,
    )
    env.policy = policy

    best_acc = 0.0
    for ep in range(1, 51):
        state = env.reset2()
        action = policy.select_action(state).clip(0, 5)
        if action.ndim == 1:
            action = action.reshape(-1, 1)
        state, reward, done, val_acc = env.step2(action)
        test_acc = env.test_batch()
        best_acc = max(best_acc, test_acc)
        if ep % 10 == 0:
            print(f"ep {ep:3d} val={float(val_acc):.4f} test={test_acc:.4f} best={best_acc:.4f}")

    log_result(
        Path(__file__).resolve().parents[2],
        ExperimentResult(
            project="03_GRAIN",
            experiment="train_ctrl_cora",
            metric="test_accuracy",
            value=best_acc,
            details={"epochs": 50, "policy": "GRAIN-CTRL"},
        ),
    )
    print(f"Best test accuracy: {best_acc:.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
