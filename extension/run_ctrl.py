#!/usr/bin/env python3
"""Smoke test GRAIN-CTRL controller."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_controller import DeterministicGranularityController


def main():
    torch.manual_seed(0)
    state_dim, n_actions, batch = 32, 6, 16
    ctrl = DeterministicGranularityController(state_dim, n_actions)
    state = torch.randn(batch, state_dim)

    ctrl.train()
    a_train, u, stop = ctrl(state, temperature=0.7)
    ctrl.eval()
    a_eval, u_eval, stop_eval = ctrl(state, hard=True)

    print("GRAIN-CTRL smoke test")
    print(f"  train action mean: {a_train.mean().item():.3f}, uncertainty: {u.mean().item():.3f}")
    print(f"  eval action mean: {a_eval.mean().item():.3f}, stop ratio: {stop.float().mean().item():.2f}")
    assert a_train.shape == (batch,)
    print("PASS")


if __name__ == "__main__":
    main()
