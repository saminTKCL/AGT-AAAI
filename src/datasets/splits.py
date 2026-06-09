"""Fallback geom-gcn style splits when precomputed masks are unavailable."""

from __future__ import annotations

import numpy as np
import torch


def geom_gcn_split(
    y: torch.Tensor,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class stratified split matching geom-gcn 60/20/20 protocol."""
    rng = np.random.RandomState(seed)
    n = y.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for c in range(int(y.max().item()) + 1):
        idx = torch.where(y == c)[0].numpy()
        rng.shuffle(idx)
        n_c = len(idx)
        n_train = max(1, int(n_c * train_ratio))
        n_val = max(1, int(n_c * val_ratio))
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train : n_train + n_val]] = True
        test_mask[idx[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask
