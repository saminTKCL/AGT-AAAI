"""Fractional-hop explicit aggregation (GRAIN paper Eq. 3)."""

from __future__ import annotations

import numpy as np
import torch


def aggregate_fractional(
    actions: torch.Tensor,
    node_indices: torch.Tensor,
    feature_powers: list[torch.Tensor],
    alpha: float = 0.2,
) -> torch.Tensor:
    """GRAIN multi-granularity aggregation with fractional hop."""
    feats = []
    for k, idx in enumerate(node_indices.tolist()):
        a = float(actions[k].item() if actions[k].dim() == 0 else actions[k, 0].item())
        a = float(np.clip(a, 0.0, 5.0))
        hop = int(round(a))
        if hop == 0:
            fea = (a - hop) * feature_powers[hop][idx] + (hop + 1 - a) * feature_powers[hop + 1][idx]
        else:
            fea = torch.zeros_like(feature_powers[0][idx])
            for j in range(hop):
                fea = fea + (1 - alpha) * feature_powers[j][idx] + alpha * feature_powers[0][idx]
            fea = fea / hop
            fea = fea + (a - hop) * feature_powers[hop][idx] + (hop + 1 - a) * feature_powers[hop + 1][idx]
        feats.append(fea.unsqueeze(0))
    return torch.cat(feats, dim=0)
