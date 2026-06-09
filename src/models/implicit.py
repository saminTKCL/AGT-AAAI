"""Implicit (non-neighbor) feature aggregation via semantic similarity."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitAggregator(nn.Module):
    """kNN feature aggregation capturing distant semantic similarity (GRAIN implicit branch)."""

    def __init__(self, x: torch.Tensor, k: int = 10, chunk: int = 512):
        super().__init__()
        self.k = k
        self.chunk = chunk
        self.register_buffer("x_norm", F.normalize(x.detach(), p=2, dim=-1))
        self._knn_idx: torch.Tensor | None = None
        self._knn_w: torch.Tensor | None = None

    @torch.no_grad()
    def build_knn(self):
        """Precompute top-k similar nodes per node (excluding self)."""
        n = self.x_norm.size(0)
        device = self.x_norm.device
        idx_list, w_list = [], []
        for start in range(0, n, self.chunk):
            end = min(start + self.chunk, n)
            sim = self.x_norm[start:end] @ self.x_norm.t()
            sim.fill_diagonal_(0.0)
            for i in range(end - start):
                row = sim[i]
                topv, topi = row.topk(min(self.k, n - 1))
                mask = topv > 0
                if mask.sum() == 0:
                    topi = torch.tensor([start + i], device=device)
                    topv = torch.tensor([1.0], device=device)
                else:
                    topi = topi[mask]
                    topv = topv[mask]
                w = topv / topv.sum()
                idx_list.append(topi)
                w_list.append(w)
        max_k = max(len(t) for t in idx_list)
        knn_idx = torch.zeros(n, max_k, dtype=torch.long, device=device)
        knn_w = torch.zeros(n, max_k, device=device)
        for i, (ind, w) in enumerate(zip(idx_list, w_list)):
            knn_idx[i, : len(ind)] = ind
            knn_w[i, : len(w)] = w
        self._knn_idx = knn_idx
        self._knn_w = knn_w

    def forward(self, x: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
        if self._knn_idx is None:
            self.build_knn()
        idx = self._knn_idx[node_indices]
        w = self._knn_w[node_indices]
        feats = x[idx]  # (B, K, F)
        return (feats * w.unsqueeze(-1)).sum(dim=1)


class ImplicitGate(nn.Module):
    """Learnable mixing weight for implicit branch (per-node or global)."""

    def __init__(self, stat_dim: int = 4, hidden: int = 32, global_gate: bool = False):
        super().__init__()
        self.global_gate = global_gate
        if global_gate:
            self.beta = nn.Parameter(torch.tensor(0.2))
        else:
            self.net = nn.Sequential(
                nn.Linear(stat_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if self.global_gate:
            return torch.sigmoid(self.beta).expand(stats.size(0))
        return self.net(stats).squeeze(-1)
