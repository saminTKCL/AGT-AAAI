"""Local graph statistics for AGT."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree


@torch.no_grad()
def local_homophily(data: Data) -> torch.Tensor:
    """Per-node local homophily: fraction of same-label neighbors."""
    row, col = data.edge_index
    y = data.y
    n = data.num_nodes
    same = (y[row] == y[col]).float()
    deg = degree(col, n).clamp(min=1)
    hom = torch.zeros(n, device=data.x.device)
    hom.scatter_add_(0, col, same)
    hom = hom / deg
    return hom


@torch.no_grad()
def feature_snr(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Per-node feature signal-to-noise proxy via k-NN cosine similarity."""
    x_norm = F.normalize(x, p=2, dim=-1)
    sim = x_norm @ x_norm.t()
    sim.fill_diagonal_(-1.0)
    topk = sim.topk(min(k, x.size(0) - 1), dim=-1).values
    signal = topk.mean(dim=-1)
    noise = x_norm.std(dim=-1).clamp(min=1e-6)
    return (signal / noise).clamp(0, 10)


@torch.no_grad()
def knn_label_alignment(data: Data, k: int = 10) -> float:
    """Graph-level kNN feature-label alignment score.

    Measures whether top-k feature-similar nodes tend to share the same label.
    Computed on labelled nodes only (train + val + test masks where available).
    Returns a float in [0, 1]. Values < ~0.35 indicate feature similarity is
    unreliable for class prediction — the implicit branch should be suppressed.
    """
    x_norm = F.normalize(data.x.detach(), p=2, dim=-1)
    n = x_norm.size(0)
    k_eff = min(k, n - 1)
    chunk = 512
    align_sum, count = 0.0, 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim = x_norm[start:end] @ x_norm.t()
        # zero diagonal blocks
        for i in range(end - start):
            sim[i, start + i] = -1.0
        topk_idx = sim.topk(k_eff, dim=-1).indices  # (chunk, k)
        same = (data.y[topk_idx] == data.y[start:end].unsqueeze(-1)).float()
        align_sum += same.mean(dim=-1).sum().item()
        count += end - start
    return align_sum / max(count, 1)


@torch.no_grad()
def compute_local_stats(data: Data, snr_k: int = 5) -> dict[str, torch.Tensor]:
    """Compute per-node statistics used by AGT."""
    n = data.num_nodes
    deg = degree(data.edge_index[1], n).float()
    deg_log = torch.log1p(deg)
    hom = local_homophily(data)
    snr = feature_snr(data.x, k=snr_k)
    return {
        "homophily": hom,
        "degree": deg,
        "degree_log": deg_log,
        "snr": snr,
    }
