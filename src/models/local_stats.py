"""Local graph statistics for AGT computed in a leakage-free manner."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree


class PseudoClassifier(nn.Module):
    """Simple MLP to compute leakage-free pseudo-labels on training data."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.enable_grad()
def estimate_pseudo_homophily(data: Data, lr: float = 0.01, epochs: int = 150) -> torch.Tensor:
    """Compute per-node pseudo-homophily using a classifier trained only on train_mask."""
    x = data.x
    y = data.y
    train_mask = getattr(data, "train_mask", None)
    device = x.device

    # Fallback if no train mask or label information
    if train_mask is None or train_mask.sum() == 0 or y is None:
        return local_homophily(data)

    num_classes = int(y.max().item()) + 1
    model = PseudoClassifier(x.size(-1), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_idx = torch.where(train_mask)[0]

    # Simple training loop for pseudo-labels
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x[train_idx])
        loss = F.cross_entropy(out, y[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        pseudo_labels = logits.argmax(dim=-1)

    # Compute local homophily using the pseudo-labels
    row, col = data.edge_index
    same = (pseudo_labels[row] == pseudo_labels[col]).float()
    n = data.num_nodes
    deg = degree(col, n).clamp(min=1)
    hom = torch.zeros(n, device=device)
    hom.scatter_add_(0, col, same)
    hom = hom / deg
    return hom


@torch.no_grad()
def local_homophily(data: Data) -> torch.Tensor:
    """Per-node local homophily: fraction of same-label neighbors (oracle/leaky)."""
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
    """Graph-level kNN feature-label alignment score (oracle/leaky)."""
    x_norm = F.normalize(data.x.detach(), p=2, dim=-1)
    n = x_norm.size(0)
    k_eff = min(k, n - 1)
    chunk = 512
    align_sum, count = 0.0, 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim = x_norm[start:end] @ x_norm.t()
        for i in range(end - start):
            sim[i, start + i] = -1.0
        topk_idx = sim.topk(k_eff, dim=-1).indices
        same = (data.y[topk_idx] == data.y[start:end].unsqueeze(-1)).float()
        align_sum += same.mean(dim=-1).sum().item()
        count += end - start
    return align_sum / max(count, 1)


@torch.no_grad()
def knn_label_alignment_train_only(data: Data, k: int = 10, pseudo_labels: torch.Tensor | None = None) -> float:
    """Graph-level kNN feature-label alignment score computed without leakage.

    Uses ground-truth labels for training nodes and pseudo-labels for non-training nodes.
    Only evaluates alignment on training set nodes to avoid validation/test set leakage.
    """
    device = data.x.device
    train_mask = getattr(data, "train_mask", None)

    if train_mask is None or train_mask.sum() == 0 or data.y is None:
        return knn_label_alignment(data, k=k)

    if pseudo_labels is None:
        x = data.x
        y = data.y
        num_classes = int(y.max().item()) + 1
        # Quick training to get pseudo-labels
        # Wrap in enabled grad because training is done here
        with torch.enable_grad():
            model = PseudoClassifier(x.size(-1), num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            train_idx = torch.where(train_mask)[0]
            model.train()
            for _ in range(100):
                optimizer.zero_grad()
                out = model(x[train_idx])
                loss = F.cross_entropy(out, y[train_idx])
                loss.backward()
                optimizer.step()
            model.eval()
            logits = model(x)
            pseudo_labels = logits.argmax(dim=-1)

    labels_est = pseudo_labels.clone()
    labels_est[train_mask] = data.y[train_mask]

    train_idx = torch.where(train_mask)[0]
    n_train = len(train_idx)
    if n_train == 0:
        return 0.5

    x_norm = F.normalize(data.x.detach(), p=2, dim=-1)
    n = x_norm.size(0)
    k_eff = min(k, n - 1)

    chunk = 512
    align_sum, count = 0.0, 0
    for start in range(0, n_train, chunk):
        end = min(start + chunk, n_train)
        batch_idx = train_idx[start:end]
        sim = x_norm[batch_idx] @ x_norm.t()
        for i, idx_val in enumerate(batch_idx.tolist()):
            sim[i, idx_val] = -1.0
        topk_idx = sim.topk(k_eff, dim=-1).indices
        same = (labels_est[topk_idx] == labels_est[batch_idx].unsqueeze(-1)).float()
        align_sum += same.mean(dim=-1).sum().item()
        count += len(batch_idx)

    return align_sum / max(count, 1)


@torch.no_grad()
def compute_local_stats(data: Data, snr_k: int = 5, train_only: bool = True) -> dict[str, torch.Tensor]:
    """Compute per-node statistics used by AGT in a leakage-free or leaky manner."""
    n = data.num_nodes
    deg = degree(data.edge_index[1], n).float()
    deg_log = torch.log1p(deg)

    if train_only:
        hom = estimate_pseudo_homophily(data)
    else:
        hom = local_homophily(data)

    snr = feature_snr(data.x, k=snr_k)
    return {
        "homophily": hom,
        "degree": deg,
        "degree_log": deg_log,
        "snr": snr,
    }

