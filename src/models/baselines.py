"""Standard GNN baselines for heterophily benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, APPNP


@dataclass
class TrainResult:
    best_val: float
    test_acc: float
    train_time: float
    params: int


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_class, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )

    def forward(self, x, edge_index=None):
        return self.net(x)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, n_class, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, n_class)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, n_class, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin = nn.Linear(hidden, n_class)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class H2GCN(nn.Module):
    """Simplified H2GCN: concatenate 1-hop and 2-hop ego+neighbor features."""

    def __init__(self, in_dim, hidden, n_class, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(in_dim + hidden * 2, n_class)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h0 = x
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        out = torch.cat([h0, h1, h2], dim=-1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin(out)


class GPRGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_class, K=10, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, n_class)
        self.alpha = nn.Parameter(torch.ones(K + 1) / (K + 1))
        self.K = K
        self.dropout = dropout

    def forward(self, x, edge_index):
        from torch_geometric.nn import GCNConv

        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        conv = GCNConv(h.size(-1), h.size(-1), add_self_loops=False)
        # Manual normalized propagation
        from torch_geometric.utils import degree

        row, col = edge_index
        deg = degree(col, x.size(0)).clamp(min=1)
        norm = deg[row].pow(-0.5) * deg[col].pow(-0.5)
        adj = torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0)), device=x.device)
        outs = [self.lin2(h)]
        cur = h
        for _ in range(self.K):
            cur = torch.sparse.mm(adj, cur)
            outs.append(self.lin2(cur))
        w = F.softmax(self.alpha, dim=0)
        return sum(w[i] * outs[i] for i in range(len(outs)))


class ACMGNN(nn.Module):
    """Adaptive channel mixing (simplified ACM-GNN)."""

    def __init__(self, in_dim, hidden, n_class, dropout=0.5):
        super().__init__()
        self.low = GCNConv(in_dim, hidden)
        self.high = nn.Linear(in_dim, hidden)
        self.gate = nn.Linear(hidden * 2, 1)
        self.lin = nn.Linear(hidden, n_class)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h_low = F.relu(self.low(x, edge_index))
        h_high = F.relu(self.high(x))
        gate = torch.sigmoid(self.gate(torch.cat([h_low, h_high], dim=-1)))
        h = gate * h_low + (1 - gate) * h_high
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(h)


class GloGNN(nn.Module):
    """Global-local operator approximation via high/low pass."""

    def __init__(self, in_dim, hidden, n_class, dropout=0.5):
        super().__init__()
        self.conv = GCNConv(in_dim, hidden)
        self.global_lin = nn.Linear(in_dim, hidden)
        self.lin = nn.Linear(hidden * 2, n_class)
        self.dropout = dropout

    def forward(self, x, edge_index):
        local = F.relu(self.conv(x, edge_index))
        global_f = F.relu(self.global_lin(x.mean(dim=0, keepdim=True).expand_as(x)))
        h = torch.cat([local, global_f], dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(h)


class FAGCN(nn.Module):
    """Frequency adaptive GCN (simplified)."""

    def __init__(self, in_dim, hidden, n_class, eps=0.2, dropout=0.5):
        super().__init__()
        self.eps = eps
        self.conv = GCNConv(in_dim, hidden)
        self.skip = nn.Linear(in_dim, hidden)
        self.lin = nn.Linear(hidden, n_class)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv(x, edge_index))
        h = (1 - self.eps) * h + self.eps * F.relu(self.skip(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(h)


from src.models.ordered_gnn import OrderedGNN


BASELINE_REGISTRY = {
    "MLP": MLP,
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "H2GCN": H2GCN,
    "GPR-GNN": GPRGNN,
    "ACM-GNN": ACMGNN,
    "GloGNN": GloGNN,
    "FAGCN": FAGCN,
    "Ordered-GNN": OrderedGNN,
}


def train_baseline(
    data: Data,
    model_name: str,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    device: torch.device | None = None,
) -> TrainResult:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    n_class = int(data.y.max().item()) + 1
    Model = BASELINE_REGISTRY[model_name]
    model = Model(data.num_features, 64, n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_test, best_state = 0.0, 0.0, None
    wait = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=-1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    return TrainResult(best_val, best_test, train_time, params)
