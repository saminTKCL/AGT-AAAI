"""Ordered GNN baseline (simplified ordered aggregation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class OrderedGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_class, order: int = 3, dropout=0.5):
        super().__init__()
        self.order = order
        self.convs = nn.ModuleList([GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(order)])
        self.lin = nn.Linear(hidden * order, n_class)
        self.dropout = dropout

    def forward(self, x, edge_index):
        hs = []
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            hs.append(h)
        out = torch.cat(hs, dim=-1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin(out)
