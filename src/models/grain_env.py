"""GRAIN-style MDP environment with multi-granularity aggregation."""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch_geometric.utils import degree

from src.models.aggregation import aggregate_fractional
from src.models.local_stats import compute_local_stats
from src.models.agt import AnalyticGranularityPolicy, CalibrationHead, CSBMParams


def _is_multiview_policy(policy) -> bool:
    from src.models.multiview import MultiViewAggregator
    return isinstance(policy, MultiViewAggregator)


class ClassifierHead(nn.Module):
    """Two-layer MLP classifier used inside GRAIN meta-training."""

    def __init__(self, in_dim: int, hidden: int, n_class: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_class)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(self.fc2(x), dim=-1)


def build_feature_powers(x: torch.Tensor, edge_index: torch.Tensor, max_hop: int = 7):
    """Precompute normalized adjacency powers for fractional-hop aggregation."""
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(
        edge_index, norm, (x.size(0), x.size(0)), device=x.device
    ).coalesce()
    powers = [x]
    for _ in range(max_hop):
        powers.append(torch.sparse.mm(adj, powers[-1]))
    return powers


class GrainEnvironment:
    """Unified environment for GRAIN-TD3 and AGT meta-training."""

    def __init__(
        self,
        data: Data,
        hidden: int = 64,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        max_action: float = 5.0,
        alpha: float = 0.2,
        device: torch.device | None = None,
        train_only: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.to(self.device)
        self.max_action = max_action
        self.alpha = alpha
        self.n_class = int(data.y.max().item()) + 1
        self.model = ClassifierHead(data.num_features, hidden, self.n_class).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.feature_powers = build_feature_powers(self.data.x, self.data.edge_index)
        self.train_idx = torch.where(data.train_mask)[0]
        self.val_idx = torch.where(data.val_mask)[0]
        self.test_idx = torch.where(data.test_mask)[0]
        self.policy = None
        self._local_stats = compute_local_stats(self.data, train_only=train_only)
        self._build_sparse_adj(max_hop=6)

    def _build_sparse_adj(self, max_hop: int):
        ei = self.data.edge_index.cpu().numpy()
        adj = csr_matrix(
            (np.ones(ei.shape[1]), (ei[0], ei[1])),
            shape=(self.data.num_nodes, self.data.num_nodes),
        )
        self.adjs = [adj.copy()]
        for _ in range(max_hop):
            self.adjs.append(self.adjs[-1] * adj)

    def stochastic_next_nodes(self, actions: np.ndarray, node_indices: torch.Tensor) -> torch.Tensor:
        """Sample next MDP state via k-hop transition (official GRAIN env)."""
        next_batch = []
        acts = actions.reshape(-1)
        for idx, act in zip(node_indices.tolist(), acts):
            hop = int(np.clip(act, 0, len(self.adjs) - 1))
            prob = self.adjs[hop].getrow(idx).toarray().flatten()
            s = prob.sum()
            if s <= 0:
                next_batch.append(idx)
                continue
            prob /= s
            next_batch.append(int(np.random.choice(len(prob), p=prob)))
        return torch.tensor(next_batch, device=self.device, dtype=torch.long)

    def make_agt_policy(self, use_calibration: bool = True, params: CSBMParams | None = None) -> AnalyticGranularityPolicy:
        calib = CalibrationHead() if use_calibration else None
        if calib is not None:
            calib = calib.to(self.device)
        p = params or getattr(self, "_csbm_params_override", None) or CSBMParams(k_max=self.max_action)
        return AnalyticGranularityPolicy(
            self._local_stats,
            params=p,
            calibration=calib,
            device=self.device,
        )

    def knn_label_alignment(self, k: int = 10, train_only: bool = True) -> float:
        """Graph-level kNN feature-label alignment computed with or without leakage."""
        from src.models.local_stats import knn_label_alignment, knn_label_alignment_train_only
        if train_only:
            return knn_label_alignment_train_only(self.data, k=k)
        return knn_label_alignment(self.data, k=k)

    def make_multiview_aggregator(
        self,
        use_calibration: bool = True,
        implicit_k: int = 10,
        learnable_gate: bool = True,
        align_threshold: float = 0.0,
        train_only: bool = True,
        params: CSBMParams | None = None,
    ) -> "MultiViewAggregator":
        from src.models.multiview import MultiViewAggregator
        from src.models.agt import CSBMParams
        agt = self.make_agt_policy(use_calibration=use_calibration, params=params)
        stats = agt._stat_matrix()
        align_scale = self.knn_label_alignment(train_only=train_only) if align_threshold > 0 else 1.0
        mv = MultiViewAggregator(
            self.feature_powers,
            self.data.x,
            agt,
            stats,
            alpha_mix=self.alpha,
            implicit_k=implicit_k,
            learnable_gate=learnable_gate,
            align_scale=align_scale,
            align_threshold=align_threshold,
        ).to(self.device)
        mv.implicit.build_knn()
        return mv

    def _aggregate_nodes(self, node_indices: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        if _is_multiview_policy(self.policy):
            return self.policy(node_indices)
        if actions is None:
            actions = self.actions_for_nodes(node_indices)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        return aggregate_fractional(actions, node_indices, self.feature_powers, self.alpha)

    def actions_for_nodes(self, node_indices: torch.Tensor) -> torch.Tensor:
        if self.policy is None:
            raise RuntimeError("Policy not set")
        if hasattr(self.policy, "granularity"):
            return self.policy.select_action(node_indices)
        if hasattr(self.policy, "select_action"):
            states = self.data.x[node_indices].cpu().numpy()
            acts = self.policy.select_action(states)
            return torch.from_numpy(np.asarray(acts, dtype=np.float32)).to(self.device)
        raise TypeError(type(self.policy))

    def aggregate_batch(self, node_indices: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._aggregate_nodes(node_indices, actions)

    def train_step(self, node_indices: torch.Tensor, actions: torch.Tensor | None = None) -> float:
        self.model.train()
        feats = self._aggregate_nodes(node_indices, actions)
        self.optimizer.zero_grad()
        loss = F.nll_loss(self.model(feats), self.data.y[node_indices])
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, split: str = "test") -> float:
        self.model.eval()
        idx = {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[split]
        feats = self._aggregate_nodes(idx)
        pred = self.model(feats).argmax(dim=-1)
        acc = (pred == self.data.y[idx]).float().mean().item()
        return acc

    def run_meta_training(
        self,
        policy,
        episodes: int = 50,
        batch_all_train: bool = True,
    ) -> dict:
        """Meta-train classifier with a fixed granularity policy."""
        self.policy = policy
        best_val, best_test = 0.0, 0.0
        history = []
        for ep in range(1, episodes + 1):
            if batch_all_train:
                idx = self.train_idx
            else:
                perm = torch.randperm(len(self.train_idx))
                idx = self.train_idx[perm[: max(32, len(self.train_idx) // 4)]]
            actions = self.actions_for_nodes(idx) if not _is_multiview_policy(self.policy) else None
            loss = self.train_step(idx, actions)
            val_acc = self.evaluate("val")
            test_acc = self.evaluate("test")
            best_val = max(best_val, val_acc)
            best_test = max(best_test, test_acc)
            history.append({"epoch": ep, "loss": loss, "val": val_acc, "test": test_acc})
        return {"best_val": best_val, "best_test": best_test, "history": history}

    def calibrate_agt(self, policy: AnalyticGranularityPolicy, steps: int = 100, lr: float = 1e-3):
        """End-to-end fine-tune calibration head on validation CE."""
        if policy.calibration is None:
            return
        policy.train_mode()
        opt = torch.optim.Adam(policy.calibration.parameters(), lr=lr)
        for _ in range(steps):
            policy.calibration.train()
            opt.zero_grad()
            policy._refresh()
            idx = self.train_idx
            actions = policy.select_action(idx)
            feats = self.aggregate_batch(idx, actions)
            loss = F.nll_loss(self.model(feats), self.data.y[idx])
            loss.backward()
            opt.step()
        policy.eval_mode()

    def reset_model(self):
        """Reinitialize classifier weights for a fresh run."""
        for layer in self.model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.model.parameters())
        if _is_multiview_policy(self.policy):
            total += sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        elif self.policy is not None and hasattr(self.policy, "calibration") and self.policy.calibration is not None:
            total += sum(p.numel() for p in self.policy.calibration.parameters())
        return total

    def multiview_trainable_params(self):
        """Parameters trained jointly in AGT-Implicit (classifier + gate + optional calib)."""
        params = list(self.model.parameters())
        if _is_multiview_policy(self.policy):
            params += list(self.policy.gate.parameters())
            if self.policy.agt.calibration is not None:
                params += list(self.policy.agt.calibration.parameters())
        return params
