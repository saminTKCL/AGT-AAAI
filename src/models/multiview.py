"""Multi-view aggregation: analytic explicit granularity + implicit similarity."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.aggregation import aggregate_fractional
from src.models.agt import AnalyticGranularityPolicy
from src.models.implicit import ImplicitAggregator, ImplicitGate


class MultiViewAggregator(nn.Module):
    """GRAIN-style multi-view: explicit (analytic k*) + implicit (kNN semantic).

    Args:
        align_scale: Graph-level kNN label alignment score in [0, 1].
            When < align_threshold, the implicit branch is suppressed.
            Computed once at construction time from training labels.
        align_threshold: Below this alignment, implicit contribution is
            linearly reduced to zero. Default 0.35 (calibrated on benchmark).
    """

    def __init__(
        self,
        feature_powers: list[torch.Tensor],
        x: torch.Tensor,
        agt_policy: AnalyticGranularityPolicy,
        stats_matrix: torch.Tensor,
        alpha_mix: float = 0.2,
        implicit_k: int = 10,
        learnable_gate: bool = True,
        align_scale: float = 1.0,
        align_threshold: float = 0.35,
    ):
        super().__init__()
        self.feature_powers = feature_powers
        self.x = x
        self.agt = agt_policy
        self.stats = stats_matrix
        self.alpha_mix = alpha_mix
        self.implicit = ImplicitAggregator(x, k=implicit_k)
        self.gate = ImplicitGate(stat_dim=stats_matrix.size(-1), global_gate=not learnable_gate)
        if agt_policy.calibration is not None:
            self.calibration = agt_policy.calibration

        # Adaptive suppression: scale implicit branch by how well features predict labels
        # align_scale in [0,1]; values below threshold linearly reduce implicit weight
        implicit_weight = max(0.0, (align_scale - align_threshold) / (1.0 - align_threshold + 1e-6))
        self.register_buffer("implicit_weight", torch.tensor(float(implicit_weight)))
        self.align_scale = align_scale
        self.align_threshold = align_threshold

    @property
    def granularity(self) -> torch.Tensor:
        return self.agt.granularity

    def explicit_features(self, node_indices: torch.Tensor) -> torch.Tensor:
        k = self.agt.select_action(node_indices)
        actions = k.unsqueeze(-1) if k.dim() == 1 else k
        return aggregate_fractional(actions, node_indices, self.feature_powers, self.alpha_mix)

    def forward(
        self,
        node_indices: torch.Tensor,
        explicit_only: bool = False,
    ) -> torch.Tensor:
        explicit = self.explicit_features(node_indices)
        if explicit_only or self.implicit_weight.item() == 0.0:
            return explicit
        implicit = self.implicit(self.x, node_indices)
        beta = self.gate(self.stats[node_indices]).unsqueeze(-1)
        return explicit + self.implicit_weight * beta * implicit

    def select_action(self, node_indices: torch.Tensor) -> torch.Tensor:
        """TD3-compatible: return analytic granularity for explicit branch."""
        return self.agt.select_action(node_indices)
