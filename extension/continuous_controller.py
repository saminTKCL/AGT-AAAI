"""Continuous action controller for GRAIN when action_dim=1 (matches TD3 regression)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousGranularityController(nn.Module):
    """Direct regression of hop granularity in [0, max_action]."""

    def __init__(self, state_dim: int, max_action: float = 5.0, hidden: int = 128):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, temperature: float = 0.5, hard: bool = False):
        raw = self.net(state).squeeze(-1)
        action = torch.sigmoid(raw) * self.max_action
        u = torch.sigmoid(-raw.abs())
        stop = u < 0.15
        return action, u, stop
