"""GRAIN-CTRL: deterministic uncertainty-aware granularity controller."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    """Predict epistemic uncertainty from node state."""

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(state)).squeeze(-1)


class DeterministicGranularityController(nn.Module):
    """Replace TD3 stochastic policy with deterministic Gumbel-Softmax controller."""

    def __init__(self, state_dim: int, n_actions: int, max_action: float = 5.0, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.max_action = max_action
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.action_logits = nn.Linear(hidden, n_actions)
        self.uncertainty = UncertaintyHead(state_dim)

    def forward(self, state: torch.Tensor, temperature: float = 0.5, hard: bool = False):
        u = self.uncertainty(state)
        h = self.backbone(torch.cat([state, u.unsqueeze(-1)], dim=-1))
        logits = self.action_logits(h)
        if self.training:
            action_soft = F.gumbel_softmax(logits, tau=temperature, hard=hard)
            action = (action_soft * torch.arange(self.n_actions, device=state.device).float()).sum(-1)
        else:
            action = logits.argmax(dim=-1).float()
        # scale to GRAIN action range
        action = action / max(self.n_actions - 1, 1) * self.max_action
        stop = u < 0.15
        return action, u, stop

    def select_action(self, state: torch.Tensor) -> tuple[float, float, bool]:
        self.eval()
        with torch.no_grad():
            a, u, stop = self.forward(state, hard=True)
        return float(a.mean()), float(u.mean()), bool(stop.all())
