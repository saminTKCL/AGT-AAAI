"""Closed-form optimal granularity under degree-corrected contextual SBM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CSBMParams:
    """Parameters for the analytic granularity map k*(h, d, SNR)."""

    k_min: float = 0.2
    k_max: float = 5.0
    alpha_h: float = 1.5
    beta_d: float = 0.35
    gamma_snr: float = 0.25
    hetero_floor: float = 0.15


def closed_form_granularity(
    homophily: torch.Tensor,
    degree_log: torch.Tensor,
    snr: torch.Tensor,
    params: CSBMParams | None = None,
) -> torch.Tensor:
    """Map local statistics to continuous optimal hop depth k*.

    Under a degree-corrected contextual SBM, class-consistent signal
    decays with hop depth while cross-class noise accumulates. The
    resulting SNR(k) is unimodal; its maximizer satisfies:

        k* proportional to h_v^alpha * (1 + beta * log(1 + d_v)) * SNR_v^gamma

    Low homophily pushes k* toward k_min (high-pass / shallow aggregation).
    """
    p = params or CSBMParams()
    h = homophily.clamp(p.hetero_floor, 1.0)
    d_term = 1.0 + p.beta_d * degree_log
    snr_term = (snr / (snr.mean() + 1e-6)).clamp(0.1, 5.0) ** p.gamma_snr
    raw = (h ** p.alpha_h) * d_term * snr_term
    raw = raw / (raw.max() + 1e-6)
    k_star = p.k_min + (p.k_max - p.k_min) * raw
    return k_star.clamp(p.k_min, p.k_max)


class CalibrationHead(nn.Module):
    """Lightweight residual calibration delta(v) in [-1, 1]."""

    def __init__(self, in_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats).squeeze(-1)


class AnalyticGranularityPolicy:
    """Deterministic per-node granularity without RL."""

    def __init__(
        self,
        stats: dict[str, torch.Tensor],
        params: CSBMParams | None = None,
        calibration: CalibrationHead | None = None,
        calib_scale: float = 0.5,
        device: torch.device | None = None,
    ):
        self.stats = stats
        self.params = params or CSBMParams()
        self.calibration = calibration
        self.calib_scale = calib_scale
        self.device = device or stats["homophily"].device
        self._k_cache: torch.Tensor | None = None
        self._refresh()

    def _stat_matrix(self) -> torch.Tensor:
        return torch.stack(
            [
                self.stats["homophily"],
                self.stats["degree_log"] / (self.stats["degree_log"].max() + 1e-6),
                self.stats["snr"] / (self.stats["snr"].max() + 1e-6),
                self.stats["degree"] / (self.stats["degree"].max() + 1e-6),
            ],
            dim=-1,
        )

    def _refresh(self):
        k = closed_form_granularity(
            self.stats["homophily"],
            self.stats["degree_log"],
            self.stats["snr"],
            self.params,
        )
        if self.calibration is not None:
            delta = self.calibration(self._stat_matrix())
            k = k + self.calib_scale * delta
        self._k_cache = k.clamp(self.params.k_min, self.params.k_max)

    @property
    def granularity(self) -> torch.Tensor:
        return self._k_cache

    def select_action(self, node_indices: torch.Tensor | None = None) -> torch.Tensor:
        k = self.granularity
        if node_indices is not None:
            k = k[node_indices]
        return k

    def train_step(self):
        if self.calibration is not None:
            self._refresh()

    def eval_mode(self):
        if self.calibration is not None:
            self.calibration.eval()

    def train_mode(self):
        if self.calibration is not None:
            self.calibration.train()
