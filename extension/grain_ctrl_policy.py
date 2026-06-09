"""TD3-compatible policy wrapper for GRAIN-CTRL."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_EXT = Path(__file__).resolve().parent
if str(_EXT) not in sys.path:
    sys.path.insert(0, str(_EXT))

from continuous_controller import ContinuousGranularityController
from deterministic_controller import DeterministicGranularityController


class GRAINCTRLPolicy:
    """Drop-in replacement for TD3_Agent.select_action without exploration noise."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float = 5.0):
        self.action_dim = action_dim
        self.max_action = max_action
        if action_dim == 1:
            self.ctrl = ContinuousGranularityController(state_dim, max_action)
        else:
            self.ctrl = DeterministicGranularityController(state_dim, action_dim, max_action)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctrl.to(self.device)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        st = torch.from_numpy(state).float().to(self.device)
        if st.dim() == 1:
            st = st.unsqueeze(0)
        with torch.no_grad():
            a, u, stop = self.ctrl(st, hard=True)
            if stop.any():
                a = torch.where(stop, a * 0.5, a)
        out = a.cpu().numpy().astype(np.float32)
        if state.ndim == 1:
            return out.squeeze() if out.size == 1 else out
        return out

    def eval_step(self, state: np.ndarray) -> np.ndarray:
        return self.select_action(state)

    def train(self, replay_buffer):
        pass
