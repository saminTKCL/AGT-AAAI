"""TD3 distillation and training helpers for GRAIN-CTRL."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def collect_td3_trajectories(env, td3_agent, n_episodes: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (state, action) pairs from TD3 teacher."""
    states, actions = [], []
    for _ in range(n_episodes):
        s, done = env.reset2(), False
        while not done:
            a = np.asarray(td3_agent.select_action(s), dtype=np.float32).reshape(-1, env.action_num)
            states.append(s.copy())
            actions.append(a.copy())
            s, _, done, _ = env.step2(a.clip(0, 5))
    st = torch.from_numpy(np.concatenate(states, axis=0)).float()
    ac = torch.from_numpy(np.concatenate(actions, axis=0)).float()
    return st, ac


def distill_ctrl(ctrl, states: torch.Tensor, actions: torch.Tensor, device, steps: int = 100, lr: float = 1e-3):
    """Supervised imitation of TD3 actions with batched regression."""
    ctrl.train()
    opt = torch.optim.Adam(ctrl.parameters(), lr=lr)
    states = states.to(device)
    actions = actions.to(device)
    bs = min(512, states.size(0))
    for _ in range(steps):
        idx = torch.randperm(states.size(0), device=device)[:bs]
        st, ac = states[idx], actions[idx]
        opt.zero_grad()
        pred, _, _ = ctrl(st, hard=False)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if ac.dim() == 1:
            ac = ac.unsqueeze(1)
        loss = F.smooth_l1_loss(pred, ac)
        loss.backward()
        opt.step()
    ctrl.eval()
