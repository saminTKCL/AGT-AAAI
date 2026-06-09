"""TD3 agent for GRAIN baseline (from official GRAIN implementation)."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, net_width: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.Tanh(),
            nn.Linear(net_width, net_width),
            nn.Tanh(),
            nn.Linear(net_width, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(state)) * self.max_action


class Q_Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, net_width: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        return self.q1(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 5000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dead = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, dead):
        n = state.shape[0]
        end = min(self.ptr + n, self.max_size)
        sl = end - self.ptr
        self.state[self.ptr:end] = state[:sl]
        self.action[self.ptr:end] = action[:sl]
        self.reward[self.ptr] = reward
        self.next_state[self.ptr:end] = next_state[:sl]
        self.dead[self.ptr:end] = dead[:sl]
        self.ptr = (self.ptr + sl) % self.max_size
        self.size = min(self.size + sl, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[idx]),
            torch.FloatTensor(self.action[idx]),
            torch.FloatTensor(self.reward[idx]),
            torch.FloatTensor(self.next_state[idx]),
            torch.FloatTensor(self.dead[idx]),
        )


class TD3Policy:
    """TD3 meta-policy compatible with GrainEnvironment."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        max_action: float = 5.0,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, 128, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Q_Critic(state_dim, action_dim, 128).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.policy_noise = 0.3
        self.noise_clip = 0.5
        self.tau = 0.005
        self.delay_counter = -1
        self.delay_freq = 1

    def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
        with torch.no_grad():
            st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            a = self.actor(st)
            if noise:
                n = torch.randn_like(a) * self.policy_noise * 0.8
                a = (a + n).clamp(0, self.max_action)
        return a.cpu().numpy().reshape(-1)

    def train(self, replay: ReplayBuffer, batch_size: int = 128):
        self.delay_counter += 1
        s, a, r, s2, dw = replay.sample(batch_size)
        s, a, r, s2, dw = [t.to(self.device) for t in (s, a, r, s2, dw)]
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_target(s2) + noise).clamp(0, self.max_action)
            tq1, tq2 = self.critic_target(s2, na)
            target_q = r + (1 - dw) * self.gamma * torch.min(tq1, tq2)
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        if self.delay_counter % self.delay_freq == 0:
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            self.delay_counter = -1

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.actor.parameters()) + sum(
            p.numel() for p in self.critic.parameters()
        )


class FixedGranularityPolicy:
    """Ablation: constant hop for all nodes."""

    def __init__(self, k: float = 2.0):
        self.k = k

    def select_action(self, state: np.ndarray) -> np.ndarray:
        n = state.shape[0] if state.ndim > 1 else 1
        return np.full((n,), self.k, dtype=np.float32)


class RandomGranularityPolicy:
    """Ablation: uniform random hop in [0, max_action]."""

    def __init__(self, max_action: float = 5.0, seed: int = 0):
        self.max_action = max_action
        self.rng = np.random.RandomState(seed)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        n = state.shape[0] if state.ndim > 1 else 1
        return self.rng.uniform(0, self.max_action, size=(n,)).astype(np.float32)
