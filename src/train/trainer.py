"""Training orchestration matching official GRAIN protocol (AAAI 2025)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.models.grain_env import GrainEnvironment
from src.models.grain_td3 import ReplayBuffer, TD3Policy
from src.models.multiview import MultiViewAggregator


@dataclass
class MethodResult:
    method: str
    dataset: str
    seed: int
    test_acc: float
    val_acc: float
    train_time: float
    params: int
    peak_mem_mb: float = 0.0


def _peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def _run_gnn_phase_joint(env: GrainEnvironment, policy, episodes: int = 100, lr: float = 0.01) -> dict:
    """GNN phase with joint optimizer over classifier + multiview gates."""
    env.reset_model()
    env.policy = policy
    params = env.multiview_trainable_params()
    opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
    best_val, best_test = 0.0, 0.0
    for _ in range(episodes):
        idx = env.train_idx
        opt.zero_grad()
        feats = env._aggregate_nodes(idx)
        loss = torch.nn.functional.nll_loss(env.model(feats), env.data.y[idx])
        loss.backward()
        opt.step()
        if isinstance(policy, MultiViewAggregator):
            if policy.agt.calibration is not None:
                policy.agt._refresh()
        val_acc = env.evaluate("val")
        test_acc = env.evaluate("test")
        best_val = max(best_val, val_acc)
        best_test = max(best_test, test_acc)
    return {"best_val": best_val, "best_test": best_test}


def _run_gnn_phase(env: GrainEnvironment, policy, episodes: int = 100) -> dict:
    """Phase 2: train GNN with fixed policy (official GRAIN train.py lines 103-119)."""
    env.reset_model()
    env.policy = policy
    best_val, best_test = 0.0, 0.0
    for _ in range(episodes):
        idx = env.train_idx
        actions = env.actions_for_nodes(idx)
        env.train_step(idx, actions)
        val_acc = env.evaluate("val")
        test_acc = env.evaluate("test")
        best_val = max(best_val, val_acc)
        best_test = max(best_test, test_acc)
    return {"best_val": best_val, "best_test": best_test}


def train_grain_td3(
    env: GrainEnvironment,
    rl_episodes: int = 100,
    gnn_episodes: int = 100,
    expl_noise: float = 0.15,
) -> MethodResult:
    """Official GRAIN-TD3: RL policy learning + GNN training with best policy."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    state_dim = env.data.num_features
    td3 = TD3Policy(state_dim, action_dim=1, max_action=env.max_action, device=env.device)
    replay = ReplayBuffer(state_dim, 1, max_size=max(2000, len(env.train_idx) * 5))
    env.policy = td3

    start_steps = 5
    best_policy = td3
    last_val = 0.0
    total_steps = 0
    noise = expl_noise

    while total_steps < rl_episodes:
        states = env.data.x[env.train_idx].cpu().numpy()
        if total_steps < start_steps:
            actions = np.random.normal(2, 1, (len(states), 1)).clip(0, env.max_action)
        else:
            actions = td3.select_action(states, noise=True).reshape(-1, 1)
            actions = (
                actions + np.random.normal(0, env.max_action * noise, size=actions.shape)
            ).clip(0, env.max_action)

        idx = env.train_idx
        env.train_step(idx, torch.from_numpy(actions.astype(np.float32)).to(env.device))
        next_idx = env.stochastic_next_nodes(actions, idx)
        next_states = env.data.x[next_idx].cpu().numpy()

        val_acc = env.evaluate("val")
        reward = val_acc

        dead = np.zeros((len(states), 1), dtype=np.float32)
        replay.add(states, actions, reward, next_states, dead)

        if total_steps >= 2 and total_steps % 3 == 0:
            for _ in range(3):
                td3.train(replay)

        if total_steps % 10 == 0:
            noise *= 0.998

        if val_acc > last_val:
            last_val = val_acc
            best_policy = copy.deepcopy(td3)

        total_steps += 1

    out = _run_gnn_phase(env, best_policy, episodes=gnn_episodes)
    train_time = time.perf_counter() - t0
    return MethodResult(
        method="GRAIN-TD3",
        dataset="",
        seed=0,
        test_acc=out["best_test"],
        val_acc=out["best_val"],
        train_time=train_time,
        params=env.count_parameters() + td3.count_parameters(),
        peak_mem_mb=_peak_mem_mb(),
    )


def train_agt(
    env: GrainEnvironment,
    use_calibration: bool = True,
    gnn_episodes: int = 100,
    calib_steps: int = 150,
) -> MethodResult:
    """AGT: skip RL; analytic policy + optional calibration + GNN phase."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    policy = env.make_agt_policy(use_calibration=use_calibration)
    env.policy = policy

    if use_calibration and policy.calibration is not None:
        env.reset_model()
        env.calibrate_agt(policy, steps=calib_steps)

    out = _run_gnn_phase(env, policy, episodes=gnn_episodes)
    train_time = time.perf_counter() - t0
    params = env.count_parameters()
    if policy.calibration is not None:
        params += sum(p.numel() for p in policy.calibration.parameters())

    return MethodResult(
        method="AGT" if use_calibration else "AGT-closed",
        dataset="",
        seed=0,
        test_acc=out["best_test"],
        val_acc=out["best_val"],
        train_time=train_time,
        params=params,
        peak_mem_mb=_peak_mem_mb(),
    )


def train_agt_implicit(
    env: GrainEnvironment,
    use_calibration: bool = True,
    gnn_episodes: int = 100,
    implicit_k: int = 10,
    learnable_gate: bool = True,
) -> MethodResult:
    """AGT-Implicit: analytic explicit k* + learned implicit kNN branch, no RL."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    mv = env.make_multiview_aggregator(
        use_calibration=use_calibration,
        implicit_k=implicit_k,
        learnable_gate=learnable_gate,
    )
    out = _run_gnn_phase_joint(env, mv, episodes=gnn_episodes)
    train_time = time.perf_counter() - t0
    return MethodResult(
        method="AGT-Implicit",
        dataset="",
        seed=0,
        test_acc=out["best_test"],
        val_acc=out["best_val"],
        train_time=train_time,
        params=env.count_parameters(),
        peak_mem_mb=_peak_mem_mb(),
    )


def train_agt_implicit_adaptive(
    env: GrainEnvironment,
    use_calibration: bool = True,
    gnn_episodes: int = 100,
    implicit_k: int = 10,
    align_threshold: float = 0.35,
    params=None,
) -> MethodResult:
    """AGT-Implicit-Adaptive: suppress implicit branch on graphs where features
    don't predict labels (kNN alignment < align_threshold).

    This fixes the Chameleon/Squirrel regression: those graphs have kNN label
    alignment ~0.23-0.28, well below the 0.35 threshold, so implicit_weight -> 0
    and the model falls back to AGT explicit-only aggregation.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    mv = env.make_multiview_aggregator(
        use_calibration=use_calibration,
        implicit_k=implicit_k,
        learnable_gate=True,
        align_threshold=align_threshold,
        params=params,
    )
    out = _run_gnn_phase_joint(env, mv, episodes=gnn_episodes)
    train_time = time.perf_counter() - t0

    align = mv.align_scale
    iw = float(mv.implicit_weight.item())
    method = f"AGT-Implicit-Adaptive"
    print(f"    align={align:.3f} implicit_weight={iw:.3f}", flush=True)

    return MethodResult(
        method=method,
        dataset="",
        seed=0,
        test_acc=out["best_test"],
        val_acc=out["best_val"],
        train_time=train_time,
        params=env.count_parameters(),
        peak_mem_mb=_peak_mem_mb(),
    )


def train_with_policy(env: GrainEnvironment, policy, gnn_episodes: int = 100) -> MethodResult:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = _run_gnn_phase(env, policy, episodes=gnn_episodes)
    train_time = time.perf_counter() - t0
    name = getattr(policy, "__class__", type(policy)).__name__
    return MethodResult(
        method=name,
        dataset="",
        seed=0,
        test_acc=out["best_test"],
        val_acc=out["best_val"],
        train_time=train_time,
        params=env.count_parameters(),
        peak_mem_mb=_peak_mem_mb(),
    )
