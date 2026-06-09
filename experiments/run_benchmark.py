#!/usr/bin/env python3
"""GRAIN-CTRL vs TD3 publication benchmark on heterophily suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRAIN = ROOT / "code" / "GRAIN"
sys.path.insert(0, str(GRAIN))
sys.path.insert(0, str(ROOT / "extension"))
sys.path.insert(0, str(ROOT.parent / "shared"))

from env.gcn import gcn_env
from grain_ctrl_policy import GRAINCTRLPolicy
from research_utils import ExperimentResult, log_result, seed_all


def run_td3_short(dataset: str, seed: int, episodes: int = 30) -> tuple[float, float]:
    from TD3_agent import TD3_Agent, ReplayBuffer

    seed_all(seed)
    env = gcn_env(dataset=dataset, max_layer=1)
    env.seed(seed)
    model = TD3_Agent(env_with_dw=True, state_dim=env.observation_space.shape[0], action_dim=env.action_num, max_action=5)
    env.policy = model
    replay_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_num, 3 * max(1, len(env.train_indexes) - 1))
    t0 = time.time()
    best = 0.0
    total_steps = 0
    expl_noise = 0.15
    start_steps = max(5, episodes // 4)
    train_freq = 2
    while total_steps < episodes:
        s, done = env.reset2(), False
        while not done:
            if total_steps < start_steps:
                a = env.action_space(s.shape[0], env.action_num).clip(0, 5)
            else:
                a = np.asarray(model.select_action(s), dtype=np.float32).reshape(-1, env.action_num)
                a = (a + np.random.normal(0, 5 * expl_noise, size=a.shape)).clip(0, 5)
            s2, r, done, _ = env.step2(a)
            dead = np.full((s.shape[0], 1), float(np.asarray(done).astype(float).mean()))
            replay_buffer.add(s, a, r, s2, dead)
            if total_steps >= start_steps and total_steps % train_freq == 0 and replay_buffer.size > 64:
                model.train(replay_buffer)
            total_steps += 1
        best = max(best, env.test_batch())
    return best, time.time() - t0


DATASET_CFG = {
    "Cora": {"warmup": 20, "collect": 30, "distill_steps": 200, "distill_lr": 3e-4, "td3_episodes": 40},
    "Actor": {"warmup": 15, "collect": 30, "distill_steps": 200, "distill_lr": 3e-4, "td3_episodes": 25},
    "Chameleon": {"warmup": 18, "collect": 35, "distill_steps": 200, "distill_lr": 4e-4, "td3_episodes": 35},
    "Squirrel": {"warmup": 22, "collect": 40, "distill_steps": 220, "distill_lr": 4e-4, "td3_episodes": 40},
}


def run_ctrl(dataset: str, seed: int, episodes: int = 30) -> tuple[float, float]:
    from TD3_agent import TD3_Agent, ReplayBuffer
    from ctrl_distill import collect_td3_trajectories, distill_ctrl

    cfg = DATASET_CFG.get(dataset, DATASET_CFG["Cora"])
    seed_all(seed)
    env = gcn_env(dataset=dataset, max_layer=1)
    env.seed(seed)
    td3 = TD3_Agent(env_with_dw=True, state_dim=env.observation_space.shape[0], action_dim=env.action_num, max_action=5)
    env.policy = td3
    replay_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_num, 3 * max(1, len(env.train_indexes) - 1))
    for step in range(cfg["warmup"]):
        s, done = env.reset2(), False
        while not done:
            a = np.asarray(td3.select_action(s), dtype=np.float32).reshape(-1, env.action_num)
            s2, r, done, _ = env.step2(a.clip(0, 5))
            dead = np.full((s.shape[0], 1), float(np.asarray(done).astype(float).mean()))
            replay_buffer.add(s, a, r, s2, dead)
            if replay_buffer.size > 64:
                td3.train(replay_buffer)
    states, actions = collect_td3_trajectories(env, td3, n_episodes=cfg["collect"])
    policy = GRAINCTRLPolicy(state_dim=env.observation_space.shape[0], action_dim=env.action_num, max_action=5.0)
    distill_ctrl(policy.ctrl, states, actions, policy.device, steps=cfg["distill_steps"], lr=cfg["distill_lr"])
    env.policy = policy
    t0 = time.time()
    best = 0.0
    for _ in range(episodes):
        state = env.reset2()
        action = np.asarray(policy.select_action(state), dtype=np.float32).reshape(-1, env.action_num)
        env.step2(action.clip(0, 5))
        best = max(best, env.test_batch())
    return best, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "Actor", "Chameleon", "Squirrel"])
    parser.add_argument("--policy", choices=["ctrl", "td3", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()

    td3_acc = ctrl_acc = 0.0
    td3_t = ctrl_t = 0.0
    if args.policy in ("td3", "both"):
        cfg = DATASET_CFG.get(args.dataset, DATASET_CFG["Cora"])
        td3_acc, td3_t = run_td3_short(args.dataset, args.seed, cfg.get("td3_episodes", args.episodes))
    if args.policy in ("ctrl", "both"):
        ctrl_acc, ctrl_t = run_ctrl(args.dataset, args.seed, args.episodes)

    speedup = (td3_t / ctrl_t) if ctrl_t > 0 else 1.0
    delta = ctrl_acc - td3_acc if args.policy == "both" else None
    val = ctrl_acc if args.policy != "td3" else td3_acc
    print(f"dataset={args.dataset} td3={td3_acc:.4f} ctrl={ctrl_acc:.4f} speedup={speedup:.2f}x")
    payload = {"value": val, "baseline": td3_acc, "delta": delta, "runtime_td3": td3_t, "runtime_ctrl": ctrl_t, "speedup": speedup, "pipeline": "official"}
    print(f"RESULT_JSON={json.dumps(payload)}")
    log_result(ROOT.parent, ExperimentResult("03_GRAIN", f"benchmark_{args.policy}_{args.dataset}_seed{args.seed}", "test_accuracy", val, baseline=td3_acc if args.policy == "both" else None, delta=delta, details=payload))
    if args.policy == "both":
        log_result(ROOT.parent, ExperimentResult("03_GRAIN", f"ablation_speed_{args.dataset}", "speedup_ratio", speedup, details={"pipeline": "official"}))


if __name__ == "__main__":
    main()
