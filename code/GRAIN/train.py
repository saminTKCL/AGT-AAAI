import torch
from TD3_agent import TD3_Agent, ReplayBuffer
from utils import evaluate_policy
from env.gcn import gcn_env
import numpy as np
from copy import deepcopy
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['td3', 'ctrl'], default='td3')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--max_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic=True

    start_steps = 5
    dataset = args.dataset
    max_episodes = args.max_episodes
    max_action = 5

    expl_noise = 0.15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gcn_env(dataset=dataset, max_layer=1)
    env.seed(args.seed)

    if args.policy == 'ctrl':
        ext = Path(__file__).resolve().parents[2] / 'extension'
        sys.path.insert(0, str(ext))
        from grain_ctrl_policy import GRAINCTRLPolicy
        model = GRAINCTRLPolicy(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_num,
            max_action=max_action,
        )
        start_steps = 0
    else:
        model = TD3_Agent(
            env_with_dw=True,
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_num,
            max_action=5,
        )
    env.policy = model
    replay_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_num, 3 * (len(env.train_indexes)-1))
    render = False
    last_val = 0.0
    # env.train_indexes
    if render:
        score = evaluate_policy(env, model, render, turns=10)
        print('EnvName:', dataset, 'score:', score)
    else:
        total_steps = 0
        while total_steps < max_episodes:
            s, done, steps, ep_r = env.reset2(), False, 0, 0

            '''Interact & trian'''
            while not done:
                steps += 1

                if total_steps < start_steps:
                    a = env.action_space(s.shape[0],1)
                elif args.policy == 'ctrl':
                    a = model.select_action(s).clip(0, max_action)
                    a = a.reshape(-1, 1)
                else:
                    a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=env.action_num)
                         ).clip(0, max_action)
                    a = a.reshape(-1, 1)
                s_prime, r, done, val = env.step2(a)
                '''Avoid impacts caused by reaching max episode steps'''
                if (done and steps != 3):
                    dw = False
                else:
                    dw = True

                replay_buffer.add(s, a, r, s_prime, dw)
                s = s_prime
                ep_r += r

                '''train if its time'''
                if total_steps >= 2 and total_steps % 3 == 0:
                    for j in range(3):
                        model.train(replay_buffer)

                '''record & log'''
                if total_steps % 10 == 0:
                    expl_noise *= 0.998
                    score = evaluate_policy(env, model, False)
                    print('steps: {}k'.format(int(total_steps/10)), 'score:', score)
                total_steps += 1

                '''save model'''
                if val > last_val:
                    last_val = val
                    best_policy = deepcopy(model)

    # Testing: Apply meta-policy to train a new GNN
    test_acc = 0.0
    print("Training GNNs with learned meta-policy")
    new_env = gcn_env(dataset=dataset, max_layer=1)
    new_env.seed(0)
    new_env.policy = best_policy
    state = new_env.reset2()
    ACC = 0
    for i_episode in range(1, 100):
        action = best_policy.select_action(state)
        action = action.reshape(-1, 1)
        state, reward, done, val_acc = new_env.step2(action)
        test_acc = new_env.test_batch()
        print("Training GNN", i_episode, "; Val_Acc:", val_acc, "; Test_Acc:", "{:.6f}".format(test_acc))
        if ACC < test_acc:
            ACC = test_acc
            # new_env.vs()

    filename = dataset + '.txt'

    with open(filename, "w") as file:
        file.write(f'Best Test Accuracy {"{:.6f}".format(ACC)}')

    print(f'Best Test Accuracy {"{:.6f}".format(ACC)}')

if __name__ == "__main__":
    main()
