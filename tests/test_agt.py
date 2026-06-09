#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.agt import CSBMParams, closed_form_granularity
from src.models.local_stats import compute_local_stats
from src.datasets.load import load_dataset
from src.models.grain_env import GrainEnvironment


class TestAGT(unittest.TestCase):
    def test_closed_form_monotone_in_homophily(self):
        h = torch.tensor([0.1, 0.3, 0.5, 0.9])
        d = torch.log1p(torch.tensor([1.0, 2.0, 5.0, 10.0]))
        snr = torch.ones_like(h)
        k = closed_form_granularity(h, d, snr)
        self.assertTrue(torch.all(k[1:] >= k[:-1] - 1e-6))

    def test_agt_policy_shapes(self):
        device = torch.device("cpu")
        data = load_dataset("Cora", seed=0, device=device)
        env = GrainEnvironment(data, device=device)
        policy = env.make_agt_policy(use_calibration=True)
        k = policy.select_action(env.train_idx)
        self.assertEqual(k.shape[0], env.train_idx.shape[0])
        self.assertFalse(torch.isnan(k).any())

    def test_environment_eval(self):
        device = torch.device("cpu")
        data = load_dataset("Cora", seed=0, device=device)
        env = GrainEnvironment(data, device=device)
        policy = env.make_agt_policy(use_calibration=False)
        env.run_meta_training(policy, episodes=3)
        acc = env.evaluate("test")
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
