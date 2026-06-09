#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.load import load_dataset
from src.models.grain_env import GrainEnvironment
from src.models.multiview import MultiViewAggregator


class TestAGTImplicit(unittest.TestCase):
    def test_multiview_forward(self):
        data = load_dataset("Texas", seed=0)
        env = GrainEnvironment(data, device=torch.device("cpu"))
        mv = env.make_multiview_aggregator(use_calibration=True, implicit_k=5)
        idx = env.train_idx[:8]
        out = mv(idx)
        self.assertEqual(out.shape, (8, data.num_features))
        self.assertFalse(torch.isnan(out).any())

    def test_explicit_only(self):
        data = load_dataset("Texas", seed=0)
        env = GrainEnvironment(data, device=torch.device("cpu"))
        mv = env.make_multiview_aggregator(implicit_k=5)
        idx = env.test_idx[:4]
        full = mv(idx)
        expl = mv(idx, explicit_only=True)
        self.assertFalse(torch.allclose(full, expl))


if __name__ == "__main__":
    unittest.main(verbosity=2)
