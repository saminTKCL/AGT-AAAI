#!/usr/bin/env python3
"""Unit tests for GRAIN-CTRL."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extension"))
from deterministic_controller import DeterministicGranularityController
from grain_ctrl_policy import GRAINCTRLPolicy


class TestGRAINCtrl(unittest.TestCase):
    def test_deterministic_no_nan(self):
        ctrl = DeterministicGranularityController(32, 6, max_action=5.0)
        s = torch.randn(8, 32)
        a, u, stop = ctrl(s, hard=True)
        self.assertFalse(torch.isnan(a).any())
        self.assertFalse(torch.isnan(u).any())

    def test_policy_compatible_shape(self):
        pol = GRAINCTRLPolicy(32, 6, max_action=5.0)
        state = np.random.randn(4, 32).astype(np.float32)
        action = pol.select_action(state)
        self.assertEqual(action.shape[0], 4)

    def test_policy_single_state(self):
        pol = GRAINCTRLPolicy(32, 6)
        action = pol.select_action(np.random.randn(32).astype(np.float32))
        self.assertTrue(0 <= action <= 5 or action.shape == (1,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
