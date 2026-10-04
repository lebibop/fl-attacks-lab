from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fl_attacks.attacks import (  # noqa: E402
    apply_model_attack,
    attack_is_active,
    malicious_client_ids,
)
from fl_attacks.constants import Attacks  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0, -1.0]))


class ModelAttackTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.model = TinyModel()
        self.global_state = {"weight": torch.tensor([1.0, 1.0])}
        self.base_config = {
            "model-attack-scale": 2.0,
            "noise-std": 0.0,
            "scaling-factor": 3.0,
            "random-update-scale": 1.5,
        }

    def test_sign_flip(self) -> None:
        apply_model_attack(
            self.model,
            self.global_state,
            Attacks.SIGN_FLIP,
            self.base_config,
        )
        expected = torch.tensor([-1.0, 5.0])
        self.assertTrue(torch.allclose(self.model.weight.detach(), expected))

    def test_scaling(self) -> None:
        apply_model_attack(
            self.model,
            self.global_state,
            Attacks.SCALING,
            self.base_config,
        )
        expected = torch.tensor([4.0, -5.0])
        self.assertTrue(torch.allclose(self.model.weight.detach(), expected))

    def test_gaussian_noise_zero_is_noop(self) -> None:
        original = self.model.weight.detach().clone()
        apply_model_attack(
            self.model,
            self.global_state,
            Attacks.GAUSSIAN_NOISE,
            self.base_config,
        )
        self.assertTrue(torch.allclose(self.model.weight.detach(), original))

    def test_random_update_has_requested_norm(self) -> None:
        honest_delta = self.model.weight.detach() - self.global_state["weight"]
        honest_norm = torch.linalg.vector_norm(honest_delta).item()
        diagnostics = apply_model_attack(
            self.model,
            self.global_state,
            Attacks.RANDOM_UPDATE,
            self.base_config,
        )
        self.assertAlmostEqual(
            diagnostics.update_l2_after,
            1.5 * honest_norm,
            places=5,
        )

    def test_malicious_selection_is_reproducible(self) -> None:
        first = malicious_client_ids(10, 0.3, seed=42)
        second = malicious_client_ids(10, 0.3, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_attack_window(self) -> None:
        malicious = malicious_client_ids(5, 1.0, seed=42)
        self.assertIn(0, malicious)
        self.assertFalse(
            attack_is_active(0, 1, 5, 1.0, 2, 3, seed=42)
        )
        self.assertTrue(
            attack_is_active(0, 2, 5, 1.0, 2, 3, seed=42)
        )
        self.assertFalse(
            attack_is_active(0, 4, 5, 1.0, 2, 3, seed=42)
        )


if __name__ == "__main__":
    unittest.main()
