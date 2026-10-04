from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fl_attacks.train_utils import evaluate_classification  # noqa: E402


class IdentityLogits(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class ClassificationMetricTests(unittest.TestCase):
    def test_macro_and_weighted_metrics(self) -> None:
        logits = torch.tensor(
            [
                [5.0, 0.0, 0.0],  # true 0 -> pred 0
                [0.0, 5.0, 0.0],  # true 1 -> pred 1
                [0.0, 5.0, 0.0],  # true 2 -> pred 1
                [0.0, 0.0, 5.0],  # true 2 -> pred 2
            ]
        )
        labels = torch.tensor([0, 1, 2, 2])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

        metrics, matrix, details = evaluate_classification(
            model=IdentityLogits(),
            loader=loader,
            device=torch.device("cpu"),
            num_classes=3,
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=7)
        self.assertAlmostEqual(metrics["precision_macro"], 5.0 / 6.0, places=7)
        self.assertAlmostEqual(metrics["recall_macro"], 5.0 / 6.0, places=7)
        self.assertAlmostEqual(metrics["f1_macro"], 7.0 / 9.0, places=7)
        self.assertAlmostEqual(metrics["precision_weighted"], 0.875, places=7)
        self.assertAlmostEqual(metrics["recall_weighted"], 0.75, places=7)
        self.assertAlmostEqual(metrics["f1_weighted"], 0.75, places=7)

        expected_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 1],
            ]
        ).numpy()
        self.assertTrue((matrix == expected_matrix).all())
        self.assertEqual(details["support_per_class"], [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
