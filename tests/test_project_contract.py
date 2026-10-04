from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fl_attacks.constants import Attacks  # noqa: E402


class ProjectContractTests(unittest.TestCase):
    def test_experiments_are_model_poisoning_only(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
        )
        experiments = config["experiments"]
        attacks = [item["attack"] for item in experiments]

        self.assertEqual(
            attacks,
            [
                Attacks.NONE,
                Attacks.SIGN_FLIP,
                Attacks.GAUSSIAN_NOISE,
                Attacks.SCALING,
                Attacks.RANDOM_UPDATE,
            ],
        )

        defaults = config["defaults"]
        for old_key in ("label-source", "label-target", "poison-fraction"):
            self.assertNotIn(old_key, defaults)

    def test_flower_version_is_preserved(self) -> None:
        project = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = project["project"]["dependencies"]
        self.assertIn("flwr==1.32.1", dependencies)

    def test_old_flower_local_runtime_contract_is_preserved(self) -> None:
        launcher = (ROOT / "run.py").read_text(encoding="utf-8")
        expected_fragments = (
            "--fleet-api-address",
            "--control-api-address",
            "--clientappio-api-address",
            '"local-deployment"',
            '"--run-config"',
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, launcher)


if __name__ == "__main__":
    unittest.main()
