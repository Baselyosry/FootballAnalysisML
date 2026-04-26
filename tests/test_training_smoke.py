from __future__ import annotations

import json
import unittest
from pathlib import Path

from football_analysis.train_position import train_position_model
from football_analysis.train_role import train_role_model


class TrainingSmokeTests(unittest.TestCase):
    def test_role_training_smoke(self) -> None:
        output_dir = Path("artifacts") / "test_role_smoke"
        result = train_role_model(
            output_dir=output_dir,
            search_iterations=1,
            cv_folds=2,
            sample_size=800,
            compute_legacy_baseline=False,
        )
        self.assertTrue((output_dir / "model.joblib").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(metrics["accuracy"], 0.5)
        self.assertEqual(result["metrics"]["task"], "role")

    def test_position_training_smoke(self) -> None:
        output_dir = Path("artifacts") / "test_position_smoke"
        result = train_position_model(
            output_dir=output_dir,
            search_iterations=1,
            cv_folds=2,
            sample_size=1000,
        )
        self.assertTrue((output_dir / "model.joblib").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("top_3_accuracy", metrics)
        self.assertEqual(result["metrics"]["task"], "position")


if __name__ == "__main__":
    unittest.main()
