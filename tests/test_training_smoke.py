from __future__ import annotations

import json
import os
import unittest
from importlib.util import find_spec
from pathlib import Path

from football_analysis.train_position import train_position_model
from football_analysis.train_role import train_role_model


class TrainingSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.setdefault("FOOTBALL_ANALYSIS_SEARCH_VERBOSE", "0")

    def test_role_training_smoke(self) -> None:
        output_dir = Path("artifacts") / "test_role_smoke"
        result = train_role_model(
            output_dir=output_dir,
            search_iterations=1,
            cv_folds=2,
            sample_size=800,
            compute_legacy_baseline=False,
            search_cv_n_jobs=1,
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
            search_cv_n_jobs=1,
        )
        self.assertTrue((output_dir / "model.joblib").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("top_3_accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("cohen_kappa", metrics)
        self.assertEqual(result["metrics"]["task"], "position")

    def test_paper_grid_position_minimal_fit(self) -> None:
        """Ensure `preset=paper` path (rich grid incl. unlimited depth on RF) executes."""
        output_dir = Path("artifacts") / "test_position_paper_smoke"
        result = train_position_model(
            output_dir=output_dir,
            search_iterations=1,
            cv_folds=2,
            sample_size=900,
            search_cv_n_jobs=1,
            search_preset="paper",
        )
        self.assertEqual(result["metrics"]["search_preset"], "paper")
        self.assertIn("top_5_accuracy", result["metrics"])

    @unittest.skipUnless(find_spec("torch") is not None, "torch is required for torch backend smoke tests")
    def test_role_training_torch_smoke(self) -> None:
        output_dir = Path("artifacts") / "test_role_torch_smoke"
        result = train_role_model(
            output_dir=output_dir,
            backend="torch",
            sample_size=1000,
            torch_epochs=3,
            torch_patience=2,
            torch_batch_size=256,
        )
        self.assertTrue((output_dir / "model.pt").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["backend"], "torch")
        self.assertEqual(result["metrics"]["task"], "role")
        self.assertIn("training_history", metrics)

    @unittest.skipUnless(find_spec("torch") is not None, "torch is required for torch backend smoke tests")
    def test_position_training_torch_smoke(self) -> None:
        output_dir = Path("artifacts") / "test_position_torch_smoke"
        result = train_position_model(
            output_dir=output_dir,
            backend="torch",
            sample_size=1200,
            torch_epochs=3,
            torch_patience=2,
            torch_batch_size=256,
        )
        self.assertTrue((output_dir / "model.pt").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["backend"], "torch")
        self.assertEqual(result["metrics"]["task"], "position")
        self.assertIn("top_3_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
