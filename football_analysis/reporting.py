from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .data import build_model_frame, load_raw_data
from .modeling import compute_metrics, load_artifact, save_json


def evaluate_saved_model(
    task: str,
    artifact_dir: Path,
    data_path: Path | str | None = None,
) -> dict[str, Any]:
    artifact = load_artifact(artifact_dir / "model.joblib")
    raw_data = load_raw_data(data_path)
    frame, _, _, target_col = build_model_frame(raw_data, task)
    test_frame = frame.iloc[artifact["test_indices"]].copy()

    X_test = test_frame[artifact["numeric_features"] + artifact["categorical_features"]]
    label_encoder = LabelEncoder()
    label_encoder.fit(artifact["label_classes"])
    y_test = label_encoder.transform(test_frame[target_col])

    metrics = compute_metrics(
        artifact["estimator"],
        X_test,
        y_test,
        label_encoder,
        task,
    )
    metrics.update(
        {
            "task": task,
            "model_name": artifact["model_name"],
            "test_rows": int(len(test_frame)),
        }
    )
    save_json(artifact_dir / "evaluation.json", metrics)
    return metrics
