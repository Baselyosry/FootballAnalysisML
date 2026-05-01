from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .data import DEFAULT_DATA_PATH, build_model_frame, load_raw_data
from .modeling import compute_metrics, load_artifact, save_json


def evaluate_saved_model(
    task: str,
    artifact_dir: Path,
    data_path: Path | str | None = None,
) -> dict[str, Any]:
    data_path_resolved = DEFAULT_DATA_PATH if data_path is None else Path(data_path)
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
    protocol: dict[str, Any] = {
        "evaluation_name": "holdout_repeat",
        "description": (
            "Re-evaluate the estimator on the identical rows indexed at training time "
            "(artifact['test_indices']). Indices refer to positions in "
            "`build_model_frame(load_raw_data())[...]` row order."
        ),
        "data_csv_path_resolved": str(data_path_resolved.resolve()),
        "stored_test_indices_count": len(artifact["test_indices"]),
        "label_order_in_metrics": artifact["label_classes"],
        "search_preset_saved_in_artifact": artifact.get(
            "search_preset", artifact.get("search_preset_at_train", "unknown")
        ),
        "parity_requirement_for_identical_numbers": (
            "Training must use the same raw CSV rows in the same order; "
            "missing-indicator columns derive from whichever rows remain after dropping "
            "unmapped targets — only then are indices comparable."
        ),
    }

    metrics.update(
        {
            "task": task,
            "model_name": artifact["model_name"],
            "test_rows": int(len(test_frame)),
            "evaluation_protocol": protocol,
        }
    )
    save_json(artifact_dir / "evaluation.json", metrics)
    return metrics
