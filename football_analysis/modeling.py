from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def model_candidates(random_state: int) -> list[tuple[str, Any, dict[str, list[Any]]]]:
    return [
        (
            "random_forest",
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [200, 300, 400, 500],
                "model__max_depth": [None, 15, 25, 40],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [200, 300, 400, 500],
                "model__max_depth": [None, 15, 25, 40],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5],
                "model__class_weight": [None, "balanced"],
            },
        ),
    ]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def compute_metrics(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    task: str,
) -> dict[str, Any]:
    y_pred = estimator.predict(X_test)
    labels = np.arange(len(label_encoder.classes_))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "labels": label_encoder.classes_.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=label_encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }

    if task == "position" and hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)
        metrics["top_3_accuracy"] = float(
            top_k_accuracy_score(y_test, y_score, k=3, labels=labels)
        )

    return metrics


def train_best_model(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
    task: str,
    output_dir: Path,
    random_state: int,
    test_size: float,
    search_iterations: int,
    cv_folds: int,
) -> dict[str, Any]:
    X = frame[numeric_features + categorical_features].copy()
    y_raw = frame[target_col].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    row_ids = np.arange(len(frame))
    X_train, X_test, y_train, y_test, row_train, row_test = train_test_split(
        X,
        y,
        row_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_name = ""
    best_search: RandomizedSearchCV | None = None
    candidate_results: list[dict[str, Any]] = []

    for name, estimator, params in model_candidates(random_state):
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("model", estimator)]
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=search_iterations,
            scoring="f1_macro",
            n_jobs=1,
            cv=cv,
            refit=True,
            random_state=random_state,
            verbose=0,
        )
        search.fit(X_train, y_train)
        candidate_results.append(
            {
                "model_name": name,
                "best_cv_score": float(search.best_score_),
                "best_params": search.best_params_,
            }
        )
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name = name
            best_search = search

    if best_search is None:
        raise RuntimeError("No model candidates were trained.")

    metrics = compute_metrics(best_search.best_estimator_, X_test, y_test, label_encoder, task)
    metrics.update(
        {
            "task": task,
            "target_column": target_col,
            "random_state": random_state,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "search_iterations": search_iterations,
            "best_model_name": best_name,
            "best_cv_score": float(best_search.best_score_),
            "best_params": best_search.best_params_,
            "candidate_results": candidate_results,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
    )

    artifact = {
        "task": task,
        "target_column": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "label_classes": label_encoder.classes_.tolist(),
        "test_indices": row_test.tolist(),
        "random_state": random_state,
        "test_size": test_size,
        "model_name": best_name,
        "estimator": best_search.best_estimator_,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_dir / "model.joblib")
    save_json(output_dir / "metrics.json", metrics)

    return {
        "artifact_path": output_dir / "model.joblib",
        "metrics_path": output_dir / "metrics.json",
        "metrics": metrics,
    }


def load_artifact(path: Path) -> dict[str, Any]:
    return joblib.load(path)
