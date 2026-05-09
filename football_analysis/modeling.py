from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
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


def model_candidates(
    random_state: int,
    task: str,
    search_preset: Literal["fast", "paper"] = "fast",
) -> list[tuple[str, Any, dict[str, list[Any]]]]:
    """Hyperparameter grids. `paper` expands position search toward the same richness as role (incl. max_depth=None).

    Trees keep `n_jobs=1`; CV parallelizes folds via RandomizedSearchCV `n_jobs`.
    """
    rf = RandomForestClassifier(random_state=random_state, n_jobs=1)
    et = ExtraTreesClassifier(random_state=random_state, n_jobs=1)

    rf_estimators = (
        [200, 300, 400, 500, 600]
        if search_preset == "paper"
        else [200, 300, 400, 500]
    )

    rf_full = (
        rf,
        {
            "model__n_estimators": rf_estimators,
            "model__max_depth": [None, 15, 25, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
    )
    et_full = (
        et,
        {
            "model__n_estimators": rf_estimators,
            "model__max_depth": [None, 15, 25, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__class_weight": [None, "balanced"],
        },
    )

    if task == "role":
        return [("random_forest", *rf_full), ("extra_trees", *et_full)]

    # position task
    if search_preset == "paper":
        return [("random_forest", *rf_full), ("extra_trees", *et_full)]

    return [
        (
            "random_forest",
            rf,
            {
                "model__n_estimators": [120, 200, 300],
                "model__max_depth": [18, 28, 40],
                "model__min_samples_split": [2, 6],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt", "log2"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        (
            "extra_trees",
            et,
            {
                "model__n_estimators": [120, 200, 300],
                "model__max_depth": [18, 28, 40],
                "model__min_samples_split": [2, 6],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt", "log2"],
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


def compute_metrics_from_outputs(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probas: np.ndarray | None,
    label_encoder: LabelEncoder,
    task: str,
) -> dict[str, Any]:
    labels = np.arange(len(label_encoder.classes_))
    n_classes = len(labels)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
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
        "confusion_matrix_labels": label_encoder.classes_.tolist(),
    }

    if probas is not None:
        eps = 1e-15
        probas = np.clip(probas, eps, 1.0 - eps)
        metrics["log_loss"] = float(log_loss(y_test, probas, labels=labels))
        if task == "position":
            for k in (3, 5, 10):
                if k <= n_classes:
                    metrics[f"top_{k}_accuracy"] = float(
                        top_k_accuracy_score(y_test, probas, k=k, labels=labels)
                    )

    return metrics


def compute_metrics(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    task: str,
) -> dict[str, Any]:
    y_pred = estimator.predict(X_test)
    probas = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
    return compute_metrics_from_outputs(y_test, y_pred, probas, label_encoder, task)


def _effective_cv_jobs(requested: int) -> int:
    """Prefer parallel CV folds; unittest / single-core envs can set FOOTBALL_ANALYSIS_CV_JOBS=1."""
    if requested == -1:
        env = os.environ.get("FOOTBALL_ANALYSIS_CV_JOBS", "").strip()
        if env.isdigit():
            return int(env)
    return requested


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
    search_cv_n_jobs: int = -1,
    search_preset: Literal["fast", "paper"] = "fast",
) -> dict[str, Any]:
    X = frame[numeric_features + categorical_features].copy()
    y_raw = frame[target_col].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    cv_jobs = _effective_cv_jobs(search_cv_n_jobs)

    row_ids = np.arange(len(frame))
    X_train, X_test, y_train, y_test, row_train, row_test = train_test_split(
        X,
        y,
        row_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(
        f"[{task}] preset={search_preset} train_rows={len(X_train)} "
        f"test_rows={len(X_test)} n_classes={len(label_encoder.classes_)} "
        f"(search: {search_iterations} iters × {cv_folds} folds, CV n_jobs={cv_jobs})",
        flush=True,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_name = ""
    best_search: RandomizedSearchCV | None = None
    candidate_results: list[dict[str, Any]] = []

    for name, estimator, params in model_candidates(
        random_state, task, search_preset
    ):
        # Fresh preprocessor avoids sharing fitted state across candidate pipelines.
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("model", estimator)]
        )
        print(f"[{task}] RandomizedSearchCV: {name} …", flush=True)
        search_verbose = int(os.environ.get("FOOTBALL_ANALYSIS_SEARCH_VERBOSE", "1"))
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=search_iterations,
            scoring="f1_macro",
            n_jobs=cv_jobs,
            cv=cv,
            refit=True,
            random_state=random_state,
            verbose=search_verbose,
        )
        search.fit(X_train, y_train)
        print(
            f"[{task}] {name} done — best_cv_macro_f1={search.best_score_:.4f}",
            flush=True,
        )
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
            "search_preset": search_preset,
            "search_cv_n_jobs": cv_jobs,
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
        "search_preset": search_preset,
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
