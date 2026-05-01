from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .data import LEGACY_CLASSIFICATION_PATH


def compute_legacy_role_baseline(
    path: Path | str | None = None,
    random_state: int = 15,
) -> dict[str, float] | None:
    baseline_path = Path(path) if path is not None else LEGACY_CLASSIFICATION_PATH
    if not baseline_path.exists():
        return None

    data = pd.read_csv(baseline_path).fillna(0)
    if "Position 1" not in data.columns:
        return None

    X = data.drop(columns=["Position 1"])
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["Position 1"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }
