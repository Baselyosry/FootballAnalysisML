from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .modeling import compute_metrics_from_outputs, save_json


@dataclass(frozen=True)
class TorchTabularConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_size: float = 0.2
    patience: int = 8
    dropout: float = 0.2
    hidden_dims: tuple[int, ...] = (256, 128)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _embedding_dim(cardinality: int) -> int:
    return min(16, max(4, (cardinality + 1) // 2))


def _build_categorical_maps(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, dict[str, int]]:
    feature_maps: dict[str, dict[str, int]] = {}
    for feature in categorical_features:
        values = frame[feature].fillna("__MISSING__").astype(str)
        uniques = sorted(values.unique().tolist())
        feature_maps[feature] = {value: idx + 1 for idx, value in enumerate(uniques)}
    return feature_maps


def _encode_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_medians: dict[str, float],
    numeric_means: dict[str, float],
    numeric_stds: dict[str, float],
    categorical_maps: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    numeric = frame[numeric_features].copy()
    for feature in numeric_features:
        numeric[feature] = numeric[feature].fillna(numeric_medians[feature])
    numeric_np = numeric.to_numpy(dtype=np.float32, copy=True)
    means = np.array([numeric_means[feature] for feature in numeric_features], dtype=np.float32)
    stds = np.array([numeric_stds[feature] for feature in numeric_features], dtype=np.float32)
    numeric_np = (numeric_np - means) / stds

    categorical_arrays: list[np.ndarray] = []
    for feature in categorical_features:
        feature_map = categorical_maps[feature]
        mapped = (
            frame[feature]
            .fillna("__MISSING__")
            .astype(str)
            .map(feature_map)
            .fillna(0)
            .to_numpy(dtype=np.int64, copy=True)
        )
        categorical_arrays.append(mapped)
    categorical_np = (
        np.stack(categorical_arrays, axis=1) if categorical_arrays else np.zeros((len(frame), 0), dtype=np.int64)
    )
    return numeric_np, categorical_np


class TabularNet(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list[int],
        hidden_dims: tuple[int, ...],
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, _embedding_dim(cardinality)) for cardinality in cat_cardinalities]
        )
        embed_total_dim = sum(embedding.embedding_dim for embedding in self.embeddings)
        input_dim = num_numeric + embed_total_dim
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, numeric_x: torch.Tensor, categorical_x: torch.Tensor) -> torch.Tensor:
        embeddings = [embedding(categorical_x[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        embedded = torch.cat(embeddings, dim=1) if embeddings else torch.empty((numeric_x.size(0), 0), device=numeric_x.device)
        combined = torch.cat([numeric_x, embedded], dim=1)
        return self.mlp(combined)


def _predict(
    model: TabularNet,
    numeric_np: np.ndarray,
    categorical_np: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dataset = TensorDataset(
        torch.tensor(numeric_np, dtype=torch.float32),
        torch.tensor(categorical_np, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_numeric, batch_categorical in loader:
            logits = model(batch_numeric, batch_categorical)
            logits_batches.append(logits.cpu().numpy())
    logits_np = np.concatenate(logits_batches, axis=0)
    probas = torch.softmax(torch.tensor(logits_np), dim=1).numpy()
    preds = probas.argmax(axis=1)
    return preds, probas


def predict_torch_artifact(artifact: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model_config = artifact["model_config"]
    model = TabularNet(
        num_numeric=model_config["num_numeric"],
        cat_cardinalities=model_config["cat_cardinalities"],
        hidden_dims=tuple(model_config["hidden_dims"]),
        dropout=model_config["dropout"],
        num_classes=model_config["num_classes"],
    )
    model.load_state_dict(artifact["model_state_dict"])
    stats = artifact["preprocessing"]
    numeric_np, categorical_np = _encode_features(
        frame=frame,
        numeric_features=artifact["numeric_features"],
        categorical_features=artifact["categorical_features"],
        numeric_medians=stats["numeric_medians"],
        numeric_means=stats["numeric_means"],
        numeric_stds=stats["numeric_stds"],
        categorical_maps=stats["categorical_maps"],
    )
    return _predict(model, numeric_np, categorical_np, batch_size=model_config["batch_size"])


def load_torch_artifact(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def train_torch_tabular_model(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
    task: str,
    output_dir: Path,
    random_state: int,
    test_size: float,
    config: TorchTabularConfig,
) -> dict[str, Any]:
    _set_seed(random_state)

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
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=config.val_size,
        random_state=random_state,
        stratify=y_train,
    )

    numeric_medians = {feature: float(X_fit[feature].median()) for feature in numeric_features}
    numeric_filled = X_fit[numeric_features].fillna(pd.Series(numeric_medians))
    numeric_means = {feature: float(numeric_filled[feature].mean()) for feature in numeric_features}
    numeric_stds = {feature: float(max(numeric_filled[feature].std(ddof=0), 1e-6)) for feature in numeric_features}
    categorical_maps = _build_categorical_maps(X_fit, categorical_features)

    fit_numeric, fit_categorical = _encode_features(
        frame=X_fit,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_medians=numeric_medians,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_maps=categorical_maps,
    )
    val_numeric, val_categorical = _encode_features(
        frame=X_val,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_medians=numeric_medians,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_maps=categorical_maps,
    )
    test_numeric, test_categorical = _encode_features(
        frame=X_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_medians=numeric_medians,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_maps=categorical_maps,
    )

    cat_cardinalities = [len(categorical_maps[feature]) for feature in categorical_features]
    model = TabularNet(
        num_numeric=len(numeric_features),
        cat_cardinalities=cat_cardinalities,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        num_classes=len(label_encoder.classes_),
    )

    fit_dataset = TensorDataset(
        torch.tensor(fit_numeric, dtype=torch.float32),
        torch.tensor(fit_categorical, dtype=torch.long),
        torch.tensor(y_fit, dtype=torch.long),
    )
    fit_loader = DataLoader(fit_dataset, batch_size=config.batch_size, shuffle=True)

    class_weights = None
    if task == "position":
        counts = np.bincount(y_fit, minlength=len(label_encoder.classes_))
        total = max(int(counts.sum()), 1)
        class_weights = total / np.maximum(counts, 1)
        class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": []}
    best_state = None
    best_f1 = float("-inf")
    best_epoch = 0
    no_improve = 0

    val_numeric_t = torch.tensor(val_numeric, dtype=torch.float32)
    val_categorical_t = torch.tensor(val_categorical, dtype=torch.long)
    val_y_t = torch.tensor(y_val, dtype=torch.long)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for batch_numeric, batch_categorical, batch_y in fit_loader:
            optimizer.zero_grad()
            logits = model(batch_numeric, batch_categorical)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(batch_y.size(0))
            n_samples += int(batch_y.size(0))
        train_loss = running_loss / max(n_samples, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_numeric_t, val_categorical_t)
            val_loss = float(criterion(val_logits, val_y_t).item())
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        val_pred = val_probs.argmax(axis=1)
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        val_acc = float((val_pred == y_val).mean())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_accuracy"].append(val_acc)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Torch training did not produce a valid model state.")

    model.load_state_dict(best_state)
    y_pred, probas = _predict(model, test_numeric, test_categorical, batch_size=config.batch_size)
    metrics = compute_metrics_from_outputs(y_test, y_pred, probas, label_encoder, task)
    metrics.update(
        {
            "task": task,
            "target_column": target_col,
            "backend": "torch",
            "model_name": "torch_tabular_mlp",
            "random_state": random_state,
            "test_size": test_size,
            "search_preset": "torch_tabular",
            "train_rows": int(len(X_train)),
            "fit_rows": int(len(X_fit)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_f1),
            "torch_config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "val_size": config.val_size,
                "patience": config.patience,
                "dropout": config.dropout,
                "hidden_dims": list(config.hidden_dims),
            },
            "training_history": history,
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
        "model_name": "torch_tabular_mlp",
        "search_preset": "torch_tabular",
        "preprocessing": {
            "numeric_medians": numeric_medians,
            "numeric_means": numeric_means,
            "numeric_stds": numeric_stds,
            "categorical_maps": categorical_maps,
        },
        "model_config": {
            "num_numeric": len(numeric_features),
            "cat_cardinalities": cat_cardinalities,
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "num_classes": int(len(label_encoder.classes_)),
            "batch_size": config.batch_size,
        },
        "model_state_dict": best_state,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"
    torch.save(artifact, model_path)
    save_json(metrics_path, metrics)

    return {
        "artifact_path": model_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }
