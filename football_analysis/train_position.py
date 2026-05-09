from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from .data import DEFAULT_DATA_PATH, build_model_frame, load_raw_data, stratified_sample
from .modeling import train_best_model


def _preset_position_iters_cv(
    preset: str, it_override: int | None, cv_override: int | None
) -> tuple[int, int]:
    base = {"fast": (5, 3), "paper": (28, 5)}
    bi, bf = base[preset]
    return (
        bi if it_override is None else it_override,
        bf if cv_override is None else cv_override,
    )


def train_position_model(
    data_path: Path | str = DEFAULT_DATA_PATH,
    output_dir: Path | str = Path("artifacts") / "position",
    random_state: int = 15,
    test_size: float = 0.2,
    search_iterations: int = 5,
    cv_folds: int = 3,
    sample_size: int | None = None,
    search_cv_n_jobs: int = -1,
    search_preset: Literal["fast", "paper"] = "fast",
    backend: Literal["sklearn", "torch"] = "sklearn",
    torch_epochs: int = 45,
    torch_batch_size: int = 256,
    torch_learning_rate: float = 1e-3,
    torch_weight_decay: float = 1e-4,
    torch_val_size: float = 0.2,
    torch_patience: int = 8,
    torch_dropout: float = 0.25,
) -> dict[str, object]:
    raw_data = load_raw_data(data_path)
    frame, numeric_features, categorical_features, target_col = build_model_frame(
        raw_data, "position"
    )
    if sample_size is not None:
        frame = stratified_sample(frame, target_col, sample_size, random_state)

    if backend == "torch":
        from .nn_tabular import TorchTabularConfig, train_torch_tabular_model

        preset_epochs = 45 if search_preset == "fast" else 90
        return train_torch_tabular_model(
            frame=frame,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_col=target_col,
            task="position",
            output_dir=Path(output_dir),
            random_state=random_state,
            test_size=test_size,
            config=TorchTabularConfig(
                epochs=torch_epochs or preset_epochs,
                batch_size=torch_batch_size,
                learning_rate=torch_learning_rate,
                weight_decay=torch_weight_decay,
                val_size=torch_val_size,
                patience=torch_patience,
                dropout=torch_dropout,
            ),
        )
    return train_best_model(
        frame=frame,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
        task="position",
        output_dir=Path(output_dir),
        random_state=random_state,
        test_size=test_size,
        search_iterations=search_iterations,
        cv_folds=cv_folds,
        search_cv_n_jobs=search_cv_n_jobs,
        search_preset=search_preset,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the exact-position football classifier.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "position")
    parser.add_argument("--random-state", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--backend",
        choices=["sklearn", "torch"],
        default="sklearn",
        help="Training backend. 'torch' trains a tabular neural network with embeddings.",
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "paper"],
        default="fast",
        help="'paper' uses the rich tree grid (same as role, incl. max_depth=None); expect long runs.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=None,
        help="Trials per model family (preset default if omitted).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Stratified CV folds (preset default if omitted).",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=-1,
        metavar="N",
        help="Parallel CV folds (default -1 = all CPUs; use 1 to debug serially).",
    )
    parser.add_argument("--torch-epochs", type=int, default=45)
    parser.add_argument("--torch-batch-size", type=int, default=256)
    parser.add_argument("--torch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--torch-val-size", type=float, default=0.2)
    parser.add_argument("--torch-patience", type=int, default=8)
    parser.add_argument("--torch-dropout", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    iters, folds = _preset_position_iters_cv(
        args.preset, args.search_iterations, args.cv_folds
    )
    result = train_position_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        search_iterations=iters,
        cv_folds=folds,
        sample_size=args.sample_size,
        search_cv_n_jobs=args.cv_jobs,
        search_preset=args.preset,
        backend=args.backend,
        torch_epochs=args.torch_epochs,
        torch_batch_size=args.torch_batch_size,
        torch_learning_rate=args.torch_learning_rate,
        torch_weight_decay=args.torch_weight_decay,
        torch_val_size=args.torch_val_size,
        torch_patience=args.torch_patience,
        torch_dropout=args.torch_dropout,
    )
    metrics = result["metrics"]
    top5 = metrics.get("top_5_accuracy", float("nan"))
    print(
        f"position preset={metrics.get('search_preset')} "
        f"backend={metrics.get('backend', 'sklearn')} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f} "
        f"kappa={metrics['cohen_kappa']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"top_3={metrics.get('top_3_accuracy', float('nan')):.4f} "
        f"top_5={top5:.4f} log_loss={metrics.get('log_loss', float('nan')):.4f} "
        f"model={metrics.get('best_model_name', metrics.get('model_name', 'unknown'))}"
    )


if __name__ == "__main__":
    main()
