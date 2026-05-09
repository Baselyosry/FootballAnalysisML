from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from .data import DEFAULT_DATA_PATH, build_model_frame, load_raw_data, stratified_sample
from .legacy import compute_legacy_role_baseline
from .modeling import save_json, train_best_model


def train_role_model(
    data_path: Path | str = DEFAULT_DATA_PATH,
    output_dir: Path | str = Path("artifacts") / "role",
    random_state: int = 15,
    test_size: float = 0.2,
    search_iterations: int = 8,
    cv_folds: int = 5,
    sample_size: int | None = None,
    compute_legacy_baseline: bool = True,
    search_cv_n_jobs: int = -1,
    search_preset: Literal["fast", "paper"] = "fast",
    backend: Literal["sklearn", "torch"] = "sklearn",
    torch_epochs: int = 40,
    torch_batch_size: int = 256,
    torch_learning_rate: float = 1e-3,
    torch_weight_decay: float = 1e-4,
    torch_val_size: float = 0.2,
    torch_patience: int = 8,
    torch_dropout: float = 0.2,
) -> dict[str, object]:
    raw_data = load_raw_data(data_path)
    frame, numeric_features, categorical_features, target_col = build_model_frame(
        raw_data, "role"
    )
    if sample_size is not None:
        frame = stratified_sample(frame, target_col, sample_size, random_state)

    if backend == "torch":
        from .nn_tabular import TorchTabularConfig, train_torch_tabular_model

        preset_epochs = 40 if search_preset == "fast" else 80
        result = train_torch_tabular_model(
            frame=frame,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_col=target_col,
            task="role",
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
    else:
        result = train_best_model(
            frame=frame,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_col=target_col,
            task="role",
            output_dir=Path(output_dir),
            random_state=random_state,
            test_size=test_size,
            search_iterations=search_iterations,
            cv_folds=cv_folds,
            search_cv_n_jobs=search_cv_n_jobs,
            search_preset=search_preset,
        )

    if backend == "sklearn" and compute_legacy_baseline:
        legacy = compute_legacy_role_baseline(random_state=random_state)
        if legacy is not None:
            result["metrics"]["legacy_baseline"] = legacy
            save_json(Path(output_dir) / "metrics.json", result["metrics"])

    return result


def _preset_role_iters_cv(preset: str, it_override: int | None, cv_override: int | None) -> tuple[int, int]:
    base = {"fast": (8, 5), "paper": (28, 5)}
    bi, bf = base[preset]
    return (
        bi if it_override is None else it_override,
        bf if cv_override is None else cv_override,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the 4-role football classifier.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "role")
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
        help="'paper' expands n_estimators (adds 600) and boosts default search_iterations/cv folds.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=None,
        help="Override randomized-search trials per model family (preset default if omitted).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override CV folds (preset default if omitted).",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--skip-legacy-baseline",
        action="store_true",
        help="Skip comparison against the legacy processed classification CSV baseline.",
    )
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "joblib parallelism for RandomizedSearchCV folds (default -1 = all CPUs; "
            "set 1 to debug). Overrides FOOTBALL_ANALYSIS_CV_JOBS."
        ),
    )
    parser.add_argument("--torch-epochs", type=int, default=40)
    parser.add_argument("--torch-batch-size", type=int, default=256)
    parser.add_argument("--torch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--torch-val-size", type=float, default=0.2)
    parser.add_argument("--torch-patience", type=int, default=8)
    parser.add_argument("--torch-dropout", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    iters, folds = _preset_role_iters_cv(
        args.preset, args.search_iterations, args.cv_folds
    )
    result = train_role_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        search_iterations=iters,
        cv_folds=folds,
        sample_size=args.sample_size,
        compute_legacy_baseline=not args.skip_legacy_baseline,
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
    print(
        f"role preset={metrics.get('search_preset')} "
        f"backend={metrics.get('backend', 'sklearn')} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f} "
        f"kappa={metrics['cohen_kappa']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"log_loss={metrics.get('log_loss', float('nan')):.4f} "
        f"model={metrics.get('best_model_name', metrics.get('model_name', 'unknown'))}"
    )


if __name__ == "__main__":
    main()
