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
) -> dict[str, object]:
    raw_data = load_raw_data(data_path)
    frame, numeric_features, categorical_features, target_col = build_model_frame(
        raw_data, "role"
    )
    if sample_size is not None:
        frame = stratified_sample(frame, target_col, sample_size, random_state)

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

    if compute_legacy_baseline:
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
    )
    metrics = result["metrics"]
    print(
        f"role preset={metrics.get('search_preset')} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f} "
        f"kappa={metrics['cohen_kappa']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"log_loss={metrics.get('log_loss', float('nan')):.4f} "
        f"model={metrics['best_model_name']}"
    )


if __name__ == "__main__":
    main()
