from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_DATA_PATH, build_model_frame, load_raw_data, stratified_sample
from .modeling import train_best_model


def train_position_model(
    data_path: Path | str = DEFAULT_DATA_PATH,
    output_dir: Path | str = Path("artifacts") / "position",
    random_state: int = 15,
    test_size: float = 0.2,
    search_iterations: int = 8,
    cv_folds: int = 5,
    sample_size: int | None = None,
) -> dict[str, object]:
    raw_data = load_raw_data(data_path)
    frame, numeric_features, categorical_features, target_col = build_model_frame(
        raw_data, "position"
    )
    if sample_size is not None:
        frame = stratified_sample(frame, target_col, sample_size, random_state)

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
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the exact-position football classifier.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "position")
    parser.add_argument("--random-state", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--search-iterations", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_position_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        search_iterations=args.search_iterations,
        cv_folds=args.cv_folds,
        sample_size=args.sample_size,
    )
    metrics = result["metrics"]
    print(
        f"position accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"top_3={metrics.get('top_3_accuracy', 0.0):.4f} "
        f"model={metrics['best_model_name']}"
    )


if __name__ == "__main__":
    main()
