from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_DATA_PATH
from .reporting import evaluate_saved_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved FootballAnalysisML model.")
    parser.add_argument("--task", choices=["role", "position"], required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = args.artifact_dir / args.task
    metrics = evaluate_saved_model(args.task, artifact_dir, args.data_path)
    line = (
        f"{args.task} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    if "top_3_accuracy" in metrics:
        line += f" top_3={metrics['top_3_accuracy']:.4f}"
    print(line)


if __name__ == "__main__":
    main()
