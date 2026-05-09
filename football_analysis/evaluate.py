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
    parser.add_argument(
        "--backend",
        choices=["auto", "sklearn", "torch"],
        default="auto",
        help="Artifact backend. 'auto' loads model.pt when present, otherwise model.joblib.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = args.artifact_dir / args.task
    metrics = evaluate_saved_model(args.task, artifact_dir, args.data_path, backend=args.backend)
    line = (
        f"{args.task} preset={metrics.get('evaluation_protocol', {}).get('search_preset_saved_in_artifact')} "
        f"backend={metrics.get('backend', 'unknown')} "
        f"n_test={metrics['test_rows']} "
        f"acc={metrics['accuracy']:.4f} balanced_acc={metrics['balanced_accuracy']:.4f} "
        f"kappa={metrics['cohen_kappa']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"w_f1={metrics['weighted_f1']:.4f} log_loss={metrics.get('log_loss', float('nan')):.4f}"
    )
    if metrics.get("top_3_accuracy") is not None:
        line += f" top_3={metrics['top_3_accuracy']:.4f}"
        if metrics.get("top_5_accuracy") is not None:
            line += f" top_5={metrics['top_5_accuracy']:.4f}"
        if metrics.get("top_10_accuracy") is not None:
            line += f" top_10={metrics['top_10_accuracy']:.4f}"
    print(line)


if __name__ == "__main__":
    main()
