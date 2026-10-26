"""Greedily evaluate a saved SUMO Q-learning table."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .q_learning_runner import DEFAULT_OUTPUT_DIR, evaluate_q_learning
else:
    from q_learning_runner import DEFAULT_OUTPUT_DIR, evaluate_q_learning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="evaluation episodes (default: 1)")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT_DIR / "q_table.npz")
    parser.add_argument("--episode-horizon-seconds", type=float, default=60.0)
    parser.add_argument("--decision-interval-seconds", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = vars(build_parser().parse_args())
    args["model_path"] = args.pop("model")
    try:
        rows = evaluate_q_learning(**args)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"SUMO Q-learning evaluation failed: {error}") from error
    print(f"Evaluated {len(rows)} episodes; outputs saved in {args['output_dir'].resolve()}")


if __name__ == "__main__":
    main()
