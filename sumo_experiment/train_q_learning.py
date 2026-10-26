"""Train the existing tabular Q-learning agent against SUMO (small by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .q_learning_runner import DEFAULT_OUTPUT_DIR, train_q_learning
else:
    from q_learning_runner import DEFAULT_OUTPUT_DIR, train_q_learning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3, help="training episodes (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-horizon-seconds", type=float, default=60.0)
    parser.add_argument("--decision-interval-seconds", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        _, rows = train_q_learning(**vars(args))
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"SUMO Q-learning training failed: {error}") from error
    print(f"Trained {len(rows)} episodes; outputs saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
