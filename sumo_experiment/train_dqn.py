"""Train the original 18-state DQN against SUMO (short by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .dqn_runner import DEFAULT_OUTPUT_DIR, train_dqn
else:
    from dqn_runner import DEFAULT_OUTPUT_DIR, train_dqn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2, help="training episodes (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-horizon-seconds", type=float, default=360.0)
    parser.add_argument("--decision-interval-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        _, rows = train_dqn(**vars(args))
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"SUMO original-DQN training failed: {error}") from error
    print(f"Trained {len(rows)} episodes; outputs saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
