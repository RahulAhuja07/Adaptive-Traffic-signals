"""Greedily evaluate a saved state-only DQN SUMO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .state_only_dqn_runner import DEFAULT_OUTPUT_DIR, evaluate_state_only_dqn
else:
    from state_only_dqn_runner import DEFAULT_OUTPUT_DIR, evaluate_state_only_dqn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="evaluation episodes (default: 1)")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT_DIR / "state_only_dqn_model.pt")
    parser.add_argument("--episode-horizon-seconds", type=float, default=360.0)
    parser.add_argument("--decision-interval-seconds", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = vars(build_parser().parse_args())
    args["model_path"] = args.pop("model")
    try:
        rows = evaluate_state_only_dqn(**args)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"SUMO state-only DQN evaluation failed: {error}") from error
    print(f"Evaluated {len(rows)} episodes; outputs saved in {args['output_dir'].resolve()}")


if __name__ == "__main__":
    main()
