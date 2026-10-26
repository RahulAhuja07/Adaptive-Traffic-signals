"""CLI for paired, non-learning SUMO controller evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .comparison_runner import DEFAULT_OUTPUT, run_comparison
except ImportError:
    from comparison_runner import DEFAULT_OUTPUT, run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--dqn-checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--state-only-checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1030)))
    parser.add_argument("--horizon", type=float, default=1800.0)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-smoke-checkpoints", action="store_true",
                        help="allow <100-episode/unverified checkpoints; outputs are labeled exploratory")
    args = parser.parse_args()
    options = {}
    if args.q_checkpoints is not None:
        options["q_checkpoints"] = args.q_checkpoints
    if args.dqn_checkpoints is not None:
        options["dqn_checkpoints"] = args.dqn_checkpoints
    if args.state_only_checkpoints is not None:
        options["state_only_checkpoints"] = args.state_only_checkpoints
    rows, summaries = run_comparison(
        seeds=args.seeds, horizon=args.horizon, interval=args.interval,
        output_dir=args.output_dir,
        allow_smoke_checkpoints=args.allow_smoke_checkpoints, **options,
    )
    print(f"Wrote {len(rows)} episodes and {len(summaries)} summary groups to {args.output_dir}")


if __name__ == "__main__":
    main()
