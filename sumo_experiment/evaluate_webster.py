"""Evaluate the existing Webster controller on SUMO's A1 intersection."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .webster_runner import DEFAULT_OUTPUT_DIR, evaluate_webster
else:
    from webster_runner import DEFAULT_OUTPUT_DIR, evaluate_webster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="evaluation episodes (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-horizon-seconds", type=float, default=360.0)
    parser.add_argument("--decision-interval-seconds", type=float, default=10.0)
    parser.add_argument("--arrival-rate-ns", type=float, default=None,
                        help="optional nominal NS arrivals per decision interval; default derives from route period")
    parser.add_argument("--arrival-rate-ew", type=float, default=None,
                        help="optional nominal EW arrivals per decision interval; default derives from route period")
    parser.add_argument("--saturation-flow-ns", type=float, default=1800.0,
                        help="assumed NS saturation flow in vehicles/hour")
    parser.add_argument("--saturation-flow-ew", type=float, default=1800.0,
                        help="assumed EW saturation flow in vehicles/hour")
    parser.add_argument("--lost-time-seconds", type=float, default=None,
                        help="total lost time per cycle; default is derived from A1 yellow durations")
    parser.add_argument("--min-cycle-seconds", type=float, default=30.0)
    parser.add_argument("--max-cycle-seconds", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = vars(build_parser().parse_args())
    rate_ns, rate_ew = args.pop("arrival_rate_ns"), args.pop("arrival_rate_ew")
    if (rate_ns is None) != (rate_ew is None):
        raise SystemExit("Specify both --arrival-rate-ns and --arrival-rate-ew, or neither")
    args["arrival_rates"] = (rate_ns, rate_ew) if rate_ns is not None else None
    sat_ns, sat_ew = args.pop("saturation_flow_ns"), args.pop("saturation_flow_ew")
    args["saturation_flows"] = (sat_ns, sat_ew)
    try:
        rows, _ = evaluate_webster(**args)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"SUMO Webster evaluation failed: {error}") from error
    print(f"Evaluated {len(rows)} episodes; outputs saved in {args['output_dir'].resolve()}")


if __name__ == "__main__":
    main()
