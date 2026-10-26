"""Evaluate the existing Webster controller on the shared SUMO environment."""

from __future__ import annotations

import csv
import json
import math
from numbers import Integral
import os
from pathlib import Path
import sys
from typing import Any, Callable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SUMO_EXPERIMENT_DIR / "runs" / "webster"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from .sumo_environment import ROUTES_FILE, SUSTAINED_ROUTES_ENV, SumoTrafficEnvironment
    from .webster_adapter import SumoWebsterAdapter
else:
    from sumo_environment import ROUTES_FILE, SUSTAINED_ROUTES_ENV, SumoTrafficEnvironment
    from webster_adapter import SumoWebsterAdapter


METRIC_FIELDS = (
    "episode", "sumo_seed", "cumulative_reward_reference",
    "total_stopped_waiting_time_sec_completed",
    "avg_stopped_waiting_time_sec_completed", "time_weighted_avg_queue_vehicles",
    "throughput_completed_veh_per_min", "completed_vehicles", "final_queue_vehicles",
    "episode_duration_sec", "decision_steps", "termination_reason",
    "green_intervals_ns", "green_intervals_ew", "yellow_clearance_seconds_total",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def _validate_metrics(env: Any) -> dict[str, float]:
    metrics = env.get_metrics()
    keys = (
        "total_stopped_waiting_time_sec_completed", "avg_stopped_waiting_time_sec",
        "time_weighted_avg_queue", "throughput_veh_per_min", "completed_vehicles",
        "final_queue", "simulated_time_sec", "cumulative_reward",
    )
    values = {key: float(metrics[key]) for key in keys}
    for key in keys:
        if key == "avg_stopped_waiting_time_sec" and math.isnan(values[key]):
            continue
        if not math.isfinite(values[key]):
            raise ValueError(f"SUMO metric {key} must be finite or NaN when no completed trip exists")
        if key != "cumulative_reward" and key != "avg_stopped_waiting_time_sec" and values[key] < 0:
            raise ValueError(f"SUMO metric {key} must be nonnegative")
    return values


def evaluate_webster(
    episodes: int = 1,
    seed: int = 42,
    episode_horizon_seconds: float = 360.0,
    decision_interval_seconds: float = 10.0,
    arrival_rates: Optional[Sequence[float]] = None,
    saturation_flows: Sequence[float] = (1800.0, 1800.0),
    lost_time_seconds: Optional[float] = None,
    min_cycle_seconds: float = 30.0,
    max_cycle_seconds: float = 120.0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    env_factory: Callable[..., Any] = SumoTrafficEnvironment,
    controller_factory: Callable[..., SumoWebsterAdapter] = SumoWebsterAdapter,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run paired-seed fixed-time Webster evaluation and write isolated outputs."""
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    first_adapter: Optional[SumoWebsterAdapter] = None

    for episode in range(1, int(episodes) + 1):
        sumo_seed = int(seed) + episode - 1
        env = env_factory(
            episode_horizon_seconds=episode_horizon_seconds,
            decision_interval_seconds=decision_interval_seconds,
        )
        try:
            env.reset(seed=sumo_seed)
            controller = controller_factory(
                env,
                arrival_rates=arrival_rates,
                saturation_flows=saturation_flows,
                lost_time_seconds=lost_time_seconds,
                min_cycle_seconds=min_cycle_seconds,
                max_cycle_seconds=max_cycle_seconds,
            )
            if first_adapter is None:
                first_adapter = controller
            controller.reset()
            done = False
            reward_sum = 0.0
            decision_steps = 0
            green_ns = 0
            green_ew = 0
            yellow_time = 0.0
            termination_reason = ""
            while not done:
                action = controller.choose_action()
                if action not in (0, 1):
                    raise ValueError(f"Webster action must be 0 (NS) or 1 (EW), got {action!r}")
                green_ns += int(action == 0)
                green_ew += int(action == 1)
                _, reward, done, info = env.step(action)
                reward = float(reward)
                if not math.isfinite(reward):
                    raise ValueError(f"SUMO reward must be finite, got {reward!r}")
                reward_sum += reward
                decision_steps += 1
                yellow_time += float(info.get("yellow_clearance_seconds", 0.0))
                if done:
                    termination_reason = str(info.get("termination_reason", "unspecified"))

            metrics = _validate_metrics(env)
            rows.append({
                "episode": episode,
                "sumo_seed": sumo_seed,
                "cumulative_reward_reference": reward_sum,
                "total_stopped_waiting_time_sec_completed": metrics["total_stopped_waiting_time_sec_completed"],
                "avg_stopped_waiting_time_sec_completed": metrics["avg_stopped_waiting_time_sec"],
                "time_weighted_avg_queue_vehicles": metrics["time_weighted_avg_queue"],
                "throughput_completed_veh_per_min": metrics["throughput_veh_per_min"],
                "completed_vehicles": int(metrics["completed_vehicles"]),
                "final_queue_vehicles": metrics["final_queue"],
                "episode_duration_sec": metrics["simulated_time_sec"],
                "decision_steps": decision_steps,
                "termination_reason": termination_reason,
                "green_intervals_ns": green_ns,
                "green_intervals_ew": green_ew,
                "yellow_clearance_seconds_total": yellow_time,
            })
        finally:
            env.close()

    if first_adapter is None:
        raise RuntimeError("no Webster episode was evaluated")
    timing_plan = first_adapter.timing_plan
    ns_intervals = int(timing_plan["green_ns_intervals"])
    ew_intervals = int(timing_plan["green_ew_intervals"])
    actual_slot_cycle = (ns_intervals + ew_intervals) * float(decision_interval_seconds)
    config = {
        "experiment": "SUMO Webster fixed-time evaluation",
        "episodes": int(episodes),
        "seed": int(seed),
        "sumo_seeds": [int(seed) + index for index in range(int(episodes))],
        "episode_horizon_seconds": float(episode_horizon_seconds),
        "decision_interval_seconds": float(decision_interval_seconds),
        "routes_file": str(Path(os.environ.get(SUSTAINED_ROUTES_ENV, str(ROUTES_FILE))).expanduser().resolve()),
        "arrival_rates_expected_per_interval_ns_ew": list(first_adapter.arrival_rates),
        "arrival_rate_source": first_adapter.arrival_rate_source,
        "saturation_flows_veh_per_hour_ns_ew": list(first_adapter.saturation_flows),
        "saturation_flow_assumption": "configurable original-controller default; not measured or calibrated for this SUMO network",
        "lost_time_seconds_per_cycle": first_adapter.lost_time_seconds,
        "lost_time_source": first_adapter.lost_time_source,
        "signal_yellow_seconds_ns_ew": list(first_adapter.signal_yellow_seconds),
        "min_cycle_seconds": float(min_cycle_seconds),
        "max_cycle_seconds": float(max_cycle_seconds),
        "timing_plan": timing_plan,
        "sumo_actuation_plan": {
            "ns_action_intervals": ns_intervals,
            "ew_action_intervals": ew_intervals,
            "steady_state_effective_green_ns_seconds": max(
                0.0, timing_plan["green_ns_seconds"] - first_adapter.signal_yellow_seconds[1]
            ),
            "steady_state_effective_green_ew_seconds": max(
                0.0, timing_plan["green_ew_seconds"] - first_adapter.signal_yellow_seconds[0]
            ),
            "cycle_wall_clock_seconds": actual_slot_cycle,
            "interpretation": "Each Webster action owns one decision interval. The yellow clearance is contained in the first interval after a switch, so steady-state effective green is that phase's rounded interval allocation minus the incoming yellow; cycle wall time is the sum of action intervals.",
        },
        "actual_actuation": "controller emits its existing discrete NS/EW interval choices; SumoTrafficEnvironment inserts A1 yellow clearance inside the first 10-second interval after a switch",
        "reward_metric": "cumulative reward is recorded for reference; comparisons should emphasize SUMO queue, stopped waiting, throughput, and duration metrics",
        "queue_metric": "time-weighted A1 inbound halting vehicles (speed <= 0.1 m/s)",
        "waiting_metric": "sampled stopped seconds for completed trips only; unfinished trips excluded",
        "throughput_metric": "completed route trips per simulated minute",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "evaluation_metrics.csv", rows)
    _write_json(output_dir / "webster_config.json", config)
    return rows, config
