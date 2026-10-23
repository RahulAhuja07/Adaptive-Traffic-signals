"""Shared configuration, filesystem, CSV, and summary helpers for experiments."""

import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from environment import TrafficEnvironment


PROJECT_DIR = Path(__file__).resolve().parent
SCENARIOS: Dict[str, Tuple[float, float]] = {
    "low": (1.0, 0.8),
    "medium": (2.8, 1.2),
    "high": (4.0, 3.5),
}
SCENARIO_LABELS = {"low": "Low demand", "medium": "Medium demand", "high": "High demand"}
CONTROLLERS = ("Q-Learning", "DQN", "Webster")
EVALUATION_FIELDS = (
    "scenario", "controller", "episode", "seed", "avg_waiting_time_sec",
    "time_weighted_avg_queue", "throughput_veh_per_min", "total_arrived",
    "total_departed", "final_queue", "cumulative_reward",
)


def ensure_output_dirs(root: Path = PROJECT_DIR) -> Dict[str, Path]:
    """Create and return the standard results, models, and plots directories."""
    directories = {name: root / name for name in ("results", "models", "plots")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable controller experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_scenarios(selection: str) -> List[Tuple[str, Tuple[float, float]]]:
    """Return configured demand scenarios in a stable order."""
    if selection == "all":
        return list(SCENARIOS.items())
    if selection not in SCENARIOS:
        raise ValueError(f"unknown scenario {selection!r}; choose low, medium, high, or all")
    return [(selection, SCENARIOS[selection])]


def make_environment(
    arrival_rates: Sequence[float],
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
    seed: Optional[int] = None,
) -> TrafficEnvironment:
    """Build the same environment configuration for every controller."""
    return TrafficEnvironment(
        arrival_rates=tuple(arrival_rates),
        episode_steps=episode_steps,
        interval_seconds=interval_seconds,
        service_capacity=service_capacity,
        seed=seed,
    )


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Write dictionaries to CSV, creating parent directories as needed."""
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = list(materialized[0].keys()) if materialized else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV rows as dictionaries of strings."""
    with path.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _valid_numbers(rows: Sequence[Mapping[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Return population mean/std, or NaN for both when no data is valid."""
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=0))


def aggregate_evaluation(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Aggregate per-episode results by scenario and controller.

    Undefined values are ignored. Standard deviations use the population
    definition (ddof=0); the waiting-time valid count is reported explicitly.
    """
    groups: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["scenario"]), str(row["controller"])), []).append(row)

    output: List[Dict[str, object]] = []
    for (scenario, controller), group in sorted(groups.items()):
        wait = _valid_numbers(group, "avg_waiting_time_sec")
        queue = _valid_numbers(group, "time_weighted_avg_queue")
        throughput = _valid_numbers(group, "throughput_veh_per_min")
        departures = _valid_numbers(group, "total_departed")
        final_queue = _valid_numbers(group, "final_queue")
        reward = _valid_numbers(group, "cumulative_reward")
        mean_wait, std_wait = _mean_std(wait)
        mean_queue, std_queue = _mean_std(queue)
        mean_throughput, std_throughput = _mean_std(throughput)
        mean_departures, _ = _mean_std(departures)
        mean_final_queue, std_final_queue = _mean_std(final_queue)
        mean_reward, _ = _mean_std(reward)
        output.append({
            "scenario": scenario,
            "controller": controller,
            "mean_avg_waiting_time_sec": mean_wait,
            "std_avg_waiting_time_sec": std_wait,
            "valid_avg_waiting_time_observations": len(wait),
            "mean_time_weighted_avg_queue": mean_queue,
            "std_time_weighted_avg_queue": std_queue,
            "mean_throughput_veh_per_min": mean_throughput,
            "std_throughput_veh_per_min": std_throughput,
            "mean_total_departed": mean_departures,
            "mean_final_queue": mean_final_queue,
            "std_final_queue": std_final_queue,
            "mean_cumulative_reward": mean_reward,
        })
    return output


def training_log_path(root: Path, algorithm: str, scenario: str, seed: int) -> Path:
    """Build a noncolliding per-run training-log path."""
    return root / "results" / f"{algorithm}_{scenario}_seed{seed}_training.csv"


def model_path(root: Path, algorithm: str, scenario: str, seed: int) -> Path:
    """Build the model path corresponding to an algorithm/scenario/seed run."""
    extension = ".npz" if algorithm == "q_learning" else ".pt"
    return root / "models" / f"{algorithm}_{scenario}_seed{seed}{extension}"
