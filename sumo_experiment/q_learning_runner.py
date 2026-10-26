"""Training and greedy evaluation helpers for the isolated SUMO Q-learning run.

The runner imports the original QLearningAgent unchanged. Metrics are labeled
with SUMO-specific definitions: waiting totals include sampled stopped seconds
for completed route trips only; average queue is the time-weighted mean of A1
inbound halting counts; throughput counts completed route trips per minute.
"""

from __future__ import annotations

import csv
import json
import math
from numbers import Integral
import os
from pathlib import Path
import sys
from typing import Any, Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SUMO_EXPERIMENT_DIR / "runs" / "q_learning"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from q_learning import QLearningAgent  # noqa: E402
if __package__:
    from .sumo_environment import ROUTES_FILE, SUSTAINED_ROUTES_ENV, SumoTrafficEnvironment  # noqa: E402
else:
    from sumo_environment import ROUTES_FILE, SUSTAINED_ROUTES_ENV, SumoTrafficEnvironment  # noqa: E402


TRAIN_FIELDS = (
    "episode", "sumo_seed", "cumulative_reward",
    "total_stopped_waiting_time_sec_completed",
    "avg_stopped_waiting_time_sec_completed", "time_weighted_avg_queue_vehicles",
    "throughput_completed_veh_per_min", "completed_vehicles",
    "episode_duration_sec", "decision_steps", "termination_reason", "epsilon",
)
EVALUATION_FIELDS = TRAIN_FIELDS[:-1] + ("exploration_enabled",)


def validate_state(state: Any, context: str = "state") -> int:
    """Require the existing tabular agent's exact integer state contract."""
    if isinstance(state, bool) or not isinstance(state, Integral) or not 0 <= int(state) <= 17:
        raise ValueError(f"SUMO {context} must be an integer in [0, 17], got {state!r}")
    return int(state)


def validate_action(action: Any, context: str = "action") -> int:
    """Require the shared action meanings: 0=NS, 1=EW."""
    if isinstance(action, bool) or not isinstance(action, Integral) or int(action) not in (0, 1):
        raise ValueError(f"SUMO {context} must be 0 (NS) or 1 (EW), got {action!r}")
    return int(action)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def _make_env(env_factory: Callable[..., Any], episode_horizon_seconds: float,
              decision_interval_seconds: float) -> Any:
    return env_factory(
        episode_horizon_seconds=episode_horizon_seconds,
        decision_interval_seconds=decision_interval_seconds,
    )


def _episode_metrics(env: Any, episode: int, sumo_seed: int, reward_sum: float,
                     decision_steps: int, termination_reason: str,
                     epsilon: Optional[float]) -> dict[str, Any]:
    metrics = env.get_metrics()
    total_stopped_wait = float(metrics["total_stopped_waiting_time_sec_completed"])
    average_stopped_wait = float(metrics["avg_stopped_waiting_time_sec"])
    average_queue = float(metrics["time_weighted_avg_queue"])
    throughput = float(metrics["throughput_veh_per_min"])
    completed = float(metrics["completed_vehicles"])
    duration = float(metrics["simulated_time_sec"])
    for key, value in (
        ("total stopped waiting time", total_stopped_wait),
        ("average queue", average_queue),
        ("throughput", throughput),
        ("completed vehicles", completed),
        ("episode duration", duration),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"SUMO metric {key} must be finite and nonnegative, got {value!r}")
    if not math.isfinite(average_stopped_wait) and not math.isnan(average_stopped_wait):
        raise ValueError("SUMO average stopped waiting time must be finite or NaN when unavailable")
    row = {
        "episode": episode,
        "sumo_seed": sumo_seed,
        "cumulative_reward": reward_sum,
        "total_stopped_waiting_time_sec_completed": total_stopped_wait,
        "avg_stopped_waiting_time_sec_completed": average_stopped_wait,
        "time_weighted_avg_queue_vehicles": average_queue,
        "throughput_completed_veh_per_min": throughput,
        "completed_vehicles": int(completed),
        "episode_duration_sec": duration,
        "decision_steps": decision_steps,
        "termination_reason": termination_reason,
    }
    if epsilon is not None:
        row["epsilon"] = epsilon
    else:
        row["exploration_enabled"] = False
    return row


def _configuration(episodes: int, seed: int, episode_horizon_seconds: float,
                   decision_interval_seconds: float, algorithm_seed: int) -> dict[str, Any]:
    return {
        "experiment": "SUMO tabular Q-learning",
        "state_size": 18,
        "action_size": 2,
        "action_meanings": {"0": "NS green", "1": "EW green"},
        "state_encoding": "phase * 9 + ns_bin * 3 + ew_bin; bins are 0-4, 5-9, and 10+ halting vehicles",
        "queue_observation": "SUMO halting-vehicle counts (speed <= 0.1 m/s) summed over A1 inbound lanes; not the custom simulator's FIFO queue",
        "reward": "-((observed_queue_before + observed_queue_after) / 2) / 10 per decision interval",
        "waiting_metric": "sum of sampled stopped seconds (speed <= 0.1 m/s) for completed route trips; unfinished trips excluded",
        "average_queue_metric": "time-weighted average of NS+EW A1 inbound halting vehicles",
        "throughput_metric": "completed route trips divided by simulated minutes",
        "termination": "configured episode horizon or SUMO reports no expected/active vehicles",
        "episodes": episodes,
        "seed": seed,
        "sumo_seeds": [seed + index for index in range(episodes)],
        "algorithm_seed": algorithm_seed,
        "episode_horizon_seconds": episode_horizon_seconds,
        "decision_interval_seconds": decision_interval_seconds,
        "routes_file": str(Path(os.environ.get(SUSTAINED_ROUTES_ENV, str(ROUTES_FILE))).expanduser().resolve()),
        "agent_hyperparameters": {
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon_initial": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
        },
        "reproducibility": "SUMO receives a per-episode seed and the agent tie-breaker/exploration RNG is seeded. Reproducibility can still vary across SUMO versions, platforms, or builds.",
    }


def train_q_learning(
    episodes: int = 3,
    seed: int = 42,
    episode_horizon_seconds: float = 60.0,
    decision_interval_seconds: float = 10.0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    env_factory: Callable[..., Any] = SumoTrafficEnvironment,
) -> tuple[QLearningAgent, list[dict[str, Any]]]:
    """Train the unchanged Q-learning agent; default run is deliberately small."""
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    output_dir = Path(output_dir)
    algorithm_seed = int(seed) + 101
    agent = QLearningAgent(seed=algorithm_seed)
    rows: list[dict[str, Any]] = []

    for episode in range(1, int(episodes) + 1):
        sumo_seed = int(seed) + episode - 1
        env = _make_env(env_factory, episode_horizon_seconds, decision_interval_seconds)
        try:
            state = validate_state(env.reset(seed=sumo_seed), "reset state")
            done = False
            reward_sum = 0.0
            decision_steps = 0
            termination_reason = ""
            while not done:
                action = validate_action(agent.choose_action(state, training=True), "agent action")
                next_state_raw, reward_raw, done, info = env.step(action)
                next_state = validate_state(next_state_raw, "next state")
                reward = float(reward_raw)
                if not math.isfinite(reward):
                    raise ValueError(f"SUMO reward must be finite, got {reward_raw!r}")
                agent.update(state, action, reward, next_state, bool(done))
                state = next_state
                reward_sum += reward
                decision_steps += 1
                if done:
                    termination_reason = str(info.get("termination_reason", "unspecified"))
            agent.decay_epsilon()
            rows.append(_episode_metrics(
                env, episode, sumo_seed, reward_sum, decision_steps,
                termination_reason, agent.epsilon,
            ))
        finally:
            env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "q_table.npz"
    agent.save(str(checkpoint_path))
    _write_csv(output_dir / "training_metrics.csv", rows, TRAIN_FIELDS)
    config = _configuration(
        int(episodes), int(seed), float(episode_horizon_seconds),
        float(decision_interval_seconds), algorithm_seed,
    )
    config.update({
        "checkpoint": checkpoint_path.name,
        "training_metrics": "training_metrics.csv",
        "exploration_enabled": True,
        "routes_file": str(Path(os.environ.get(SUSTAINED_ROUTES_ENV, str(ROUTES_FILE))).expanduser().resolve()),
    })
    _write_json(output_dir / "training_config.json", config)
    return agent, rows


def evaluate_q_learning(
    episodes: int = 1,
    seed: int = 142,
    model_path: Path | str = DEFAULT_OUTPUT_DIR / "q_table.npz",
    episode_horizon_seconds: float = 60.0,
    decision_interval_seconds: float = 10.0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    env_factory: Callable[..., Any] = SumoTrafficEnvironment,
) -> list[dict[str, Any]]:
    """Evaluate a saved table with exploration disabled and write episode CSV."""
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    model_path, output_dir = Path(model_path), Path(output_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"SUMO Q-table not found: {model_path}; run train_q_learning.py first")

    agent = QLearningAgent(seed=int(seed) + 101)
    agent.load(str(model_path))
    rows: list[dict[str, Any]] = []
    for episode in range(1, int(episodes) + 1):
        sumo_seed = int(seed) + episode - 1
        env = _make_env(env_factory, episode_horizon_seconds, decision_interval_seconds)
        try:
            state = validate_state(env.reset(seed=sumo_seed), "reset state")
            done = False
            reward_sum = 0.0
            decision_steps = 0
            termination_reason = ""
            while not done:
                action = validate_action(agent.choose_action(state, training=False), "agent action")
                next_state_raw, reward_raw, done, info = env.step(action)
                next_state = validate_state(next_state_raw, "next state")
                reward = float(reward_raw)
                if not math.isfinite(reward):
                    raise ValueError(f"SUMO reward must be finite, got {reward_raw!r}")
                # Evaluation is read-only: no Q updates or epsilon decay.
                state = next_state
                reward_sum += reward
                decision_steps += 1
                if done:
                    termination_reason = str(info.get("termination_reason", "unspecified"))
            rows.append(_episode_metrics(
                env, episode, sumo_seed, reward_sum, decision_steps,
                termination_reason, None,
            ))
        finally:
            env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "evaluation_metrics.csv", rows, EVALUATION_FIELDS)
    config = _configuration(
        int(episodes), int(seed), float(episode_horizon_seconds),
        float(decision_interval_seconds), int(seed) + 101,
    )
    config.update({
        "checkpoint": str(model_path.resolve()),
        "evaluation_metrics": "evaluation_metrics.csv",
        "exploration_enabled": False,
        "routes_file": str(Path(os.environ.get(SUSTAINED_ROUTES_ENV, str(ROUTES_FILE))).expanduser().resolve()),
    })
    _write_json(output_dir / "evaluation_config.json", config)
    return rows
