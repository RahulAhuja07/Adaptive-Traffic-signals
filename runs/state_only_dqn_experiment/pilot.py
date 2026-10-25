"""Small state-only DQN pilot: train and evaluate a richer directional DQN."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.state_only_dqn_experiment.dqn_agent import StateOnlyDQNAgent
from runs.state_only_dqn_experiment.environment import RichStateTrafficEnvironment


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train_state_only_dqn(
    scenario: str,
    rates: Tuple[float, float],
    episodes: int,
    seed: int,
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
    output_dir: Path = PROJECT_DIR,
) -> List[Dict[str, object]]:
    agent = StateOnlyDQNAgent(seed=seed + 202)
    env = RichStateTrafficEnvironment(
        arrival_rates=rates,
        episode_steps=episode_steps,
        interval_seconds=interval_seconds,
        service_capacity=service_capacity,
        reward_mode="baseline",
        queue_scale=20.0,
    )
    rows: List[Dict[str, object]] = []
    for episode in range(1, episodes + 1):
        state = env.reset(seed=seed + episode)
        done = False
        losses: List[float] = []
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            loss = agent.update(state, action, reward, next_state, done)
            if loss is not None and math.isfinite(loss):
                losses.append(loss)
            state = next_state
        metrics = env.get_metrics()
        agent.decay_epsilon()
        rows.append(
            {
                "episode": episode,
                "scenario": scenario,
                **metrics,
                "epsilon": agent.epsilon,
                "mean_training_loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )
    write_csv(
        output_dir / "results" / f"state_only_dqn_{scenario}_seed{seed}_training.csv",
        rows,
        [
            "episode", "scenario", "cumulative_reward", "avg_waiting_time_sec",
            "time_weighted_avg_queue", "throughput_veh_per_min", "total_arrived",
            "total_departed", "final_queue", "epsilon", "mean_training_loss",
        ],
    )
    agent.save(str(output_dir / "models" / f"state_only_dqn_{scenario}_seed{seed}.pt"))
    return rows


def evaluate_state_only_dqn(
    scenario: str,
    rates: Tuple[float, float],
    episodes: int,
    seed: int,
    model_seed: int,
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
    output_dir: Path = PROJECT_DIR,
) -> List[Dict[str, object]]:
    agent = StateOnlyDQNAgent(seed=model_seed + 202)
    model_path = output_dir / "models" / f"state_only_dqn_{scenario}_seed{model_seed}.pt"
    agent.load(str(model_path))
    env = RichStateTrafficEnvironment(
        arrival_rates=rates,
        episode_steps=episode_steps,
        interval_seconds=interval_seconds,
        service_capacity=service_capacity,
        reward_mode="baseline",
        queue_scale=20.0,
    )
    rows: List[Dict[str, object]] = []
    for episode in range(1, episodes + 1):
        episode_seed = seed + episode - 1
        state = env.reset(seed=episode_seed)
        done = False
        green_intervals_ns = 0
        green_intervals_ew = 0
        departures_ns = 0
        departures_ew = 0
        while not done:
            action = agent.choose_action(state, training=False)
            if action not in (0, 1):
                raise RuntimeError(f"invalid action {action!r}")
            green_intervals_ns += int(action == 0)
            green_intervals_ew += int(action == 1)
            state, _, done, info = env.step(action)
            departures_ns += int(info["departures_ns"])
            departures_ew += int(info["departures_ew"])
        metrics = env.get_metrics()
        rows.append(
            {
                "scenario": scenario,
                "controller": "State-Only DQN",
                "episode": episode,
                "seed": episode_seed,
                "green_intervals_ns": green_intervals_ns,
                "green_intervals_ew": green_intervals_ew,
                "departures_ns": departures_ns,
                "departures_ew": departures_ew,
                **metrics,
            }
        )
    return rows


def main() -> None:
    ensure_dirs()
    scenario = "high"
    seed = 42
    rates = (4.0, 3.5)
    train_rows = train_state_only_dqn(
        scenario=scenario,
        rates=rates,
        episodes=10,
        seed=seed,
        episode_steps=180,
        interval_seconds=10.0,
        service_capacity=5,
        output_dir=PROJECT_DIR,
    )
    eval_rows = evaluate_state_only_dqn(
        scenario=scenario,
        rates=rates,
        episodes=5,
        seed=seed,
        model_seed=seed,
        episode_steps=180,
        interval_seconds=10.0,
        service_capacity=5,
        output_dir=PROJECT_DIR,
    )
    write_csv(
        PROJECT_DIR / "results" / f"state_only_dqn_{scenario}_seed{seed}_evaluation.csv",
        eval_rows,
        [
            "scenario", "controller", "episode", "seed", "green_intervals_ns",
            "green_intervals_ew", "departures_ns", "departures_ew",
            "avg_waiting_time_sec", "time_weighted_avg_queue", "throughput_veh_per_min",
            "total_arrived", "total_departed", "final_queue", "cumulative_reward",
        ],
    )
    print(f"Training rows: {len(train_rows)}")
    print(f"Evaluation rows: {len(eval_rows)}")
    print("State-only DQN pilot complete.")


if __name__ == "__main__":
    main()
