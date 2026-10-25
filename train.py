"""Train Q-learning and DQN agents on configured traffic-demand scenarios."""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dqn_agent import DQNAgent
from q_learning import QLearningAgent
from utils import (
    PROJECT_DIR,
    ensure_output_dirs,
    make_environment,
    model_path,
    selected_scenarios,
    set_global_seed,
    training_log_path,
    write_csv,
)


TRAIN_FIELDS = (
    "episode", "scenario", "cumulative_reward", "avg_waiting_time_sec",
    "time_weighted_avg_queue", "throughput_veh_per_min", "total_arrived",
    "total_departed", "final_queue", "epsilon", "mean_training_loss",
)


def train_q_learning(
    scenario: str,
    rates: tuple,
    episodes: int,
    seed: int,
    episode_steps: int,
    interval_seconds: float,
    service_capacity: int,
    output_root: Path = PROJECT_DIR,
    reward_mode: str = "baseline",
) -> List[Dict[str, object]]:
    """Train one tabular agent and save its per-episode log and Q-table."""
    agent = QLearningAgent(seed=seed + 101)
    rows: List[Dict[str, object]] = []
    env = make_environment(rates, episode_steps, interval_seconds, service_capacity, reward_mode=reward_mode)
    for episode in range(1, episodes + 1):
        state = env.reset(seed=seed + episode)
        done = False
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        metrics = env.get_metrics()
        agent.decay_epsilon()
        rows.append({
            "episode": episode,
            "scenario": scenario,
            **metrics,
            "epsilon": agent.epsilon,
            "mean_training_loss": float("nan"),
        })
    write_csv(training_log_path(output_root, "q_learning", scenario, seed), rows, TRAIN_FIELDS)
    agent.save(str(model_path(output_root, "q_learning", scenario, seed)))
    return rows


def train_dqn(
    scenario: str,
    rates: tuple,
    episodes: int,
    seed: int,
    episode_steps: int,
    interval_seconds: float,
    service_capacity: int,
    output_root: Path = PROJECT_DIR,
    reward_mode: str = "baseline",
) -> List[Dict[str, object]]:
    """Train one DQN agent and save its per-episode log and checkpoint."""
    agent = DQNAgent(seed=seed + 202)
    rows: List[Dict[str, object]] = []
    env = make_environment(rates, episode_steps, interval_seconds, service_capacity, reward_mode=reward_mode)
    for episode in range(1, episodes + 1):
        state = env.reset(seed=seed + episode)
        done = False
        losses: List[float] = []
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            loss = agent.update(state, action, reward, next_state, done)
            if loss is not None and math.isfinite(loss):
                losses.append(loss)
            state = next_state
        metrics = env.get_metrics()
        agent.decay_epsilon()
        rows.append({
            "episode": episode,
            "scenario": scenario,
            **metrics,
            "epsilon": agent.epsilon,
            "mean_training_loss": float(np.mean(losses)) if losses else float("nan"),
        })
    write_csv(training_log_path(output_root, "dqn", scenario, seed), rows, TRAIN_FIELDS)
    agent.save(str(model_path(output_root, "dqn", scenario, seed)))
    return rows


def run_training(
    episodes: int = 5000,
    seed: int = 42,
    scenario: str = "all",
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
    output_root: Path = PROJECT_DIR,
    reward_mode: str = "baseline",
) -> None:
    """Train both algorithms for each selected demand scenario."""
    if episodes <= 0 or episode_steps <= 0:
        raise ValueError("episodes and episode_steps must be positive")
    ensure_output_dirs(output_root)
    set_global_seed(seed)
    for scenario_name, rates in selected_scenarios(scenario):
        print(f"Training {scenario_name} demand: NS={rates[0]}, EW={rates[1]}")
        train_q_learning(
            scenario_name, rates, episodes, seed, episode_steps,
            interval_seconds, service_capacity, output_root, reward_mode,
        )
        train_dqn(
            scenario_name, rates, episodes, seed, episode_steps,
            interval_seconds, service_capacity, output_root, reward_mode,
        )
    print(f"Training logs and models saved under {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=("low", "medium", "high", "all"), default="all")
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--service-capacity", type=int, default=5)
    parser.add_argument("--output-dir", dest="output_root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--reward-mode", choices=("baseline", "directional"), default="baseline")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_training(**vars(args))
    except (ValueError, OSError) as error:
        raise SystemExit(f"Training failed: {error}") from error


if __name__ == "__main__":
    main()
