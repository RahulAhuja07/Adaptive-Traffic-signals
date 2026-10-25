"""Extended state-only DQN experiment in an isolated directory.

This script keeps baseline and directional-pilot files untouched. It trains the
new richer-state DQN independently and compares it against the original root DQN,
Q-learning, and Webster under identical evaluation seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_agent import DQNAgent
from q_learning import QLearningAgent
from webster_controller import WebsterController
from environment import TrafficEnvironment
from runs.state_only_dqn_experiment.dqn_agent import StateOnlyDQNAgent
from runs.state_only_dqn_experiment.environment import RichStateTrafficEnvironment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"

SCENARIOS: Dict[str, Tuple[float, float]] = {
    "low": (1.0, 0.8),
    "medium": (2.8, 1.2),
    "high": (4.0, 3.5),
}

TRAIN_FIELDS = [
    "episode", "scenario", "seed", "cumulative_reward", "avg_waiting_time_sec",
    "time_weighted_avg_queue", "throughput_veh_per_min", "total_arrived",
    "total_departed", "final_queue", "epsilon", "mean_training_loss",
]
EVAL_FIELDS = [
    "scenario", "controller", "episode", "seed", "green_intervals_ns",
    "green_intervals_ew", "departures_ns", "departures_ew",
    "avg_waiting_time_sec", "time_weighted_avg_queue", "throughput_veh_per_min",
    "total_arrived", "total_departed", "final_queue", "cumulative_reward",
]


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def feature_summary(states: Sequence[np.ndarray]) -> Dict[str, float]:
    if not states:
        return {"phase_ns_min": 0.0, "phase_ns_max": 0.0, "phase_ew_min": 0.0, "phase_ew_max": 0.0,
                "ns_queue_norm_min": 0.0, "ns_queue_norm_max": 0.0, "ew_queue_norm_min": 0.0,
                "ew_queue_norm_max": 0.0, "total_queue_norm_min": 0.0, "total_queue_norm_max": 0.0,
                "imbalance_norm_min": 0.0, "imbalance_norm_max": 0.0}
    arr = np.asarray(states, dtype=np.float32)
    labels = [
        "phase_ns", "phase_ew", "ns_queue_norm", "ew_queue_norm",
        "total_queue_norm", "imbalance_norm",
    ]
    summary: Dict[str, float] = {}
    for idx, name in enumerate(labels):
        values = arr[:, idx]
        summary[f"{name}_min"] = float(values.min())
        summary[f"{name}_p01"] = float(np.percentile(values, 1))
        summary[f"{name}_p50"] = float(np.percentile(values, 50))
        summary[f"{name}_p99"] = float(np.percentile(values, 99))
        summary[f"{name}_max"] = float(values.max())
        summary[f"{name}_mean"] = float(values.mean())
    return summary


def train_state_only_dqn(
    scenario: str,
    rates: Tuple[float, float],
    episodes: int,
    seed: int,
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
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
    feature_samples: List[np.ndarray] = []
    for episode in range(1, episodes + 1):
        state = env.reset(seed=seed + episode)
        done = False
        losses: List[float] = []
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            feature_samples.append(state)
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
                "seed": seed,
                **metrics,
                "epsilon": agent.epsilon,
                "mean_training_loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )
    path = MODELS_DIR / f"state_only_dqn_{scenario}_seed{seed}.pt"
    agent.save(str(path))
    write_csv(
        RESULTS_DIR / f"state_only_dqn_{scenario}_seed{seed}_training.csv",
        rows,
        TRAIN_FIELDS + ["seed"],
    )
    summary = feature_summary(feature_samples)
    summary_path = RESULTS_DIR / f"state_only_dqn_{scenario}_seed{seed}_feature_summary.json"
    summary_path.write_text(json.dumps({"scenario": scenario, "seed": seed, "queue_scale": 20.0, **summary}, indent=2), encoding="utf-8")
    return rows


def evaluate_controller(
    controller_name: str,
    controller,
    scenario: str,
    rates: Tuple[float, float],
    episodes: int,
    seed: int,
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
    reward_mode: str = "baseline",
) -> List[Dict[str, object]]:
    env = TrafficEnvironment(
        arrival_rates=rates,
        episode_steps=episode_steps,
        interval_seconds=interval_seconds,
        service_capacity=service_capacity,
        seed=seed,
        reward_mode=reward_mode,
    )
    rows: List[Dict[str, object]] = []
    for episode in range(1, episodes + 1):
        episode_seed = seed + episode - 1
        state = env.reset(seed=episode_seed)
        if controller_name == "Webster":
            controller.reset()
        done = False
        green_intervals_ns = 0
        green_intervals_ew = 0
        departures_ns = 0
        departures_ew = 0
        while not done:
            if controller_name == "Webster":
                action = controller.choose_action()
            else:
                action = controller.choose_action(state, training=False)
            if action not in (0, 1):
                raise RuntimeError(f"{controller_name} returned invalid action {action!r}")
            green_intervals_ns += int(action == 0)
            green_intervals_ew += int(action == 1)
            state, _, done, info = env.step(action)
            departures_ns += int(info["departures_ns"])
            departures_ew += int(info["departures_ew"])
        metrics = env.get_metrics()
        rows.append(
            {
                "scenario": scenario,
                "controller": controller_name,
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


def evaluate_state_only_model(
    scenario: str,
    rates: Tuple[float, float],
    model_path: Path,
    episodes: int,
    seed: int,
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
) -> List[Dict[str, object]]:
    agent = StateOnlyDQNAgent(seed=seed + 202)
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
                raise RuntimeError(f"State-only DQN returned invalid action {action!r}")
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


def load_original_controllers(scenario: str, rates: Tuple[float, float], seed: int):
    q_path = PROJECT_ROOT / "models" / f"q_learning_{scenario}_seed{seed}.npz"
    dqn_path = PROJECT_ROOT / "models" / f"dqn_{scenario}_seed{seed}.pt"
    if not q_path.is_file() or not dqn_path.is_file():
        raise FileNotFoundError(f"Missing original model(s) for {scenario}: {q_path}, {dqn_path}")
    q_agent = QLearningAgent(seed=seed + 101)
    q_agent.load(str(q_path))
    dqn_agent = DQNAgent(seed=seed + 202)
    dqn_agent.load(str(dqn_path))
    webster = WebsterController(arrival_rates=rates, interval_seconds=10.0)
    return {
        "Q-Learning": q_agent,
        "DQN": dqn_agent,
        "Webster": webster,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--train-seeds", type=int, nargs="*", default=[7, 21, 42])
    parser.add_argument("--scenarios", type=str, nargs="*", default=["high", "medium", "low"])
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")

    if args.estimate_only:
        scenario = "high"
        seed = 42
        start = time.perf_counter()
        train_state_only_dqn(scenario, SCENARIOS[scenario], min(args.train_episodes, 50), seed)
        end = time.perf_counter()
        print(f"Runtime estimate for 1 scenario/seed ({scenario}, seed={seed}): {end - start:.2f}s")
        return

    all_rows: List[Dict[str, object]] = []
    for scenario in args.scenarios:
        for seed in args.train_seeds:
            print(f"Training state-only DQN: scenario={scenario}, seed={seed}, episodes={args.train_episodes}")
            train_state_only_dqn(scenario, SCENARIOS[scenario], args.train_episodes, seed)

    for scenario in args.scenarios:
        rates = SCENARIOS[scenario]
        controllers = load_original_controllers(scenario, rates, seed=42)
        state_only_model = MODELS_DIR / f"state_only_dqn_{scenario}_seed42.pt"
        if not state_only_model.is_file():
            raise FileNotFoundError(f"Missing trained state-only model for {scenario}: {state_only_model}")

        # Evaluate original controllers and the state-only DQN on the same fixed episodes.
        for name, controller in controllers.items():
            rows = evaluate_controller(
                name,
                controller,
                scenario,
                rates,
                episodes=args.eval_episodes,
                seed=42,
            )
            all_rows.extend(rows)

        rows = evaluate_state_only_model(
            scenario,
            rates,
            state_only_model,
            episodes=args.eval_episodes,
            seed=42,
        )
        all_rows.extend(rows)

    write_csv(RESULTS_DIR / "state_only_extended_evaluation.csv", all_rows, EVAL_FIELDS)
    print(f"Saved evaluation rows to {RESULTS_DIR / 'state_only_extended_evaluation.csv'}")
    print(f"Controllers evaluated: original DQN, original Q-Learning, Webster, state-only DQN")
    print(f"Scenarios: {args.scenarios}")
    print(f"Train seeds: {args.train_seeds}")
    print(f"Evaluation episodes per scenario: {args.eval_episodes}")


if __name__ == "__main__":
    main()
