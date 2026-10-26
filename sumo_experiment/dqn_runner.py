"""Training and greedy evaluation for the original DQN on the SUMO adapter.

The original DQN consumes integer observations 0..17, one-hot encodes them to
18 network inputs, and outputs values for action 0 (NS) and action 1 (EW). The
agent, network, replay buffer, optimizer, target network, update rule, and
checkpoint format are imported unchanged from the project root.
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

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SUMO_EXPERIMENT_DIR / "runs" / "original_dqn"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dqn_agent import DQNAgent  # noqa: E402
if __package__:
    from .sumo_environment import ROUTES_FILE, SUSTAINED_ROUTES_ENV, SumoTrafficEnvironment  # noqa: E402
else:
    from sumo_environment import ROUTES_FILE, SUSTAINED_ROUTES_ENV, SumoTrafficEnvironment  # noqa: E402


METRIC_FIELDS = (
    "episode", "sumo_seed", "cumulative_reward",
    "total_stopped_waiting_time_sec_completed",
    "avg_stopped_waiting_time_sec_completed", "time_weighted_avg_queue_vehicles",
    "throughput_completed_veh_per_min", "completed_vehicles",
    "final_queue_vehicles", "episode_duration_sec", "decision_steps", "termination_reason",
    "epsilon", "learning_updates", "mean_training_loss", "exploration_enabled",
)


def validate_state(state: Any, context: str = "state") -> int:
    """Require a non-boolean integer in the original DQN's 18-state space."""
    if isinstance(state, bool) or not isinstance(state, Integral) or not 0 <= int(state) < 18:
        raise ValueError(f"SUMO DQN {context} must be an integer in [0, 17], got {state!r}")
    return int(state)


def validate_action(action: Any, context: str = "action") -> int:
    if isinstance(action, bool) or not isinstance(action, Integral) or int(action) not in (0, 1):
        raise ValueError(f"SUMO DQN {context} must be 0 (NS) or 1 (EW), got {action!r}")
    return int(action)


def _validate_agent(agent: Any) -> None:
    if getattr(agent, "state_size", None) != 18 or getattr(agent, "action_size", None) != 2:
        raise ValueError("SUMO original DQN requires state_size=18 and action_size=2")
    network = getattr(agent, "online_network", None)
    linear_layers = [layer for layer in network.modules() if isinstance(layer, nn.Linear)] if network else []
    if not linear_layers or linear_layers[0].in_features != 18 or linear_layers[-1].out_features != 2:
        raise ValueError("SUMO original DQN network must map 18 one-hot inputs to 2 action values")


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


def _make_env(env_factory: Callable[..., Any], episode_horizon_seconds: float,
              decision_interval_seconds: float) -> Any:
    return env_factory(
        episode_horizon_seconds=episode_horizon_seconds,
        decision_interval_seconds=decision_interval_seconds,
    )


def _episode_metrics(env: Any, episode: int, sumo_seed: int, reward_sum: float,
                     decision_steps: int, termination_reason: str, epsilon: float,
                     learning_updates: int, losses: list[float],
                     exploration_enabled: bool) -> dict[str, Any]:
    metrics = env.get_metrics()
    total_stopped_wait = float(metrics["total_stopped_waiting_time_sec_completed"])
    average_stopped_wait = float(metrics["avg_stopped_waiting_time_sec"])
    average_queue = float(metrics["time_weighted_avg_queue"])
    throughput = float(metrics["throughput_veh_per_min"])
    completed = float(metrics["completed_vehicles"])
    final_queue = float(metrics["final_queue"])
    duration = float(metrics["simulated_time_sec"])
    for key, value in (
        ("total stopped waiting time", total_stopped_wait),
        ("average queue", average_queue),
        ("throughput", throughput),
        ("completed vehicles", completed),
        ("final queue", final_queue),
        ("episode duration", duration),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"SUMO metric {key} must be finite and nonnegative, got {value!r}")
    if not math.isfinite(average_stopped_wait) and not math.isnan(average_stopped_wait):
        raise ValueError("SUMO average stopped waiting time must be finite or NaN when unavailable")
    return {
        "episode": episode,
        "sumo_seed": sumo_seed,
        "cumulative_reward": reward_sum,
        "total_stopped_waiting_time_sec_completed": total_stopped_wait,
        "avg_stopped_waiting_time_sec_completed": average_stopped_wait,
        "time_weighted_avg_queue_vehicles": average_queue,
        "throughput_completed_veh_per_min": throughput,
        "completed_vehicles": int(completed),
        "final_queue_vehicles": final_queue,
        "episode_duration_sec": duration,
        "decision_steps": decision_steps,
        "termination_reason": termination_reason,
        "epsilon": epsilon,
        "learning_updates": learning_updates,
        "mean_training_loss": sum(losses) / len(losses) if losses else float("nan"),
        "exploration_enabled": exploration_enabled,
    }


def _agent_configuration(agent: DQNAgent) -> dict[str, Any]:
    return {
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "network": [18, 64, 64, 2],
        "learning_rate": agent.learning_rate,
        "gamma": agent.gamma,
        "epsilon_initial": 1.0,
        "epsilon_final": agent.epsilon,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "replay_capacity": agent.replay_capacity,
        "batch_size": agent.batch_size,
        "target_update_frequency": agent.target_update_frequency,
        "device": str(agent.device),
    }


def _configuration(episodes: int, seed: int, episode_horizon_seconds: float,
                   decision_interval_seconds: float, algorithm_seed: int,
                   agent: DQNAgent) -> dict[str, Any]:
    return {
        "experiment": "SUMO original DQN",
        "state_encoding": "integer state 0..17; original DQN one-hot encodes to 18 floats",
        "action_meanings": {"0": "NS green", "1": "EW green"},
        "queue_observation": "SUMO halting vehicles (speed <= 0.1 m/s) on A1 inbound lanes; not the custom simulator's FIFO queue",
        "reward": "-((observed_queue_before + observed_queue_after) / 2) / 10 per decision interval",
        "waiting_metric": "sum of sampled stopped seconds for completed route trips; unfinished trips excluded",
        "average_queue_metric": "time-weighted mean of NS+EW A1 inbound halting vehicles",
        "throughput_metric": "completed route trips per simulated minute",
        "termination": "configured horizon or SUMO reports no expected/active vehicles",
        "episodes": episodes,
        "seed": seed,
        "sumo_seeds": [seed + index for index in range(episodes)],
        "algorithm_seed": algorithm_seed,
        "episode_horizon_seconds": episode_horizon_seconds,
        "decision_interval_seconds": decision_interval_seconds,
        "routes_file": str(Path(os.environ.get(SUSTAINED_ROUTES_ENV, str(ROUTES_FILE))).expanduser().resolve()),
        "agent": _agent_configuration(agent),
        "checkpoint_contents": [
            "online network weights", "target network weights", "optimizer state",
            "epsilon", "learning-step count", "agent hyperparameters",
        ],
        "checkpoint_omissions": [
            "replay-buffer transitions", "Python random generator state", "PyTorch random generator state",
        ],
        "reproducibility": "SUMO and agent RNGs are seeded per run. Exact resumed training is not guaranteed because the existing checkpoint omits replay contents and RNG states; cross-version/platform determinism is also not guaranteed.",
    }


def train_dqn(
    episodes: int = 2,
    seed: int = 42,
    episode_horizon_seconds: float = 360.0,
    decision_interval_seconds: float = 10.0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 64,
    env_factory: Callable[..., Any] = SumoTrafficEnvironment,
    agent_factory: Callable[..., DQNAgent] = DQNAgent,
    agent_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[DQNAgent, list[dict[str, Any]]]:
    """Train the unchanged DQN; defaults are a short smoke experiment."""
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    output_dir = Path(output_dir)
    algorithm_seed = int(seed) + 202
    kwargs = dict(agent_kwargs or {})
    if "state_size" in kwargs and kwargs["state_size"] != 18:
        raise ValueError("SUMO original DQN requires state_size=18; state-only checkpoints are incompatible")
    if "action_size" in kwargs and kwargs["action_size"] != 2:
        raise ValueError("SUMO original DQN requires action_size=2")
    kwargs["batch_size"] = batch_size
    agent = agent_factory(seed=algorithm_seed, **kwargs)
    _validate_agent(agent)
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
            start_learning_steps = agent.learning_steps
            losses: list[float] = []
            while not done:
                action = validate_action(agent.choose_action(state, training=True), "agent action")
                next_state_raw, reward_raw, done, info = env.step(action)
                next_state = validate_state(next_state_raw, "next state")
                reward = float(reward_raw)
                if not math.isfinite(reward):
                    raise ValueError(f"SUMO reward must be finite, got {reward_raw!r}")
                loss = agent.update(state, action, reward, next_state, bool(done))
                if loss is not None:
                    loss = float(loss)
                    if not math.isfinite(loss):
                        raise ValueError(f"DQN training loss must be finite, got {loss!r}")
                    losses.append(loss)
                state = next_state
                reward_sum += reward
                decision_steps += 1
                if done:
                    termination_reason = str(info.get("termination_reason", "unspecified"))
            agent.decay_epsilon()
            rows.append(_episode_metrics(
                env, episode, sumo_seed, reward_sum, decision_steps,
                termination_reason, agent.epsilon,
                agent.learning_steps - start_learning_steps, losses, True,
            ))
        finally:
            env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "dqn_model.pt"
    agent.save(str(checkpoint_path))
    _write_csv(output_dir / "training_metrics.csv", rows)
    config = _configuration(
        int(episodes), int(seed), float(episode_horizon_seconds),
        float(decision_interval_seconds), algorithm_seed, agent,
    )
    config.update({
        "checkpoint": checkpoint_path.name,
        "training_metrics": "training_metrics.csv",
        "exploration_enabled": True,
        "routes_file": str(Path(os.environ.get(SUSTAINED_ROUTES_ENV, str(ROUTES_FILE))).expanduser().resolve()),
    })
    _write_json(output_dir / "training_config.json", config)
    return agent, rows


def evaluate_dqn(
    episodes: int = 1,
    seed: int = 142,
    model_path: Path | str = DEFAULT_OUTPUT_DIR / "dqn_model.pt",
    episode_horizon_seconds: float = 360.0,
    decision_interval_seconds: float = 10.0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    env_factory: Callable[..., Any] = SumoTrafficEnvironment,
    agent_factory: Callable[..., DQNAgent] = DQNAgent,
) -> list[dict[str, Any]]:
    """Evaluate a saved original-DQN checkpoint without exploration or updates."""
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    model_path, output_dir = Path(model_path), Path(output_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"SUMO DQN checkpoint not found: {model_path}; run train_dqn.py first")

    agent = agent_factory(seed=int(seed) + 202)
    agent.load(str(model_path))
    _validate_agent(agent)
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
            losses: list[float] = []
            while not done:
                action = validate_action(agent.choose_action(state, training=False), "agent action")
                next_state_raw, reward_raw, done, info = env.step(action)
                next_state = validate_state(next_state_raw, "next state")
                reward = float(reward_raw)
                if not math.isfinite(reward):
                    raise ValueError(f"SUMO reward must be finite, got {reward_raw!r}")
                # No remember/update/learn call in evaluation.
                state = next_state
                reward_sum += reward
                decision_steps += 1
                if done:
                    termination_reason = str(info.get("termination_reason", "unspecified"))
            rows.append(_episode_metrics(
                env, episode, sumo_seed, reward_sum, decision_steps,
                termination_reason, agent.epsilon, 0, losses, False,
            ))
        finally:
            env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "evaluation_metrics.csv", rows)
    config = _configuration(
        int(episodes), int(seed), float(episode_horizon_seconds),
        float(decision_interval_seconds), int(seed) + 202, agent,
    )
    config.update({
        "checkpoint": str(model_path.resolve()),
        "evaluation_metrics": "evaluation_metrics.csv",
        "exploration_enabled": False,
        "routes_file": str(Path(os.environ.get(SUSTAINED_ROUTES_ENV, str(ROUTES_FILE))).expanduser().resolve()),
    })
    _write_json(output_dir / "evaluation_config.json", config)
    return rows
