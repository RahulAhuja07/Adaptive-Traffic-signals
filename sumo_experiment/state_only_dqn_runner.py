"""SUMO runner for the existing six-feature StateOnlyDQNAgent.

This module reuses ``runs.state_only_dqn_experiment.dqn_agent`` without
changing its network or learning logic. SUMO raw counts feed the exact original
feature formula, with its queue scale of 20 and no clipping.
"""

from __future__ import annotations

import csv
import json
import math
from numbers import Integral
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SUMO_EXPERIMENT_DIR / "runs" / "state_only_dqn"
QUEUE_SCALE = 20.0
FEATURE_NAMES = (
    "phase_ns", "phase_ew", "ns_queue_norm", "ew_queue_norm",
    "total_queue_norm", "imbalance_norm",
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.state_only_dqn_experiment.dqn_agent import StateOnlyDQNAgent  # noqa: E402
if __package__:
    from .sumo_environment import ROUTES_FILE, SUSTAINED_ROUTES_ENV, SumoTrafficEnvironment  # noqa: E402
else:
    from sumo_environment import ROUTES_FILE, SUSTAINED_ROUTES_ENV, SumoTrafficEnvironment  # noqa: E402


METRIC_FIELDS = (
    "episode", "sumo_seed", "cumulative_reward",
    "total_stopped_waiting_time_sec_completed",
    "avg_stopped_waiting_time_sec_completed", "time_weighted_avg_queue_vehicles",
    "throughput_completed_veh_per_min", "completed_vehicles", "final_queue_vehicles",
    "episode_duration_sec", "decision_steps", "termination_reason", "epsilon",
    "learning_updates", "mean_training_loss", "exploration_enabled",
)


def validate_action(action: Any, context: str = "action") -> int:
    if isinstance(action, bool) or not isinstance(action, Integral) or int(action) not in (0, 1):
        raise ValueError(f"SUMO state-only DQN {context} must be 0 (NS) or 1 (EW), got {action!r}")
    return int(action)


def validate_features(features: Any, context: str = "state") -> np.ndarray:
    """Validate the six original features without clipping or rescaling them."""
    try:
        vector = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"SUMO state-only DQN {context} must be a numeric six-feature vector") from error
    if vector.shape != (6,):
        raise ValueError(f"SUMO state-only DQN {context} must have shape (6,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"SUMO state-only DQN {context} must contain only finite values")
    if vector[0] not in (0.0, 1.0) or vector[1] not in (0.0, 1.0) or vector[0] + vector[1] != 1.0:
        raise ValueError("phase features must be a one-hot NS/EW pair")
    if np.any(vector[2:] < 0.0):
        raise ValueError("normalized queue and imbalance features must be nonnegative")
    expected_total = (float(vector[2]) + float(vector[3])) / 2.0
    expected_imbalance = abs(float(vector[2]) - float(vector[3]))
    if not math.isclose(float(vector[4]), expected_total, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("total_queue_norm is inconsistent with the directional queue features")
    if not math.isclose(float(vector[5]), expected_imbalance, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("imbalance_norm is inconsistent with the directional queue features")
    return vector.copy()


def features_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    """Apply the original RichStateTrafficEnvironment feature formula to SUMO counts."""
    required = ("phase", "queue_ns", "queue_ew")
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"SUMO queue observation is missing fields: {', '.join(missing)}")
    phase, queue_ns, queue_ew = (observation[key] for key in required)
    if isinstance(phase, bool) or not isinstance(phase, Integral) or int(phase) not in (0, 1):
        raise ValueError(f"SUMO phase must be 0 (NS) or 1 (EW), got {phase!r}")
    for direction, queue in (("NS", queue_ns), ("EW", queue_ew)):
        if isinstance(queue, bool) or not isinstance(queue, Integral) or int(queue) < 0:
            raise ValueError(f"SUMO {direction} queue must be a nonnegative integer, got {queue!r}")
    ns_norm = float(queue_ns) / QUEUE_SCALE
    ew_norm = float(queue_ew) / QUEUE_SCALE
    return validate_features(np.asarray([
        1.0 if phase == 0 else 0.0,
        1.0 if phase == 1 else 0.0,
        ns_norm,
        ew_norm,
        (float(queue_ns) + float(queue_ew)) / (2.0 * QUEUE_SCALE),
        abs(float(queue_ns) - float(queue_ew)) / QUEUE_SCALE,
    ], dtype=np.float32))


def _validate_agent(agent: Any) -> None:
    if getattr(agent, "state_size", None) != 6 or getattr(agent, "action_size", None) != 2:
        raise ValueError("SUMO state-only DQN requires state_size=6 and action_size=2")
    network = getattr(agent, "online_network", None)
    linear_layers = [layer for layer in network.modules() if isinstance(layer, nn.Linear)] if network else []
    if not linear_layers or linear_layers[0].in_features != 6 or linear_layers[-1].out_features != 2:
        raise ValueError("state-only DQN network must map six feature inputs to two action values")


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
    total_wait = float(metrics["total_stopped_waiting_time_sec_completed"])
    average_wait = float(metrics["avg_stopped_waiting_time_sec"])
    average_queue = float(metrics["time_weighted_avg_queue"])
    throughput = float(metrics["throughput_veh_per_min"])
    completed = float(metrics["completed_vehicles"])
    final_queue = float(metrics["final_queue"])
    duration = float(metrics["simulated_time_sec"])
    for name, value in (
        ("total stopped waiting time", total_wait), ("average queue", average_queue),
        ("throughput", throughput), ("completed vehicles", completed),
        ("final queue", final_queue), ("episode duration", duration),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"SUMO metric {name} must be finite and nonnegative, got {value!r}")
    if not math.isfinite(average_wait) and not math.isnan(average_wait):
        raise ValueError("SUMO average stopped waiting time must be finite or NaN when unavailable")
    return {
        "episode": episode,
        "sumo_seed": sumo_seed,
        "cumulative_reward": reward_sum,
        "total_stopped_waiting_time_sec_completed": total_wait,
        "avg_stopped_waiting_time_sec_completed": average_wait,
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


def _agent_configuration(agent: StateOnlyDQNAgent) -> dict[str, Any]:
    return {
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "network": [6, 64, 64, 2],
        "feature_order": list(FEATURE_NAMES),
        "queue_scale_vehicles": QUEUE_SCALE,
        "learning_rate": agent.learning_rate,
        "gamma": agent.gamma,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "epsilon_final": agent.epsilon,
        "replay_capacity": agent.replay_capacity,
        "batch_size": agent.batch_size,
        "target_update_frequency": agent.target_update_frequency,
        "device": str(agent.device),
    }


def _configuration(episodes: int, seed: int, episode_horizon_seconds: float,
                   decision_interval_seconds: float, algorithm_seed: int,
                   agent: StateOnlyDQNAgent) -> dict[str, Any]:
    return {
        "experiment": "SUMO state-only DQN",
        "state_size": 6,
        "action_size": 2,
        "feature_order": list(FEATURE_NAMES),
        "features": {
            "phase_ns": "1 when NS green is active, else 0",
            "phase_ew": "1 when EW green is active, else 0",
            "ns_queue_norm": "NS inbound halting count / 20",
            "ew_queue_norm": "EW inbound halting count / 20",
            "total_queue_norm": "(NS + EW inbound halting counts) / 40",
            "imbalance_norm": "absolute(NS - EW inbound halting counts) / 20",
        },
        "queue_observation": "SUMO halting vehicles (speed <= 0.1 m/s) summed over A1 inbound lanes; same feature formulas and scale as the source environment, but different queue semantics from its FIFO length",
        "action_meanings": {"0": "NS green", "1": "EW green"},
        "reward": "-((observed_queue_before + observed_queue_after) / 2) / 10 per decision interval",
        "waiting_metric": "sampled stopped seconds for completed route trips; unfinished trips excluded",
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
            "replay-buffer transitions", "NumPy random generator state", "PyTorch random generator state",
        ],
        "reproducibility": "SUMO and agent RNGs are seeded, but exact resumed training is not guaranteed because replay and RNG state are not checkpointed; cross-platform and cross-version determinism may vary.",
    }


def train_state_only_dqn(
    episodes: int = 2,
    seed: int = 42,
    episode_horizon_seconds: float = 360.0,
    decision_interval_seconds: float = 10.0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 64,
    env_factory: Callable[..., Any] = SumoTrafficEnvironment,
    agent_factory: Callable[..., StateOnlyDQNAgent] = StateOnlyDQNAgent,
    agent_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[StateOnlyDQNAgent, list[dict[str, Any]]]:
    """Train the existing six-feature agent; defaults are a short smoke run."""
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    kwargs = dict(agent_kwargs or {})
    if "state_size" in kwargs and kwargs["state_size"] != 6:
        raise ValueError("SUMO state-only DQN requires state_size=6; 18-state checkpoints are incompatible")
    if "action_size" in kwargs and kwargs["action_size"] != 2:
        raise ValueError("SUMO state-only DQN requires action_size=2")
    kwargs["batch_size"] = batch_size
    algorithm_seed = int(seed) + 202
    agent = agent_factory(seed=algorithm_seed, **kwargs)
    _validate_agent(agent)
    rows: list[dict[str, Any]] = []

    for episode in range(1, int(episodes) + 1):
        sumo_seed = int(seed) + episode - 1
        env = _make_env(env_factory, episode_horizon_seconds, decision_interval_seconds)
        try:
            # reset still returns the adapter's 18-state integer for its
            # original agents; this agent deliberately reads raw queue counts.
            env.reset(seed=sumo_seed)
            state = features_from_observation(env.get_queue_observation())
            done = False
            reward_sum = 0.0
            decision_steps = 0
            termination_reason = ""
            start_learning_steps = agent.learning_steps
            losses: list[float] = []
            while not done:
                action = validate_action(agent.choose_action(state, training=True), "agent action")
                _, reward_raw, done, info = env.step(action)
                next_state = features_from_observation(info)
                reward = float(reward_raw)
                if not math.isfinite(reward):
                    raise ValueError(f"SUMO reward must be finite, got {reward_raw!r}")
                loss = agent.update(state, action, reward, next_state, bool(done))
                if loss is not None:
                    loss = float(loss)
                    if not math.isfinite(loss):
                        raise ValueError(f"state-only DQN training loss must be finite, got {loss!r}")
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "state_only_dqn_model.pt"
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


def evaluate_state_only_dqn(
    episodes: int = 1,
    seed: int = 142,
    model_path: Path | str = DEFAULT_OUTPUT_DIR / "state_only_dqn_model.pt",
    episode_horizon_seconds: float = 360.0,
    decision_interval_seconds: float = 10.0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    env_factory: Callable[..., Any] = SumoTrafficEnvironment,
    agent_factory: Callable[..., StateOnlyDQNAgent] = StateOnlyDQNAgent,
) -> list[dict[str, Any]]:
    """Evaluate a saved six-feature checkpoint with exploration disabled."""
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    model_path, output_dir = Path(model_path), Path(output_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"SUMO state-only DQN checkpoint not found: {model_path}")
    agent = agent_factory(seed=int(seed) + 202)
    agent.load(str(model_path))
    _validate_agent(agent)
    rows: list[dict[str, Any]] = []

    for episode in range(1, int(episodes) + 1):
        sumo_seed = int(seed) + episode - 1
        env = _make_env(env_factory, episode_horizon_seconds, decision_interval_seconds)
        try:
            env.reset(seed=sumo_seed)
            state = features_from_observation(env.get_queue_observation())
            done = False
            reward_sum = 0.0
            decision_steps = 0
            termination_reason = ""
            while not done:
                action = validate_action(agent.choose_action(state, training=False), "agent action")
                _, reward_raw, done, info = env.step(action)
                state = features_from_observation(info)
                reward = float(reward_raw)
                if not math.isfinite(reward):
                    raise ValueError(f"SUMO reward must be finite, got {reward_raw!r}")
                # Evaluation intentionally does not call update/remember/learn.
                reward_sum += reward
                decision_steps += 1
                if done:
                    termination_reason = str(info.get("termination_reason", "unspecified"))
            rows.append(_episode_metrics(
                env, episode, sumo_seed, reward_sum, decision_steps,
                termination_reason, agent.epsilon, 0, [], False,
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
