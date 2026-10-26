"""Tests for integrating the existing six-feature agent with SUMO observations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from runs.state_only_dqn_experiment.dqn_agent import StateOnlyDQNAgent
from sumo_experiment.state_only_dqn_runner import (
    _validate_agent,
    evaluate_state_only_dqn,
    features_from_observation,
    train_state_only_dqn,
    validate_action,
    validate_features,
)


class TinySumoObservationEnv:
    def __init__(self, *, episode_horizon_seconds: float, decision_interval_seconds: float,
                 fail_on_reset: bool = False, fail_on_step: bool = False) -> None:
        self.episode_horizon_seconds = episode_horizon_seconds
        self.decision_interval_seconds = decision_interval_seconds
        self.fail_on_reset = fail_on_reset
        self.fail_on_step = fail_on_step
        self.closed = False
        self.step_count = 0
        self.actions: list[int] = []
        self.observation = {"phase": 0, "queue_ns": 4, "queue_ew": 3}

    def reset(self, seed: int | None = None) -> int:
        self.seed = seed
        self.step_count = 0
        self.observation = {"phase": 0, "queue_ns": 4, "queue_ew": 3}
        if self.fail_on_reset:
            return 18
        return 0  # Existing adapter return; the state-only runner uses raw counts.

    def get_queue_observation(self) -> dict[str, int]:
        if self.fail_on_reset:
            raise ValueError("test should reject the reset state before reading queues")
        return dict(self.observation)

    def step(self, action: int):
        if self.fail_on_step:
            raise RuntimeError("simulated TraCI failure")
        self.actions.append(action)
        self.step_count += 1
        self.observation = {
            "phase": action,
            "queue_ns": 5 + self.step_count,
            "queue_ew": 8,
        }
        done = self.step_count == 4
        return 0, -0.75, done, {
            **self.observation,
            "termination_reason": "horizon" if done else "",
        }

    def get_metrics(self) -> dict[str, float]:
        return {
            "total_stopped_waiting_time_sec_completed": 16.0,
            "avg_stopped_waiting_time_sec": 8.0,
            "time_weighted_avg_queue": 6.5,
            "throughput_veh_per_min": 1.5,
            "completed_vehicles": 2.0,
            "final_queue": 9.0,
            "simulated_time_sec": 40.0,
        }

    def close(self) -> None:
        self.closed = True


class StateOnlySumoTests(unittest.TestCase):
    def test_exact_six_feature_mapping_preserves_order_scale_and_unclipped_ranges(self) -> None:
        features = features_from_observation({"phase": 1, "queue_ns": 40, "queue_ew": 10})
        self.assertEqual(features.shape, (6,))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_allclose(features, [0.0, 1.0, 2.0, 0.5, 1.25, 1.5])
        self.assertGreater(float(features[2]), 1.0)  # Normalization is not clipped.

        agent = StateOnlyDQNAgent(seed=9)
        _validate_agent(agent)
        self.assertEqual(agent.state_size, 6)
        self.assertEqual(agent.action_size, 2)
        self.assertIn(agent.choose_action(features, training=False), (0, 1))
        for action in (0, 1):
            self.assertEqual(validate_action(action), action)

    def test_invalid_observation_shapes_values_and_actions_are_rejected(self) -> None:
        for value in ([0, 1, 0.1, 0.2, 0.15], [0, 1, 0.1, 0.2, 0.15, float("nan")],
                      [0, 0, 0.1, 0.2, 0.15, 0.1], [1, 0, -0.1, 0.2, 0.05, 0.3],
                      [1, 0, 0.1, 0.2, 0.4, 0.1]):
            with self.assertRaises(ValueError):
                validate_features(value)
        for observation in (
            {"phase": 2, "queue_ns": 0, "queue_ew": 0},
            {"phase": 0, "queue_ns": -1, "queue_ew": 0},
            {"phase": 0, "queue_ns": 1.5, "queue_ew": 0},
            {"phase": 0, "queue_ns": 0},
        ):
            with self.assertRaises(ValueError):
                features_from_observation(observation)
        for action in (-1, 2, 0.0, True):
            with self.assertRaisesRegex(ValueError, r"0 \(NS\) or 1 \(EW\)"):
                validate_action(action)

    def test_tiny_training_learns_and_saves_original_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            created: list[TinySumoObservationEnv] = []

            def env_factory(**kwargs) -> TinySumoObservationEnv:
                env = TinySumoObservationEnv(**kwargs)
                created.append(env)
                return env

            trained, rows = train_state_only_dqn(
                episodes=1,
                seed=32,
                episode_horizon_seconds=40,
                decision_interval_seconds=10,
                batch_size=2,
                output_dir=root,
                env_factory=env_factory,
                agent_kwargs={"target_update_frequency": 1, "replay_capacity": 16},
            )
            self.assertEqual(rows[0]["learning_updates"], 3)
            self.assertTrue(torch.isfinite(torch.tensor(rows[0]["mean_training_loss"])))
            self.assertEqual(rows[0]["cumulative_reward"], -3.0)
            self.assertEqual(rows[0]["time_weighted_avg_queue_vehicles"], 6.5)
            self.assertEqual(rows[0]["throughput_completed_veh_per_min"], 1.5)
            self.assertEqual(rows[0]["episode_duration_sec"], 40.0)
            self.assertEqual(rows[0]["termination_reason"], "horizon")
            self.assertTrue(all(env.closed for env in created))

            checkpoint = root / "state_only_dqn_model.pt"
            restored = StateOnlyDQNAgent(seed=101)
            restored.load(str(checkpoint))
            self.assertEqual(restored.state_size, 6)
            self.assertEqual(restored.action_size, 2)
            self.assertEqual(restored.learning_steps, trained.learning_steps)
            self.assertEqual(restored.epsilon, trained.epsilon)
            self.assertEqual(restored.batch_size, 2)
            self.assertGreater(len(restored.optimizer.state), 0)
            for name, tensor in trained.online_network.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.online_network.state_dict()[name]))
            config = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["feature_order"], [
                "phase_ns", "phase_ew", "ns_queue_norm", "ew_queue_norm",
                "total_queue_norm", "imbalance_norm",
            ])
            self.assertEqual(config["agent"]["network"], [6, 64, 64, 2])
            self.assertIn("NumPy random generator state", config["checkpoint_omissions"])
            with (root / "training_metrics.csv").open(encoding="utf-8", newline="") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 1)

    def test_greedy_evaluation_does_not_learn_and_closes_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "model.pt"
            StateOnlyDQNAgent(batch_size=2, seed=2).save(str(checkpoint))
            created: list[TinySumoObservationEnv] = []

            def env_factory(**kwargs) -> TinySumoObservationEnv:
                env = TinySumoObservationEnv(**kwargs)
                created.append(env)
                return env

            modes: list[bool] = []
            original_choose = StateOnlyDQNAgent.choose_action

            def choose_spy(agent, state, training=True):
                modes.append(training)
                return original_choose(agent, state, training=training)

            with patch.object(StateOnlyDQNAgent, "choose_action", choose_spy), patch.object(
                StateOnlyDQNAgent, "update", side_effect=AssertionError("evaluation must not learn")
            ):
                rows = evaluate_state_only_dqn(
                    episodes=1,
                    seed=81,
                    model_path=checkpoint,
                    episode_horizon_seconds=40,
                    decision_interval_seconds=10,
                    output_dir=root,
                    env_factory=env_factory,
                )
            self.assertEqual(modes, [False, False, False, False])
            self.assertEqual(rows[0]["learning_updates"], 0)
            self.assertFalse(rows[0]["exploration_enabled"])
            self.assertTrue(all(env.closed for env in created))
            config = json.loads((root / "evaluation_config.json").read_text(encoding="utf-8"))
            self.assertFalse(config["exploration_enabled"])
            with (root / "evaluation_metrics.csv").open(encoding="utf-8", newline="") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 1)

    def test_errors_close_environments(self) -> None:
        for failure in ("reset", "step"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary_dir:
                created: list[TinySumoObservationEnv] = []

                def env_factory(**kwargs) -> TinySumoObservationEnv:
                    env = TinySumoObservationEnv(
                        **kwargs,
                        fail_on_reset=failure == "reset",
                        fail_on_step=failure == "step",
                    )
                    created.append(env)
                    return env

                with self.assertRaises((ValueError, RuntimeError)):
                    train_state_only_dqn(
                        episodes=1,
                        episode_horizon_seconds=40,
                        decision_interval_seconds=10,
                        batch_size=2,
                        output_dir=Path(temporary_dir),
                        env_factory=env_factory,
                    )
                self.assertTrue(created[0].closed)


if __name__ == "__main__":
    unittest.main()
