"""Unit tests for SUMO Q-learning orchestration without a live SUMO process."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from q_learning import QLearningAgent
from sumo_experiment.q_learning_runner import (
    evaluate_q_learning,
    train_q_learning,
    validate_action,
    validate_state,
)


class TinyEnvironment:
    def __init__(self, *, episode_horizon_seconds: float, decision_interval_seconds: float,
                 start_state: int = 0, fail_on_reset: bool = False) -> None:
        self.episode_horizon_seconds = episode_horizon_seconds
        self.decision_interval_seconds = decision_interval_seconds
        self.start_state = start_state
        self.fail_on_reset = fail_on_reset
        self.closed = False
        self.actions: list[int] = []
        self.step_count = 0

    def reset(self, seed: int | None = None) -> int:
        self.seed = seed
        if self.fail_on_reset:
            return 18
        return self.start_state

    def step(self, action: int):
        self.actions.append(action)
        self.step_count += 1
        next_state = 1 if self.step_count == 1 else 17
        done = self.step_count == 2
        return next_state, -1.25, done, {
            "termination_reason": "horizon" if done else "",
        }

    def get_metrics(self) -> dict[str, float]:
        return {
            "total_stopped_waiting_time_sec_completed": 13.0,
            "avg_stopped_waiting_time_sec": 6.5,
            "time_weighted_avg_queue": 4.0,
            "throughput_veh_per_min": 2.0,
            "completed_vehicles": 2.0,
            "simulated_time_sec": 20.0,
        }

    def close(self) -> None:
        self.closed = True


class SumoQLearningTests(unittest.TestCase):
    def test_state_and_action_contract_is_explicit(self) -> None:
        for state in (0, 17):
            self.assertEqual(validate_state(state), state)
        for action in (0, 1):
            self.assertEqual(validate_action(action), action)
        for invalid_state in (-1, 18, 1.0, True):
            with self.assertRaisesRegex(ValueError, r"\[0, 17\]"):
                validate_state(invalid_state)
        for invalid_action in (-1, 2, 0.0, True):
            with self.assertRaisesRegex(ValueError, r"0 \(NS\) or 1 \(EW\)"):
                validate_action(invalid_action)

    def test_tiny_training_saves_loadable_q_table_config_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            created: list[TinyEnvironment] = []

            def env_factory(**kwargs) -> TinyEnvironment:
                env = TinyEnvironment(**kwargs)
                created.append(env)
                return env

            trained_agent, rows = train_q_learning(
                episodes=2,
                seed=71,
                episode_horizon_seconds=20,
                decision_interval_seconds=10,
                output_dir=root,
                env_factory=env_factory,
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(env.closed for env in created))
            self.assertEqual([row["sumo_seed"] for row in rows], [71, 72])
            self.assertEqual(rows[0]["cumulative_reward"], -2.5)
            self.assertEqual(rows[0]["total_stopped_waiting_time_sec_completed"], 13.0)
            self.assertEqual(rows[0]["time_weighted_avg_queue_vehicles"], 4.0)
            self.assertEqual(rows[0]["throughput_completed_veh_per_min"], 2.0)
            self.assertEqual(rows[0]["episode_duration_sec"], 20.0)
            self.assertEqual(rows[0]["termination_reason"], "horizon")

            checkpoint = root / "q_table.npz"
            loaded = QLearningAgent(seed=9)
            loaded.load(str(checkpoint))
            self.assertEqual(loaded.q_table.shape, (18, 2))
            self.assertEqual(loaded.q_table.tolist(), trained_agent.q_table.tolist())
            with (root / "training_metrics.csv").open(encoding="utf-8", newline="") as source:
                saved_rows = list(csv.DictReader(source))
            self.assertEqual(len(saved_rows), 2)
            config = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["action_meanings"], {"0": "NS green", "1": "EW green"})
            self.assertEqual(config["agent_hyperparameters"]["alpha"], 0.1)

    def test_evaluation_loads_checkpoint_and_disables_exploration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "q_table.npz"
            QLearningAgent(seed=1).save(str(checkpoint))
            created: list[TinyEnvironment] = []

            def env_factory(**kwargs) -> TinyEnvironment:
                env = TinyEnvironment(**kwargs)
                created.append(env)
                return env

            called_modes: list[bool] = []
            original_choose_action = QLearningAgent.choose_action

            def choose_action_spy(agent, state, training=True):
                called_modes.append(training)
                return original_choose_action(agent, state, training=training)

            with patch.object(QLearningAgent, "choose_action", choose_action_spy):
                rows = evaluate_q_learning(
                    episodes=1,
                    seed=90,
                    model_path=checkpoint,
                    episode_horizon_seconds=20,
                    decision_interval_seconds=10,
                    output_dir=root,
                    env_factory=env_factory,
                )
            self.assertEqual(called_modes, [False, False])
            self.assertTrue(all(env.closed for env in created))
            self.assertEqual(rows[0]["termination_reason"], "horizon")
            config = json.loads((root / "evaluation_config.json").read_text(encoding="utf-8"))
            self.assertFalse(config["exploration_enabled"])
            with (root / "evaluation_metrics.csv").open(encoding="utf-8", newline="") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 1)

    def test_invalid_environment_state_raises_and_closes_environment(self) -> None:
        created: list[TinyEnvironment] = []

        def env_factory(**kwargs) -> TinyEnvironment:
            env = TinyEnvironment(**kwargs, fail_on_reset=True)
            created.append(env)
            return env

        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, r"reset state.*\[0, 17\]"):
                train_q_learning(
                    episodes=1,
                    episode_horizon_seconds=20,
                    decision_interval_seconds=10,
                    output_dir=Path(temporary_dir),
                    env_factory=env_factory,
                )
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)


if __name__ == "__main__":
    unittest.main()
