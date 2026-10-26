"""Short mock-environment tests for the original DQN SUMO experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dqn_agent import DQNAgent
from sumo_experiment.dqn_runner import (
    _validate_agent,
    evaluate_dqn,
    train_dqn,
    validate_action,
    validate_state,
)


class TinyEnvironment:
    def __init__(self, *, episode_horizon_seconds: float, decision_interval_seconds: float,
                 start_state: int = 0, fail_on_reset: bool = False,
                 fail_on_step: bool = False) -> None:
        self.episode_horizon_seconds = episode_horizon_seconds
        self.decision_interval_seconds = decision_interval_seconds
        self.start_state = start_state
        self.fail_on_reset = fail_on_reset
        self.fail_on_step = fail_on_step
        self.closed = False
        self.actions: list[int] = []
        self.step_count = 0

    def reset(self, seed: int | None = None) -> int:
        self.seed = seed
        if self.fail_on_reset:
            return 18
        return self.start_state

    def step(self, action: int):
        if self.fail_on_step:
            raise RuntimeError("simulated TraCI failure")
        self.actions.append(action)
        self.step_count += 1
        next_state = self.step_count % 18
        done = self.step_count == 4
        return next_state, -1.0, done, {"termination_reason": "horizon" if done else ""}

    def get_metrics(self) -> dict[str, float]:
        return {
            "total_stopped_waiting_time_sec_completed": 15.0,
            "avg_stopped_waiting_time_sec": 7.5,
            "time_weighted_avg_queue": 5.25,
            "throughput_veh_per_min": 2.4,
            "completed_vehicles": 3.0,
            "final_queue": 8.0,
            "simulated_time_sec": 40.0,
        }

    def close(self) -> None:
        self.closed = True


class SumoDQNTests(unittest.TestCase):
    def test_original_agent_state_encoding_and_action_contract(self) -> None:
        agent = DQNAgent(seed=3)
        _validate_agent(agent)
        self.assertEqual(agent.state_size, 18)
        self.assertEqual(agent.action_size, 2)
        self.assertEqual(agent.encode_state(0).shape, (18,))
        self.assertEqual(agent.encode_state(17).shape, (18,))
        self.assertEqual(float(agent.encode_state(17).sum()), 1.0)
        for state in (0, 17):
            self.assertEqual(validate_state(state), state)
        for action in (0, 1):
            self.assertEqual(validate_action(action), action)
        for state in (-1, 18, 1.0, True):
            with self.assertRaisesRegex(ValueError, r"\[0, 17\]"):
                validate_state(state)
        for action in (-1, 2, 0.0, True):
            with self.assertRaisesRegex(ValueError, r"0 \(NS\) or 1 \(EW\)"):
                validate_action(action)
        with self.assertRaisesRegex(ValueError, "state_size=18"):
            train_dqn(agent_kwargs={"state_size": 6})

    def test_tiny_training_performs_replay_update_and_round_trips_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            created: list[TinyEnvironment] = []

            def env_factory(**kwargs) -> TinyEnvironment:
                env = TinyEnvironment(**kwargs)
                created.append(env)
                return env

            trained, rows = train_dqn(
                episodes=1,
                seed=31,
                episode_horizon_seconds=40,
                decision_interval_seconds=10,
                batch_size=2,
                output_dir=root,
                env_factory=env_factory,
                agent_kwargs={"target_update_frequency": 1, "replay_capacity": 16},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["learning_updates"], 3)
            self.assertTrue(torch.isfinite(torch.tensor(rows[0]["mean_training_loss"])))
            self.assertEqual(rows[0]["cumulative_reward"], -4.0)
            self.assertEqual(rows[0]["time_weighted_avg_queue_vehicles"], 5.25)
            self.assertEqual(rows[0]["throughput_completed_veh_per_min"], 2.4)
            self.assertEqual(rows[0]["episode_duration_sec"], 40.0)
            self.assertEqual(rows[0]["termination_reason"], "horizon")
            self.assertTrue(all(env.closed for env in created))

            checkpoint = root / "dqn_model.pt"
            restored = DQNAgent(seed=101)
            restored.load(str(checkpoint))
            self.assertEqual(restored.state_size, 18)
            self.assertEqual(restored.action_size, 2)
            self.assertEqual(restored.learning_steps, trained.learning_steps)
            self.assertEqual(restored.epsilon, trained.epsilon)
            self.assertEqual(restored.batch_size, 2)
            self.assertGreater(len(restored.optimizer.state), 0)
            for name, tensor in trained.online_network.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.online_network.state_dict()[name]))
            for name, tensor in trained.target_network.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.target_network.state_dict()[name]))

            config = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["agent"]["network"], [18, 64, 64, 2])
            self.assertIn("optimizer state", config["checkpoint_contents"])
            self.assertIn("replay-buffer transitions", config["checkpoint_omissions"])
            with (root / "training_metrics.csv").open(encoding="utf-8", newline="") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 1)

    def test_evaluation_is_greedy_and_does_not_update_or_remember(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "model.pt"
            DQNAgent(batch_size=2, seed=2).save(str(checkpoint))
            created: list[TinyEnvironment] = []

            def env_factory(**kwargs) -> TinyEnvironment:
                env = TinyEnvironment(**kwargs)
                created.append(env)
                return env

            modes: list[bool] = []
            original_choose = DQNAgent.choose_action

            def choose_spy(agent, state, training=True):
                modes.append(training)
                return original_choose(agent, state, training=training)

            with patch.object(DQNAgent, "choose_action", choose_spy), patch.object(
                DQNAgent, "update", side_effect=AssertionError("evaluation must not learn")
            ):
                rows = evaluate_dqn(
                    episodes=1,
                    seed=80,
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

    def test_environment_errors_close_the_episode_environment(self) -> None:
        for failure in ("reset", "step"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary_dir:
                created: list[TinyEnvironment] = []

                def env_factory(**kwargs) -> TinyEnvironment:
                    env = TinyEnvironment(
                        **kwargs,
                        fail_on_reset=failure == "reset",
                        fail_on_step=failure == "step",
                    )
                    created.append(env)
                    return env

                with self.assertRaises((ValueError, RuntimeError)):
                    train_dqn(
                        episodes=1,
                        episode_horizon_seconds=40,
                        decision_interval_seconds=10,
                        batch_size=2,
                        output_dir=Path(temporary_dir),
                        env_factory=env_factory,
                    )
                self.assertEqual(len(created), 1)
                self.assertTrue(created[0].closed)


if __name__ == "__main__":
    unittest.main()
