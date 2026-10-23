"""Small integration tests for Part 3 training, evaluation, and plotting."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from dqn_agent import DQNAgent
from evaluate import evaluate_scenario, run_evaluation, run_greedy_episode
from plot_results import generate_plots
from q_learning import QLearningAgent
from train import run_training
from utils import (
    EVALUATION_FIELDS,
    aggregate_evaluation,
    make_environment,
    model_path,
    read_csv,
    training_log_path,
)
from webster_controller import WebsterController


class PipelineTests(unittest.TestCase):
    def test_small_training_saves_logs_and_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_training(
                episodes=2,
                seed=23,
                scenario="low",
                episode_steps=5,
                output_root=root,
            )
            for algorithm in ("q_learning", "dqn"):
                log_path = training_log_path(root, algorithm, "low", 23)
                self.assertEqual(len(read_csv(log_path)), 2)
                self.assertTrue(model_path(root, algorithm, "low", 23).is_file())

    def test_missing_models_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Run train.py"):
                evaluate_scenario(
                    "low", (1.0, 0.8), 1, 42, 42,
                    episode_steps=2, output_root=Path(directory),
                )

    def test_greedy_evaluation_does_not_update_agents(self):
        rates = (1.0, 0.8)
        env = make_environment(rates, episode_steps=5, seed=1)
        q_agent = QLearningAgent(seed=1)
        q_agent.q_table[0] = [0.3, -0.1]
        q_before = q_agent.q_table.copy()
        dqn = DQNAgent(batch_size=2, seed=1)
        dqn_before = [parameter.detach().clone() for parameter in dqn.online_network.parameters()]
        for name, controller in (("Q-Learning", q_agent), ("DQN", dqn)):
            row = run_greedy_episode(env, controller, name, "low", 1, 99)
            self.assertEqual(row["total_arrived"], row["total_departed"] + row["final_queue"])
        self.assertEqual(len(q_agent.replay_buffer) if hasattr(q_agent, "replay_buffer") else 0, 0)
        self.assertEqual(len(dqn.replay_buffer), 0)
        self.assertTrue((q_agent.q_table == q_before).all())
        for before, after in zip(dqn_before, dqn.online_network.parameters()):
            torch.testing.assert_close(before, after)
        self.assertEqual(dqn.learning_steps, 0)

    def test_evaluation_uses_shared_seeds_and_writes_required_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_training(episodes=1, seed=31, scenario="low", episode_steps=4, output_root=root)
            rows, summary = run_evaluation(
                episodes=2,
                seed=70,
                model_seed=31,
                scenario="low",
                episode_steps=4,
                output_root=root,
            )
            self.assertEqual(len(rows), 6)
            for episode in (1, 2):
                paired_seeds = {row["seed"] for row in rows if row["episode"] == episode}
                self.assertEqual(len(paired_seeds), 1)
            saved_rows = read_csv(root / "results" / "evaluation_results.csv")
            self.assertEqual(set(saved_rows[0]), set(EVALUATION_FIELDS))
            self.assertEqual(len(summary), 3)
            self.assertTrue((root / "results" / "summary_results.csv").is_file())
            repeated_rows, _ = run_evaluation(
                episodes=2,
                seed=70,
                model_seed=31,
                scenario="low",
                episode_steps=4,
                output_root=root,
                make_plots=False,
            )
            self.assertEqual(rows, repeated_rows)

    def test_all_controllers_receive_the_same_environment_configuration(self):
        rates = (4.0, 3.5)
        env = make_environment(rates, episode_steps=7, interval_seconds=10, service_capacity=5)
        webster = WebsterController(arrival_rates=rates, interval_seconds=10)
        self.assertEqual(env.arrival_rates, (4.0, 3.5))
        self.assertEqual(env.episode_steps, 7)
        self.assertEqual(env.interval_seconds, 10)
        self.assertEqual(env.service_capacity, 5)
        self.assertEqual(webster.interval_seconds, env.interval_seconds)

    def test_summary_ignores_nan_and_preserves_undefined_wait(self):
        rows = [
            {"scenario": "low", "controller": "DQN", "avg_waiting_time_sec": "nan",
             "time_weighted_avg_queue": 2, "throughput_veh_per_min": 1,
             "total_departed": 5, "final_queue": 3, "cumulative_reward": -2},
            {"scenario": "low", "controller": "DQN", "avg_waiting_time_sec": "nan",
             "time_weighted_avg_queue": 4, "throughput_veh_per_min": 3,
             "total_departed": 7, "final_queue": 5, "cumulative_reward": -4},
        ]
        summary = aggregate_evaluation(rows)[0]
        self.assertTrue(math.isnan(summary["mean_avg_waiting_time_sec"]))
        self.assertTrue(math.isnan(summary["std_avg_waiting_time_sec"]))
        self.assertEqual(summary["valid_avg_waiting_time_observations"], 0)
        self.assertEqual(summary["mean_time_weighted_avg_queue"], 3)

    def test_plot_generation_works_from_csv_and_creates_six_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_training(episodes=1, seed=5, scenario="low", episode_steps=3, output_root=root)
            run_evaluation(
                episodes=1, seed=8, model_seed=5, scenario="low",
                episode_steps=3, output_root=root, make_plots=False,
            )
            paths = generate_plots(root)
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
