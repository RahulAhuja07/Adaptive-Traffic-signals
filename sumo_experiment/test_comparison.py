"""Fast comparison tests using fake controllers and no SUMO process."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sumo_experiment import comparison_runner as comparison


class FakeAgent:
    def choose_action(self, state, training=False):
        if training:
            raise AssertionError("comparison must be greedy")
        return 0


class FakeWebster:
    def __init__(self, env, **kwargs):
        del env, kwargs

    def reset(self):
        pass

    def choose_action(self):
        return 0


class FakeEnv:
    def __init__(self, episode_horizon_seconds, decision_interval_seconds):
        self.horizon = episode_horizon_seconds
        self.interval = decision_interval_seconds
        self.closed = False
        self.steps = 0

    def reset(self, seed=None):
        self.seed = seed
        return 0

    def get_queue_observation(self):
        return {"phase": 0, "queue_ns": 2, "queue_ew": 3}

    def step(self, action):
        if action not in (0, 1):
            raise ValueError(action)
        self.steps += 1
        return 0, -0.2, True, {"termination_reason": "horizon"}

    def get_metrics(self):
        return {
            "demand_completed": False, "remaining_expected_vehicles": 4,
            "sumo_inserted_vehicles": 8, "time_weighted_avg_queue": 5.0,
            "total_stopped_waiting_time_sec_completed": 10.0,
            "avg_stopped_waiting_time_sec": 2.0,
            "throughput_veh_per_min": 1.5, "completed_vehicles": 3,
            "simulated_time_sec": self.interval, "final_queue": 4.0,
        }

    def close(self):
        self.closed = True


class ComparisonTests(unittest.TestCase):
    def _checkpoint(self, path: Path, episodes: int = 200) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake checkpoint")
        (path.parent / "training_config.json").write_text(
            json.dumps({"episodes": episodes, "seed": 7}), encoding="utf-8")
        return path

    def test_smoke_checkpoints_are_refused_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            q = self._checkpoint(base / "q" / "q.npz", 3)
            d = self._checkpoint(base / "d" / "d.pt", 2)
            s = self._checkpoint(base / "s" / "s.pt", 2)
            with self.assertRaisesRegex(ValueError, "refusing smoke/unverified"):
                comparison.run_comparison([q], [d], [s], seeds=[1], output_dir=base / "out")

    def test_paired_comparison_writes_metrics_config_plots_and_closes_envs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            q = self._checkpoint(base / "q" / "q.npz")
            d = self._checkpoint(base / "d" / "d.pt")
            s = self._checkpoint(base / "s" / "s.pt")
            made = []

            def factory(**kwargs):
                env = FakeEnv(**kwargs)
                made.append(env)
                return env

            with patch.object(comparison, "_new_agent", side_effect=lambda *args: FakeAgent()):
                rows, summaries = comparison.run_comparison(
                    [q], [d], [s], seeds=[11, 12], horizon=20, interval=10,
                    output_dir=base / "out", env_factory=factory,
                    controller_factory=FakeWebster,
                )
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(summaries), 4)
            self.assertTrue(all(env.closed for env in made))
            self.assertTrue(all(row["paired_inserted_count_match"] for row in rows))
            self.assertEqual({row["evaluation_seed"] for row in rows}, {11, 12})
            output = base / "out"
            for filename in ("episode_metrics.csv", "summary.csv", "comparison_config.json",
                             "comparison_report.md", "time_weighted_avg_queue_vehicles.png"):
                self.assertTrue((output / filename).is_file(), filename)
            config = json.loads((output / "comparison_config.json").read_text())
            self.assertEqual(config["label"], "candidate_comparison")
            self.assertEqual(config["evaluation_seeds"], [11, 12])

    def test_environment_closes_on_policy_exception(self):
        made = []

        def factory(**kwargs):
            env = FakeEnv(**kwargs)
            made.append(env)
            return env

        with patch.object(comparison, "_new_agent", side_effect=RuntimeError("load failure")):
            with self.assertRaisesRegex(RuntimeError, "load failure"):
                comparison._evaluate_one("q_learning", Path("checkpoint"), 1, 20, 10,
                                         factory, FakeWebster, {})
        self.assertTrue(made[0].closed)


if __name__ == "__main__":
    unittest.main()
