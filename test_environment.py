"""Unit tests for the custom traffic simulation environment."""

import math
import unittest

from environment import TrafficEnvironment


class TrafficEnvironmentTests(unittest.TestCase):
    def test_initialization(self):
        env = TrafficEnvironment(seed=1)
        self.assertEqual(env.arrival_rates, (2.8, 1.2))
        self.assertEqual(env.episode_steps, 180)
        self.assertEqual(env.interval_seconds, 10.0)
        self.assertEqual(env.service_capacity, 5)
        self.assertEqual(env.current_phase, 0)
        self.assertEqual(env._get_discrete_state(), 0)

    def test_reset_empties_queues_and_counters(self):
        env = TrafficEnvironment(seed=4)
        env.step(0)
        self.assertEqual(env.reset(seed=4), 0)
        self.assertEqual(len(env.queue_ns), 0)
        self.assertEqual(len(env.queue_ew), 0)
        self.assertEqual(env.step_count, 0)
        self.assertEqual(env.time_seconds, 0)
        self.assertEqual(env.total_arrived, 0)
        self.assertEqual(env.total_departed, 0)
        self.assertEqual(env.cumulative_reward, 0)

    def test_reproducible_with_same_seed_and_actions(self):
        actions = [0, 1, 1, 0, 1, 0]
        outcomes = []
        for _ in range(2):
            env = TrafficEnvironment(episode_steps=10)
            env.reset(seed=88)
            outcomes.append([env.step(action) for action in actions])
        self.assertEqual(outcomes[0], outcomes[1])

    def test_actions_select_signal_phase(self):
        env = TrafficEnvironment(seed=2)
        _, _, _, ns_info = env.step(0)
        self.assertEqual(ns_info["phase"], 0)
        _, _, _, ew_info = env.step(1)
        self.assertEqual(ew_info["phase"], 1)

    def test_invalid_actions_are_rejected(self):
        env = TrafficEnvironment(seed=1)
        for action in (-1, 2, "0", True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)

    def test_episode_ends_after_configured_calls(self):
        env = TrafficEnvironment(episode_steps=3, seed=1)
        self.assertFalse(env.step(0)[2])
        self.assertFalse(env.step(1)[2])
        self.assertTrue(env.step(0)[2])

    def test_states_stay_in_range(self):
        env = TrafficEnvironment(episode_steps=30, seed=5)
        self.assertIn(env.reset(), range(18))
        for step in range(30):
            state, _, _, _ = env.step(step % 2)
            self.assertIn(state, range(18))

    def test_queue_lengths_never_negative(self):
        env = TrafficEnvironment(episode_steps=20, seed=9)
        for step in range(20):
            _, _, _, _ = env.step(step % 2)
            self.assertGreaterEqual(len(env.queue_ns), 0)
            self.assertGreaterEqual(len(env.queue_ew), 0)

    def test_only_green_direction_has_departures(self):
        env = TrafficEnvironment(arrival_rates=(8, 8), service_capacity=5, seed=7)
        for action in (0, 1, 0, 1):
            _, _, _, info = env.step(action)
            if action == 0:
                self.assertEqual(info["departures_ew"], 0)
            else:
                self.assertEqual(info["departures_ns"], 0)

    def test_departures_do_not_exceed_capacity(self):
        env = TrafficEnvironment(service_capacity=2, seed=7)
        for action in (0, 1, 0, 1, 0):
            _, _, _, info = env.step(action)
            self.assertLessEqual(info["departed_this_step"], 2)

    def test_traffic_accounting(self):
        env = TrafficEnvironment(episode_steps=25, seed=19)
        for step in range(25):
            _, _, _, info = env.step(step % 2)
            self.assertEqual(
                info["total_arrived"], info["total_departed"] + info["final_queue"]
            )

    def test_reward_formula(self):
        env = TrafficEnvironment(episode_steps=2, seed=13)
        queue_before = len(env.queue_ns) + len(env.queue_ew)
        _, reward, _, info = env.step(0)
        expected = -((queue_before + info["queue_after"]) / 2.0) / 10.0
        self.assertAlmostEqual(reward, expected)
        self.assertTrue(math.isfinite(reward))

    def test_waiting_times_are_nonnegative(self):
        env = TrafficEnvironment(arrival_rates=(8, 8), seed=21)
        for action in (0, 1, 0):
            _, _, _, info = env.step(action)
            self.assertTrue(all(wait >= 0 for wait in info["waiting_times_sec"]))

    def test_metrics_have_required_keys(self):
        env = TrafficEnvironment(seed=1)
        self.assertEqual(
            set(env.get_metrics()),
            {
                "avg_waiting_time_sec", "time_weighted_avg_queue",
                "throughput_veh_per_min", "total_arrived", "total_departed",
                "final_queue", "cumulative_reward",
            },
        )

    def test_average_wait_is_nan_without_departures(self):
        env = TrafficEnvironment(arrival_rates=(0, 0), seed=1)
        self.assertTrue(math.isnan(env.get_metrics()["avg_waiting_time_sec"]))

    def test_zero_elapsed_time_throughput_is_safe(self):
        env = TrafficEnvironment(seed=1)
        self.assertEqual(env.get_metrics()["throughput_veh_per_min"], 0.0)

    def test_configuration_validation(self):
        for kwargs in (
            {"arrival_rates": (-1, 1)},
            {"arrival_rates": (1,)},
            {"episode_steps": 0},
            {"interval_seconds": 0},
            {"service_capacity": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrafficEnvironment(**kwargs)


if __name__ == "__main__":
    unittest.main()
