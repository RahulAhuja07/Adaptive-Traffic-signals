"""Deterministic unit tests for all Part 2 traffic signal controllers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dqn_agent import DQNAgent, QNetwork
from environment import TrafficEnvironment
from q_learning import QLearningAgent
from webster_controller import WebsterController


class QLearningTests(unittest.TestCase):
    def test_table_shape_and_zero_initialization(self):
        agent = QLearningAgent(seed=1)
        self.assertEqual(agent.q_table.shape, (18, 2))
        np.testing.assert_array_equal(agent.q_table, np.zeros((18, 2)))

    def test_actions_are_valid_and_greedy_chooses_best_value(self):
        agent = QLearningAgent(epsilon=0, epsilon_min=0, seed=1)
        agent.q_table[3] = [1.0, 2.0]
        self.assertEqual(agent.choose_action(3), 1)
        agent.epsilon = 1
        self.assertIn(agent.choose_action(3), (0, 1))

    def test_tie_breaking_returns_valid_actions(self):
        agent = QLearningAgent(seed=11)
        self.assertTrue(all(agent.choose_action(0, training=False) in (0, 1) for _ in range(50)))

    def test_q_learning_update_equation(self):
        agent = QLearningAgent(alpha=0.1, gamma=0.95)
        agent.q_table[1, 0] = 2.0
        agent.q_table[2] = [3.0, 4.0]
        agent.update(1, 0, 1.0, 2, False)
        self.assertAlmostEqual(agent.q_table[1, 0], 2.28)

    def test_terminal_update_uses_only_immediate_reward(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.95)
        agent.q_table[2] = [100, 90]
        agent.update(1, 0, 4.0, 2, True)
        self.assertAlmostEqual(agent.q_table[1, 0], 2.0)

    def test_epsilon_decay_has_minimum(self):
        agent = QLearningAgent(epsilon=0.1, epsilon_min=0.05, epsilon_decay=0.5)
        agent.decay_epsilon()
        self.assertEqual(agent.epsilon, 0.05)
        agent.decay_epsilon()
        self.assertEqual(agent.epsilon, 0.05)

    def test_save_and_load_preserves_table_and_configuration(self):
        agent = QLearningAgent(seed=1)
        agent.q_table[4] = [2.5, -1.0]
        agent.decay_epsilon()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q_table.npz"
            agent.save(str(path))
            restored = QLearningAgent()
            restored.load(str(path))
        np.testing.assert_array_equal(restored.q_table, agent.q_table)
        self.assertEqual(restored.epsilon, agent.epsilon)


class DQNTests(unittest.TestCase):
    def test_network_batch_output_shape(self):
        network = QNetwork()
        output = network(torch.zeros((4, 18), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (4, 2))

    def test_one_hot_state_encoding(self):
        encoded = DQNAgent.encode_state(7)
        self.assertEqual(encoded.shape, (18,))
        self.assertEqual(float(encoded.sum()), 1.0)
        self.assertEqual(float(encoded[7]), 1.0)
        self.assertEqual(int(np.count_nonzero(encoded)), 1)

    def test_action_selection_is_valid(self):
        agent = DQNAgent(seed=4)
        self.assertIn(agent.choose_action(0), (0, 1))
        agent.epsilon = 0
        self.assertIn(agent.choose_action(17, training=False), (0, 1))

    def test_replay_buffer_stores_and_samples_transitions(self):
        agent = DQNAgent(batch_size=2, seed=1)
        agent.remember(0, 1, -1.0, 3, False)
        agent.remember(3, 0, 0.0, 2, True)
        self.assertEqual(len(agent.replay_buffer), 2)
        sample = agent._rng.sample(list(agent.replay_buffer), 2)
        self.assertEqual(len(sample), 2)
        self.assertTrue(all(len(transition) == 5 for transition in sample))

    def test_learning_skips_when_batch_is_incomplete(self):
        agent = DQNAgent(batch_size=2)
        agent.remember(0, 0, 0, 1, False)
        self.assertIsNone(agent.learn())

    def test_learning_update_runs_with_full_batch(self):
        agent = DQNAgent(batch_size=4, target_update_frequency=2, seed=3)
        for index in range(4):
            agent.remember(index, index % 2, -0.5, index + 1, index == 3)
        loss = agent.learn()
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.learning_steps, 1)

    def test_terminal_target_has_no_future_value(self):
        agent = DQNAgent(gamma=0.95)
        with torch.no_grad():
            for parameter in agent.target_network.parameters():
                parameter.zero_()
            agent.target_network.layers[-1].bias.copy_(torch.tensor([2.0, 4.0]))
        rewards = torch.tensor([1.0, 1.0])
        states = torch.zeros((2, 18))
        dones = torch.tensor([1.0, 0.0])
        targets = agent._calculate_targets(rewards, states, dones)
        torch.testing.assert_close(targets, torch.tensor([1.0, 4.8]))

    def test_target_network_starts_equal_to_online_network(self):
        agent = DQNAgent(seed=2)
        for online, target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        ):
            torch.testing.assert_close(online, target)

    def test_target_network_updates_on_configured_schedule(self):
        agent = DQNAgent(batch_size=1, target_update_frequency=1, seed=5)
        agent.remember(0, 0, -1.0, 1, False)
        agent.learn()
        for online, target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        ):
            torch.testing.assert_close(online, target)

    def test_save_and_load_preserves_usable_weights(self):
        agent = DQNAgent(gamma=0.8, epsilon_decay=0.9, batch_size=8, seed=3)
        agent.epsilon = 0.23
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dqn.pt"
            agent.save(str(path))
            restored = DQNAgent(seed=8)
            restored.load(str(path))
        self.assertEqual(restored.epsilon, 0.23)
        self.assertEqual(restored.gamma, 0.8)
        self.assertEqual(restored.epsilon_decay, 0.9)
        self.assertEqual(restored.batch_size, 8)
        for online, target in zip(
            agent.online_network.parameters(), restored.online_network.parameters()
        ):
            torch.testing.assert_close(online, target)
        self.assertIn(restored.choose_action(2, training=False), (0, 1))


class WebsterTests(unittest.TestCase):
    def test_valid_demand_produces_timing_plan(self):
        controller = WebsterController()
        plan = controller.timing_plan
        self.assertGreaterEqual(plan["cycle_length_seconds"], 30)
        self.assertLessEqual(plan["cycle_length_seconds"], 120)
        self.assertGreater(plan["green_ns_seconds"], 0)
        self.assertGreater(plan["green_ew_seconds"], 0)

    def test_zero_demand_uses_safe_equal_split(self):
        controller = WebsterController(arrival_rates=(0, 0))
        self.assertEqual(controller.timing_plan["green_ns_intervals"], 1)
        self.assertEqual(controller.timing_plan["green_ew_intervals"], 1)

    def test_negative_demand_is_rejected(self):
        controller = WebsterController()
        with self.assertRaises(ValueError):
            controller.calculate_timing((-1, 2))

    def test_green_time_is_positive_and_rounded_to_intervals(self):
        controller = WebsterController(interval_seconds=10)
        plan = controller.calculate_timing((1.0, 1.0))
        self.assertGreaterEqual(plan["green_ns_intervals"], 1)
        self.assertGreaterEqual(plan["green_ew_intervals"], 1)
        self.assertEqual(plan["green_ns_seconds"] % 10, 0)
        self.assertEqual(plan["green_ew_seconds"] % 10, 0)

    def test_controller_actions_follow_timing_split(self):
        controller = WebsterController()
        ns_intervals = int(controller.timing_plan["green_ns_intervals"])
        self.assertEqual([controller.choose_action() for _ in range(ns_intervals)], [0] * ns_intervals)
        self.assertEqual(controller.choose_action(), 1)

    def test_reset_starts_new_cycle_at_ns(self):
        controller = WebsterController()
        for _ in range(int(controller.timing_plan["green_ns_intervals"])):
            controller.choose_action()
        self.assertEqual(controller.choose_action(), 1)
        controller.reset()
        self.assertEqual(controller.choose_action(), 0)
        self.assertEqual(controller.current_phase, 0)

    def test_high_demand_uses_max_cycle_safeguard(self):
        controller = WebsterController()
        plan = controller.calculate_timing((100.0, 100.0))
        self.assertEqual(plan["cycle_length_seconds"], controller.max_cycle_seconds)
        self.assertGreater(plan["green_ns_seconds"], 0)
        self.assertGreater(plan["green_ew_seconds"], 0)

    def test_only_binary_actions_are_returned(self):
        controller = WebsterController()
        self.assertTrue(all(controller.choose_action() in (0, 1) for _ in range(20)))

    def test_all_controllers_use_environment_interface(self):
        env = TrafficEnvironment(episode_steps=3, seed=17)
        controllers = [
            QLearningAgent(seed=1),
            DQNAgent(batch_size=2, seed=1),
            WebsterController(arrival_rates=env.arrival_rates,
                              interval_seconds=env.interval_seconds),
        ]
        for controller in controllers:
            state = env.reset(seed=17)
            controller.reset() if isinstance(controller, WebsterController) else None
            done = False
            while not done:
                action = controller.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                if isinstance(controller, (QLearningAgent, DQNAgent)):
                    controller.update(state, action, reward, next_state, done)
                state = next_state
            self.assertEqual(env.step_count, 3)


if __name__ == "__main__":
    unittest.main()
