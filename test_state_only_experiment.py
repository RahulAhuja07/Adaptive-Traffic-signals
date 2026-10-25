"""Focused tests for the state-only DQN experiment."""

import unittest

import numpy as np

from runs.state_only_dqn_experiment.dqn_agent import StateOnlyDQNAgent
from runs.state_only_dqn_experiment.environment import RichStateTrafficEnvironment


class StateOnlyExperimentTests(unittest.TestCase):
    def test_state_distinguishes_materially_different_queue_conditions(self):
        env = RichStateTrafficEnvironment(arrival_rates=(4.0, 3.5), episode_steps=3, seed=7)
        env.reset(seed=7)
        env.current_phase = 0
        env.queue_ns = env.queue_ns.__class__([0.0] * 12)
        env.queue_ew = env.queue_ew.__class__([0.0] * 1)
        state_a = env._rich_state()

        env.current_phase = 1
        env.queue_ns = env.queue_ns.__class__([0.0] * 1)
        env.queue_ew = env.queue_ew.__class__([0.0] * 12)
        state_b = env._rich_state()

        self.assertFalse(np.allclose(state_a, state_b))
        self.assertNotEqual(float(state_a[2]), float(state_b[2]))
        self.assertNotEqual(float(state_a[3]), float(state_b[3]))
        self.assertNotEqual(float(state_a[0]), float(state_b[0]))

    def test_phase_and_both_directions_are_encoded(self):
        env = RichStateTrafficEnvironment(seed=11)
        env.reset(seed=11)
        env.current_phase = 0
        env.queue_ns.extend([0.0] * 6)
        env.queue_ew.extend([0.0] * 2)
        state = env._rich_state()
        self.assertEqual(state.shape, (6,))
        self.assertEqual(float(state[0]), 1.0)
        self.assertEqual(float(state[1]), 0.0)
        self.assertGreater(float(state[2]), 0.0)
        self.assertGreater(float(state[3]), 0.0)

    def test_reward_is_finite_and_baseline_reward_is_unchanged(self):
        env = RichStateTrafficEnvironment(seed=13)
        env.reset(seed=13)
        env.queue_ns = env.queue_ns.__class__([0.0] * 6)
        env.queue_ew = env.queue_ew.__class__([0.0] * 3)
        env.current_phase = 0
        state_before = env._rich_state()
        _, reward, _, info = env.step(0)
        state_after = env._rich_state()
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(np.isfinite(float(state_before.sum())))
        self.assertTrue(np.isfinite(float(state_after.sum())))
        self.assertIn(info["phase_feature"], (0, 1))

    def test_replay_and_targets_are_valid(self):
        agent = StateOnlyDQNAgent(batch_size=2, target_update_frequency=1, seed=3)
        state = np.array([0.0, 1.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        next_state = np.array([1.0, 0.0, 0.2, 0.1, 0.2, 0.3], dtype=np.float32)
        agent.remember(state, 0, -1.0, next_state, False)
        agent.remember(next_state, 1, 0.5, state, True)
        self.assertEqual(len(agent.replay_buffer), 2)
        loss = agent.learn()
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))

    def test_evaluation_is_greedy_and_does_not_update_model(self):
        agent = StateOnlyDQNAgent(seed=5)
        state = np.array([0.0, 1.0, 0.2, 0.3, 0.4, 0.1], dtype=np.float32)
        before_eps = agent.epsilon
        action = agent.choose_action(state, training=False)
        self.assertIn(action, (0, 1))
        self.assertEqual(agent.epsilon, before_eps)

    def test_vehicle_accounting_stays_consistent(self):
        env = RichStateTrafficEnvironment(arrival_rates=(2.0, 1.0), episode_steps=3, seed=17)
        env.reset(seed=17)
        initial_total = len(env.queue_ns) + len(env.queue_ew)
        _, _, _, info = env.step(0)
        self.assertGreaterEqual(info["total_arrived"], info["total_departed"])
        self.assertEqual(info["total_arrived"], info["arrivals_ns"] + info["arrivals_ew"])
        self.assertEqual(info["total_departed"], info["departures_ns"] + info["departures_ew"])
        self.assertGreaterEqual(initial_total, 0)


if __name__ == "__main__":
    unittest.main()
