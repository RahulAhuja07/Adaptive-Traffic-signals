"""Tabular Q-learning controller for the 18-state traffic environment."""

from pathlib import Path
from typing import Optional

import numpy as np


class QLearningAgent:
    """Learn a value for each state/action pair using an epsilon-greedy policy."""

    def __init__(
        self,
        state_size: int = 18,
        action_size: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        if state_size <= 0 or action_size != 2:
            raise ValueError("state_size must be positive and action_size must be 2")
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("alpha must be in (0, 1] and gamma in [0, 1]")
        if not 0 <= epsilon_min <= epsilon <= 1 or not 0 < epsilon_decay <= 1:
            raise ValueError("invalid epsilon configuration")
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.q_table = np.zeros((self.state_size, self.action_size), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def _check_state(self, state: int) -> int:
        if isinstance(state, bool) or not isinstance(state, (int, np.integer)):
            raise ValueError("state must be an integer")
        if not 0 <= int(state) < self.state_size:
            raise ValueError("state is outside the configured state space")
        return int(state)

    def choose_action(self, state: int, training: bool = True) -> int:
        """Choose an action, exploring only while training."""
        state = self._check_state(state)
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.action_size))
        values = self.q_table[state]
        best_actions = np.flatnonzero(values == np.max(values))
        return int(self._rng.choice(best_actions))

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """Apply one standard Q-learning update."""
        state = self._check_state(state)
        next_state = self._check_state(next_state)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < self.action_size:
            raise ValueError("action is outside the configured action space")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        target = float(reward)
        if not done:
            target += self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state, int(action)] += self.alpha * (
            target - self.q_table[state, int(action)]
        )

    def decay_epsilon(self) -> None:
        """Decay exploration once after a training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save the Q-table and learning configuration in a NumPy archive."""
        np.savez(
            path,
            q_table=self.q_table,
            state_size=self.state_size,
            action_size=self.action_size,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
        )

    def load(self, path: str) -> None:
        """Load a compatible Q-table and its saved exploration settings."""
        with np.load(Path(path), allow_pickle=False) as data:
            table = np.asarray(data["q_table"], dtype=np.float64)
            if table.shape != (self.state_size, self.action_size):
                raise ValueError(
                    f"saved Q-table has shape {table.shape}; expected "
                    f"({self.state_size}, {self.action_size})"
                )
            self.q_table = table.copy()
            self.alpha = float(data["alpha"])
            self.gamma = float(data["gamma"])
            self.epsilon = float(data["epsilon"])
            self.epsilon_min = float(data["epsilon_min"])
            self.epsilon_decay = float(data["epsilon_decay"])
