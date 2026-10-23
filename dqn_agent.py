"""A compact PyTorch DQN for the custom 18-state traffic environment."""

from collections import deque
from pathlib import Path
import random
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


Transition = Tuple[int, int, float, int, bool]


class QNetwork(nn.Module):
    """Small feed-forward network mapping one-hot states to two Q-values."""

    def __init__(self, state_size: int = 18, action_size: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class DQNAgent:
    """Deep Q-learning with a replay buffer and a periodically copied target."""

    def __init__(
        self,
        state_size: int = 18,
        action_size: int = 2,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        target_update_frequency: int = 100,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if state_size != 18 or action_size != 2:
            raise ValueError("this environment uses 18 states and 2 actions")
        if learning_rate <= 0 or not 0 <= gamma <= 1:
            raise ValueError("learning_rate must be positive and gamma in [0, 1]")
        if not 0 <= epsilon_min <= epsilon <= 1 or not 0 < epsilon_decay <= 1:
            raise ValueError("invalid epsilon configuration")
        if replay_capacity <= 0 or batch_size <= 0 or target_update_frequency <= 0:
            raise ValueError("replay capacity, batch size, and target frequency must be positive")
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.target_update_frequency = int(target_update_frequency)
        self.device = torch.device(device or "cpu")
        self._rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.online_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.learning_rate
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer: Deque[Transition] = deque(maxlen=self.replay_capacity)
        self.learning_steps = 0

    @staticmethod
    def encode_state(state: int) -> np.ndarray:
        """Represent a discrete state using the same information as tabular Q."""
        if isinstance(state, bool) or not isinstance(state, (int, np.integer)) or not 0 <= state < 18:
            raise ValueError("state must be an integer from 0 through 17")
        encoded = np.zeros(18, dtype=np.float32)
        encoded[int(state)] = 1.0
        return encoded

    def choose_action(self, state: int, training: bool = True) -> int:
        """Choose an epsilon-greedy action for an integer environment state."""
        if training and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.action_size)
        encoded = torch.tensor(
            self.encode_state(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            values = self.online_network(encoded)[0].cpu().numpy()
        best_actions = np.flatnonzero(values == values.max())
        return int(self._rng.choice(best_actions.tolist()))

    def remember(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """Add one environment transition to the bounded replay buffer."""
        self.encode_state(state)
        self.encode_state(next_state)
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.replay_buffer.append(
            (int(state), int(action), float(reward), int(next_state), bool(done))
        )

    def _calculate_targets(
        self, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Bellman targets; terminal transitions have no future value."""
        with torch.no_grad():
            future = self.target_network(next_states).max(dim=1).values
            return rewards + self.gamma * future * (1.0 - dones)

    def learn(self) -> Optional[float]:
        """Run one replay update, or return None until a full batch is available."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self._rng.sample(list(self.replay_buffer), self.batch_size)
        states = torch.tensor(
            np.stack([self.encode_state(item[0]) for item in batch]),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.tensor(
            [item[1] for item in batch], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [item[2] for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.stack([self.encode_state(item[3]) for item in batch]),
            dtype=torch.float32, device=self.device,
        )
        dones = torch.tensor(
            [item[4] for item in batch], dtype=torch.float32, device=self.device
        )

        predicted = self.online_network(states).gather(1, actions).squeeze(1)
        targets = self._calculate_targets(rewards, next_states, dones)
        loss = self.loss_function(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learning_steps += 1
        if self.learning_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss.item())

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> Optional[float]:
        """Store a transition and attempt a replay learning step."""
        self.remember(state, action, reward, next_state, done)
        return self.learn()

    def decay_epsilon(self) -> None:
        """Decay exploration once at the end of a training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save online/target weights, optimizer, exploration, and configuration."""
        torch.save(
            {
                "online_state_dict": self.online_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learning_steps": self.learning_steps,
                "config": {
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "replay_capacity": self.replay_capacity,
                    "batch_size": self.batch_size,
                    "target_update_frequency": self.target_update_frequency,
                },
            },
            Path(path),
        )

    def load(self, path: str) -> None:
        """Load weights and training settings from a saved checkpoint."""
        try:
            checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        except TypeError:  # Compatibility with older supported PyTorch versions.
            checkpoint = torch.load(Path(path), map_location=self.device)
        config = checkpoint.get("config", {})
        if config.get("state_size", 18) != self.state_size or config.get("action_size", 2) != self.action_size:
            raise ValueError("saved model dimensions do not match this agent")
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))
        self.gamma = float(config.get("gamma", self.gamma))
        self.epsilon_min = float(config.get("epsilon_min", self.epsilon_min))
        self.epsilon_decay = float(config.get("epsilon_decay", self.epsilon_decay))
        self.replay_capacity = int(config.get("replay_capacity", self.replay_capacity))
        self.batch_size = int(config.get("batch_size", self.batch_size))
        self.target_update_frequency = int(
            config.get("target_update_frequency", self.target_update_frequency)
        )
        self.replay_buffer = deque(maxlen=self.replay_capacity)
        self.online_network.load_state_dict(checkpoint["online_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint["epsilon"])
        self.learning_steps = int(checkpoint["learning_steps"])
