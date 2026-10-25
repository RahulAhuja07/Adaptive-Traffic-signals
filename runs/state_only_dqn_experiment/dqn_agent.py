"""DQN for a richer directional state vector without changing baseline reward logic."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class RichStateDQN(nn.Module):
    def __init__(self, state_size: int = 6, action_size: int = 2) -> None:
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


class StateOnlyDQNAgent:
    """DQN using directional queue features instead of a coarse one-hot state."""

    def __init__(
        self,
        state_size: int = 6,
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
        if state_size <= 0 or action_size != 2:
            raise ValueError("state_size must be positive and action_size must be 2")
        if learning_rate <= 0 or not 0 <= gamma <= 1:
            raise ValueError("learning_rate must be positive and gamma in [0, 1]")
        if not 0 <= epsilon_min <= epsilon <= 1 or not 0 < epsilon_decay <= 1:
            raise ValueError("invalid epsilon configuration")
        if replay_capacity <= 0 or batch_size <= 0 or target_update_frequency <= 0:
            raise ValueError("replay capacity, batch size, and target frequency must be positive")
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.target_update_frequency = int(target_update_frequency)
        self.device = torch.device(device or "cpu")
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.online_network = RichStateDQN(self.state_size, self.action_size).to(self.device)
        self.target_network = RichStateDQN(self.state_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer: Deque[Transition] = deque(maxlen=self.replay_capacity)
        self.learning_steps = 0

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_size,):
            raise ValueError(f"state must have shape ({self.state_size},)")
        if training and float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(self.action_size))
        encoded = torch.tensor(state_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            values = self.online_network(encoded)[0].cpu().numpy()
        best_actions = np.flatnonzero(values == values.max())
        return int(self._rng.choice(best_actions))

    def remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        next_array = np.asarray(next_state, dtype=np.float32)
        if state_array.shape != (self.state_size,) or next_array.shape != (self.state_size,):
            raise ValueError(f"state vectors must have shape ({self.state_size},)")
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.replay_buffer.append((state_array.copy(), int(action), float(reward), next_array.copy(), bool(done)))

    def _calculate_targets(
        self, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            future = self.target_network(next_states).max(dim=1).values
            return rewards + self.gamma * future * (1.0 - dones)

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        sample_items = list(self.replay_buffer)
        indices = self._rng.choice(len(sample_items), size=self.batch_size, replace=False)
        batch = [sample_items[i] for i in indices]
        states = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=self.device)

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
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> Optional[float]:
        self.remember(state, action, reward, next_state, done)
        return self.learn()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
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
        try:
            checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(Path(path), map_location=self.device)
        config = checkpoint.get("config", {})
        if config.get("state_size", self.state_size) != self.state_size or config.get("action_size", self.action_size) != self.action_size:
            raise ValueError("saved model dimensions do not match this agent")
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))
        self.gamma = float(config.get("gamma", self.gamma))
        self.epsilon_min = float(config.get("epsilon_min", self.epsilon_min))
        self.epsilon_decay = float(config.get("epsilon_decay", self.epsilon_decay))
        self.replay_capacity = int(config.get("replay_capacity", self.replay_capacity))
        self.batch_size = int(config.get("batch_size", self.batch_size))
        self.target_update_frequency = int(config.get("target_update_frequency", self.target_update_frequency))
        self.replay_buffer = deque(maxlen=self.replay_capacity)
        self.online_network.load_state_dict(checkpoint["online_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint["epsilon"])
        self.learning_steps = int(checkpoint["learning_steps"])
