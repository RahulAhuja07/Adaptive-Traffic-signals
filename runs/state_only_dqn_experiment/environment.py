"""State-only DQN experiment: richer state features without changing traffic dynamics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPT_DIR]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import TrafficEnvironment


class RichStateTrafficEnvironment(TrafficEnvironment):
    """Return directional queue features while preserving baseline env dynamics.

    The underlying simulator is unchanged: arrival process, service capacity,
    episode duration, reward function, and action space remain exactly as in the
    original baseline environment. Only the observed state for the DQN is richer.

    Feature vector:
        [phase_ns, phase_ew, ns_queue_norm, ew_queue_norm, total_queue_norm, imbalance_norm]

    Normalization uses a queue scale of 20 vehicles, which is large enough to
    preserve overloaded-queue severity without clipping the raw congestion.
    """

    def __init__(
        self,
        arrival_rates: Tuple[float, float] = (2.8, 1.2),
        episode_steps: int = 180,
        interval_seconds: float = 10,
        service_capacity: int = 5,
        seed: int | None = None,
        reward_mode: str = "baseline",
        queue_scale: float = 20.0,
    ) -> None:
        if queue_scale <= 0 or not np.isfinite(queue_scale):
            raise ValueError("queue_scale must be a positive finite number")
        self.queue_scale = float(queue_scale)
        super().__init__(
            arrival_rates=arrival_rates,
            episode_steps=episode_steps,
            interval_seconds=interval_seconds,
            service_capacity=service_capacity,
            seed=seed,
            reward_mode=reward_mode,
        )

    def _rich_state(self) -> np.ndarray:
        ns_queue = len(self.queue_ns)
        ew_queue = len(self.queue_ew)
        phase_ns = 1.0 if self.current_phase == 0 else 0.0
        phase_ew = 1.0 if self.current_phase == 1 else 0.0
        total_queue = ns_queue + ew_queue
        imbalance = abs(ns_queue - ew_queue)
        features = np.array(
            [
                float(phase_ns),
                float(phase_ew),
                float(ns_queue / self.queue_scale),
                float(ew_queue / self.queue_scale),
                float(total_queue / (2.0 * self.queue_scale)),
                float(imbalance / self.queue_scale),
            ],
            dtype=np.float32,
        )
        return features

    def reset(self, seed: int | None = None) -> np.ndarray:
        super().reset(seed=seed)
        return self._rich_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        next_state_int, reward, done, info = super().step(action)
        rich_state = self._rich_state()
        info["discrete_state"] = int(next_state_int)
        info["phase_feature"] = int(self.current_phase)
        return rich_state, float(reward), bool(done), info
