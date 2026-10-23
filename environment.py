"""A small, reproducible two-direction traffic signal environment.

This module deliberately contains no controller or learning algorithm. Controllers
choose one of the two signal phases and interact through ``reset`` and ``step``.
"""

from collections import deque
import math
from numbers import Integral, Real
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class TrafficEnvironment:
    """Simulate NS/EW vehicle queues controlled by a two-phase traffic signal.

    ``current_phase`` is 0 for NS green and 1 for EW green. Each vehicle is
    represented by its arrival time in seconds. Departing vehicles are assigned
    the end time of the interval as their departure time.
    """

    def __init__(
        self,
        arrival_rates: Tuple[float, float] = (2.8, 1.2),
        episode_steps: int = 180,
        interval_seconds: float = 10,
        service_capacity: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        if len(arrival_rates) != 2:
            raise ValueError("arrival_rates must contain (NS rate, EW rate)")
        if any(
            not isinstance(rate, Real)
            or not math.isfinite(float(rate))
            or rate < 0
            for rate in arrival_rates
        ):
            raise ValueError("arrival rates must be finite, nonnegative numbers")
        if isinstance(episode_steps, bool) or not isinstance(episode_steps, Integral) or episode_steps <= 0:
            raise ValueError("episode_steps must be a positive integer")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, Real)
            or not math.isfinite(float(interval_seconds))
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be a finite positive number")
        if isinstance(service_capacity, bool) or not isinstance(service_capacity, Integral) or service_capacity < 0:
            raise ValueError("service_capacity must be a nonnegative integer")

        self.arrival_rates = (float(arrival_rates[0]), float(arrival_rates[1]))
        self.episode_steps = int(episode_steps)
        self.interval_seconds = float(interval_seconds)
        self.service_capacity = int(service_capacity)
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> int:
        """Start a fresh episode and return its initial discrete state.

        The initial phase is NS green (0). Supplying a seed restarts the random
        generator; omitting it preserves the generator's current random stream.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.queue_ns: Deque[float] = deque()
        self.queue_ew: Deque[float] = deque()
        self.current_phase = 0
        self.step_count = 0
        self.time_seconds = 0.0
        self.total_arrived = 0
        self.total_arrived_ns = 0
        self.total_arrived_ew = 0
        self.total_departed = 0
        self.total_departed_ns = 0
        self.total_departed_ew = 0
        self.cumulative_reward = 0.0
        self._waiting_times: List[float] = []
        self._queue_time_sum = 0.0
        return self._get_discrete_state()

    @staticmethod
    def _queue_bin(queue_length: int) -> int:
        """Map a queue length to low (0), medium (1), or high (2)."""
        if queue_length <= 4:
            return 0
        if queue_length <= 9:
            return 1
        return 2

    def _get_discrete_state(self) -> int:
        """Encode phase and queue bins as an integer in the range 0..17."""
        ns_bin = self._queue_bin(len(self.queue_ns))
        ew_bin = self._queue_bin(len(self.queue_ew))
        return int(self.current_phase * 9 + ns_bin * 3 + ew_bin)

    def _add_arrivals(self, queue: Deque[float], count: int, start: float) -> None:
        """Generate and append timestamped arrivals for one direction."""
        timestamps = self._rng.uniform(
            start, start + self.interval_seconds, size=count
        )
        for timestamp in sorted(float(value) for value in timestamps):
            queue.append(timestamp)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, object]]:
        """Run one interval and return ``(next_state, reward, done, info)``."""
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in (0, 1):
            raise ValueError("action must be 0 (NS green) or 1 (EW green)")
        if self.step_count >= self.episode_steps:
            raise RuntimeError("episode is finished; call reset() before stepping again")

        self.current_phase = int(action)
        queue_before = len(self.queue_ns) + len(self.queue_ew)
        interval_start = self.time_seconds
        interval_end = interval_start + self.interval_seconds

        arrivals_ns = int(self._rng.poisson(self.arrival_rates[0]))
        arrivals_ew = int(self._rng.poisson(self.arrival_rates[1]))
        self._add_arrivals(self.queue_ns, arrivals_ns, interval_start)
        self._add_arrivals(self.queue_ew, arrivals_ew, interval_start)
        queue_after_arrivals = len(self.queue_ns) + len(self.queue_ew)

        departures_ns = 0
        departures_ew = 0
        departed_waits: List[float] = []
        green_queue = self.queue_ns if self.current_phase == 0 else self.queue_ew
        served = min(self.service_capacity, len(green_queue))
        for _ in range(served):
            arrival_time = green_queue.popleft()
            waiting_time = max(0.0, interval_end - arrival_time)
            departed_waits.append(waiting_time)
        if self.current_phase == 0:
            departures_ns = served
        else:
            departures_ew = served

        self.time_seconds = interval_end
        self.step_count += 1
        queue_after = len(self.queue_ns) + len(self.queue_ew)
        # Reward uses average queue at the start and end of the interval. It is
        # a queue-length penalty, not the sum of individual vehicle waits.
        reward = -((queue_before + queue_after) / 2.0) / 10.0
        self.cumulative_reward += reward
        self._waiting_times.extend(departed_waits)
        self._queue_time_sum += queue_after * self.interval_seconds

        self.total_arrived_ns += arrivals_ns
        self.total_arrived_ew += arrivals_ew
        self.total_arrived += arrivals_ns + arrivals_ew
        self.total_departed_ns += departures_ns
        self.total_departed_ew += departures_ew
        self.total_departed += served

        done = self.step_count >= self.episode_steps
        info: Dict[str, object] = {
            "step": self.step_count,
            "time_seconds": self.time_seconds,
            "phase": self.current_phase,
            "queue_ns": len(self.queue_ns),
            "queue_ew": len(self.queue_ew),
            "queue_total": queue_after,
            "arrivals_ns": arrivals_ns,
            "arrivals_ew": arrivals_ew,
            "departures_ns": departures_ns,
            "departures_ew": departures_ew,
            "departed_this_step": served,
            "total_arrived": self.total_arrived,
            "total_departed": self.total_departed,
            "final_queue": queue_after,
            "queue_before": queue_before,
            "queue_after_arrivals": queue_after_arrivals,
            "queue_after": queue_after,
            "waiting_times_sec": tuple(departed_waits),
        }
        return self._get_discrete_state(), float(reward), done, info

    def get_metrics(self) -> Dict[str, float]:
        """Return cumulative episode metrics through the current simulation time."""
        avg_wait = (
            float(np.mean(self._waiting_times))
            if self._waiting_times
            else float("nan")
        )
        elapsed_minutes = self.time_seconds / 60.0
        throughput = (
            self.total_departed / elapsed_minutes if elapsed_minutes > 0 else 0.0
        )
        avg_queue = (
            self._queue_time_sum / self.time_seconds if self.time_seconds > 0 else 0.0
        )
        return {
            "avg_waiting_time_sec": avg_wait,
            "time_weighted_avg_queue": avg_queue,
            "throughput_veh_per_min": throughput,
            "total_arrived": float(self.total_arrived),
            "total_departed": float(self.total_departed),
            "final_queue": float(len(self.queue_ns) + len(self.queue_ew)),
            "cumulative_reward": float(self.cumulative_reward),
        }
