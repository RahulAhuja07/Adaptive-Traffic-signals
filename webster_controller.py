"""Simplified Webster-style fixed-time baseline for two traffic directions."""

import math
from numbers import Real
from typing import Dict, Optional, Sequence


class WebsterController:
    """Apply a fixed cycle computed from known synthetic arrival-rate inputs.

    Rates in the environment are expected vehicles per interval. They are
    converted to vehicles/hour, then divided by configurable saturation flows
    to form critical flow ratios. These assumptions are illustrative and are
    not calibrated traffic-engineering values.
    """

    def __init__(
        self,
        arrival_rates: Sequence[float] = (2.8, 1.2),
        interval_seconds: float = 10.0,
        saturation_flows: Sequence[float] = (1800.0, 1800.0),
        lost_time_seconds: float = 4.0,
        min_cycle_seconds: float = 30.0,
        max_cycle_seconds: float = 120.0,
    ) -> None:
        if not self._finite_positive(interval_seconds):
            raise ValueError("interval_seconds must be positive and finite")
        if len(saturation_flows) != 2 or any(not self._finite_positive(x) for x in saturation_flows):
            raise ValueError("saturation_flows must contain two positive finite values")
        if not isinstance(lost_time_seconds, Real) or not math.isfinite(float(lost_time_seconds)) or lost_time_seconds < 0:
            raise ValueError("lost_time_seconds must be finite and nonnegative")
        if (
            not self._finite_positive(min_cycle_seconds)
            or not self._finite_positive(max_cycle_seconds)
            or min_cycle_seconds > max_cycle_seconds
        ):
            raise ValueError("cycle limits must be positive and minimum <= maximum")
        self.interval_seconds = float(interval_seconds)
        self.saturation_flows = (float(saturation_flows[0]), float(saturation_flows[1]))
        self.lost_time_seconds = float(lost_time_seconds)
        self.min_cycle_seconds = float(min_cycle_seconds)
        self.max_cycle_seconds = float(max_cycle_seconds)
        self.timing_plan = self.calculate_timing(arrival_rates)
        self.reset()

    @staticmethod
    def _finite_positive(value: object) -> bool:
        return isinstance(value, Real) and math.isfinite(float(value)) and value > 0

    def calculate_timing(self, arrival_rates: Sequence[float]) -> Dict[str, float]:
        """Calculate a clipped cycle and interval-rounded phase durations.

        For total critical flow Y >= 1, Webster's formula is unsuitable. This
        baseline then uses the maximum configured cycle and allocates its green
        time in proportion to demand. With zero demand it uses an equal split.
        Each phase receives at least one simulation interval so the controller
        can always alternate safely.
        """
        if len(arrival_rates) != 2 or any(
            not isinstance(rate, Real)
            or not math.isfinite(float(rate))
            or rate < 0
            for rate in arrival_rates
        ):
            raise ValueError("arrival_rates must contain two finite nonnegative values")
        hourly_flows = [float(rate) / self.interval_seconds * 3600.0 for rate in arrival_rates]
        if any(not math.isfinite(flow) for flow in hourly_flows):
            raise ValueError("arrival rates are too large to convert to vehicles per hour")
        ratios = [hourly_flows[i] / self.saturation_flows[i] for i in range(2)]
        total_y = sum(ratios)
        if total_y >= 1.0:
            cycle = self.max_cycle_seconds
            green_total = max(2.0 * self.interval_seconds, cycle - self.lost_time_seconds)
            shares = [ratio / total_y for ratio in ratios]
        elif total_y == 0:
            cycle = self.min_cycle_seconds
            green_total = max(2.0 * self.interval_seconds, cycle - self.lost_time_seconds)
            shares = [0.5, 0.5]
        else:
            raw_cycle = (1.5 * self.lost_time_seconds + 5.0) / (1.0 - total_y)
            cycle = min(self.max_cycle_seconds, max(self.min_cycle_seconds, raw_cycle))
            green_total = max(2.0 * self.interval_seconds, cycle - self.lost_time_seconds)
            shares = [ratio / total_y for ratio in ratios]

        raw_green = [green_total * share for share in shares]
        # Round to interval counts, then guarantee at least one interval per phase.
        counts = [max(1, int(math.floor(seconds / self.interval_seconds + 0.5))) for seconds in raw_green]
        rounded_cycle = sum(counts) * self.interval_seconds + self.lost_time_seconds
        return {
            "cycle_length_seconds": float(cycle),
            "green_ns_seconds": counts[0] * self.interval_seconds,
            "green_ew_seconds": counts[1] * self.interval_seconds,
            "green_ns_intervals": float(counts[0]),
            "green_ew_intervals": float(counts[1]),
            "rounded_cycle_seconds": rounded_cycle,
            "critical_flow_ratio_ns": ratios[0],
            "critical_flow_ratio_ew": ratios[1],
        }

    def reset(self) -> None:
        """Start a cycle at NS and reset its interval counter."""
        self.current_phase = 0
        self._intervals_in_phase = 0

    def choose_action(self, info: Optional[dict] = None) -> int:
        """Return a phase and advance the deterministic timing-plan counter."""
        del info  # The known demand plan does not need hidden environment state.
        action = self.current_phase
        self._intervals_in_phase += 1
        phase_key = "green_ns_intervals" if self.current_phase == 0 else "green_ew_intervals"
        phase_duration = int(self.timing_plan[phase_key])
        if self._intervals_in_phase >= phase_duration:
            self.current_phase = 1 - self.current_phase
            self._intervals_in_phase = 0
        return action
