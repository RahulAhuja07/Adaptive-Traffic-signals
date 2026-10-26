"""Adapt the existing Webster timing controller to A1's SUMO actions.

The original controller calculates a cycle plan from configured nominal demand
rates. This adapter supplies rates derived from the route XML (or explicit
overrides), keeps its computed green-interval schedule, and returns its binary
NS/EW actions unchanged. ``SumoTrafficEnvironment.step`` performs the actual
3-second yellow clearance when the selected direction changes.
"""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Optional, Sequence
import xml.etree.ElementTree as ET

from webster_controller import WebsterController


HERE = Path(__file__).resolve().parent
ROUTES_FILE = HERE / "rl_routes.rou.xml"


def rates_from_route_flows(decision_interval_seconds: float,
                           routes_file: Path | str = ROUTES_FILE) -> tuple[float, float]:
    """Derive nominal per-decision arrivals from the XML's periodic flows.

    This uses configured route schedules, not observed/measured traffic counts.
    The current file has one vehicle every two seconds per axis, giving 5
    nominal vehicles per 10-second decision interval in each direction.
    """
    if not isinstance(decision_interval_seconds, Real) or not math.isfinite(float(decision_interval_seconds)) or decision_interval_seconds <= 0:
        raise ValueError("decision_interval_seconds must be finite and positive")
    routes_path = Path(routes_file)
    root = ET.parse(routes_path).getroot()
    rates = {"NS_route": 0.0, "EW_route": 0.0}
    seen_flow = {route_id: False for route_id in rates}
    for flow in root.findall("flow"):
        route_id = flow.get("route")
        if route_id not in rates:
            continue
        period_text = flow.get("period")
        if period_text is None:
            raise ValueError(
                f"flow {flow.get('id')} must use a 'period' for nominal rate derivation; "
                "provide explicit arrival_rates for other flow forms"
            )
        period = float(period_text)
        if not math.isfinite(period) or period <= 0:
            raise ValueError(f"flow {flow.get('id')} period must be finite and positive")
        rates[route_id] += float(decision_interval_seconds) / period
        seen_flow[route_id] = True
    if not all(seen_flow.values()):
        missing = [route_id for route_id, found in seen_flow.items() if not found]
        raise ValueError(f"route XML has no periodic flow for: {', '.join(missing)}")
    return rates["NS_route"], rates["EW_route"]


class SumoWebsterAdapter:
    """Use the root WebsterController plan for binary actions in the SUMO env.

    Args:
        environment: Shared SumoTrafficEnvironment instance. Its A1 yellow
            durations determine total cycle lost time unless explicitly set.
        arrival_rates: Expected vehicles per decision interval (NS, EW). If
            omitted, these are calculated from the periodic route-file flows.
        saturation_flows: Assumed vehicles/hour per direction. SUMO does not
            provide calibrated saturation flow; defaults retain the original
            controller's illustrative 1800/1800 assumptions.
        lost_time_seconds: Total lost time per two-phase cycle. By default this
            is the sum of A1's two configured yellow intervals (3+3 seconds).
    """

    def __init__(
        self,
        environment,
        arrival_rates: Optional[Sequence[float]] = None,
        saturation_flows: Sequence[float] = (1800.0, 1800.0),
        lost_time_seconds: Optional[float] = None,
        min_cycle_seconds: float = 30.0,
        max_cycle_seconds: float = 120.0,
        routes_file: Path | str | None = None,
    ) -> None:
        self.interval_seconds = float(environment.decision_interval_seconds)
        selected_routes_file = routes_file or getattr(environment, "routes_file", ROUTES_FILE)
        if arrival_rates is None:
            arrival_rates = rates_from_route_flows(self.interval_seconds, selected_routes_file)
            self.arrival_rate_source = str(Path(selected_routes_file).resolve())
        else:
            if len(arrival_rates) != 2:
                raise ValueError("arrival_rates must contain (NS rate, EW rate)")
            self.arrival_rate_source = "explicit configuration override"
        self.arrival_rates = (float(arrival_rates[0]), float(arrival_rates[1]))
        self.saturation_flows = (float(saturation_flows[0]), float(saturation_flows[1]))
        signal_yellow = environment.signal["yellow_seconds"]
        self.signal_yellow_seconds = (float(signal_yellow[0]), float(signal_yellow[1]))
        self.lost_time_seconds = (
            sum(self.signal_yellow_seconds)
            if lost_time_seconds is None else float(lost_time_seconds)
        )
        self.lost_time_source = (
            "sum of A1 NS/EW yellow phase durations from rl_intersection.net.xml"
            if lost_time_seconds is None else "explicit configuration override"
        )
        self.controller = WebsterController(
            arrival_rates=self.arrival_rates,
            interval_seconds=self.interval_seconds,
            saturation_flows=self.saturation_flows,
            lost_time_seconds=self.lost_time_seconds,
            min_cycle_seconds=min_cycle_seconds,
            max_cycle_seconds=max_cycle_seconds,
        )
        self.timing_plan = dict(self.controller.timing_plan)

    def reset(self) -> None:
        """Restart the Webster schedule at NS, aligned with SUMO reset()."""
        self.controller.reset()

    def choose_action(self) -> int:
        """Return the existing controller's action (0 NS, 1 EW)."""
        action = self.controller.choose_action()
        if action not in (0, 1):
            raise RuntimeError(f"WebsterController returned invalid action {action!r}")
        return int(action)
