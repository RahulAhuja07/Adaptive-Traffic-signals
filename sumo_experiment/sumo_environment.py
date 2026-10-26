"""TraCI-backed adapter for the project's original 18-state RL agents.

This module intentionally does not import SUMO or TraCI when imported. Those
optional dependencies are loaded only when :meth:`reset` starts SUMO. The
environment follows the custom simulator's ``reset`` / ``step`` shape:

* ``reset(seed=None) -> int`` returns the initial 18-state integer.
* ``step(action) -> (state, reward, done, info)`` accepts 0 (NS) or 1 (EW).
* ``get_metrics()`` returns episode metrics, including stopped waiting time.

Each action controls one configurable decision interval (10 seconds by
default). A direction change runs A1's actual three-second yellow phase before
the requested green phase. Queue counts are the number of halting vehicles
(speed <= 0.1 m/s) on A1's inbound lanes, summed by the verified road axis;
they are SUMO observations, not exact replicas of the custom simulator's FIFO
queues. The reward preserves the custom simulator's formula and scale:
``-((queue_before + queue_after) / 2) / 10`` per decision. Only the observed
queue definition differs.

SUMO waiting-time metrics here count seconds spent at or below 0.1 m/s,
sampled each SUMO step, for vehicles that complete their route. Throughput is
completed route trips per simulated minute. These definitions are explicit
because SUMO vehicle dynamics and the custom simulator's arrival/departure
timestamps are different.
"""

from __future__ import annotations

import importlib
import math
from numbers import Integral
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
NETWORK_FILE = HERE / "rl_intersection.net.xml"
ROUTES_FILE = HERE / "rl_routes.rou.xml"
SUSTAINED_ROUTES_ENV = "SUMO_ROUTES_FILE"
TRAFFIC_LIGHT_ID = "A1"
HALTING_SPEED_MPS = 0.1


def _integer_steps(seconds: float, step_length: float, label: str) -> int:
    steps = seconds / step_length
    rounded = round(steps)
    if rounded <= 0 or not math.isclose(steps, rounded, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"{label} must be a positive multiple of SUMO step length")
    return int(rounded)


def _read_signal_configuration() -> dict[str, Any]:
    """Read A1's actual incoming lanes, route links, green phases, and yellows."""
    if not NETWORK_FILE.is_file() or not ROUTES_FILE.is_file():
        raise FileNotFoundError(
            f"SUMO input files must exist beside this module: {NETWORK_FILE} and {ROUTES_FILE}"
        )

    network = ET.parse(NETWORK_FILE).getroot()
    routes_root = ET.parse(ROUTES_FILE).getroot()
    junctions = {node.get("id"): node for node in network.findall("junction")}
    edges = {edge.get("id"): edge for edge in network.findall("edge") if edge.get("function") != "internal"}
    center = junctions.get(TRAFFIC_LIGHT_ID)
    if center is None or center.get("type") != "traffic_light":
        raise ValueError("rl_intersection.net.xml must contain traffic-light junction A1")
    center_x, center_y = float(center.get("x", "nan")), float(center.get("y", "nan"))

    inbound_by_axis: dict[str, list[str]] = {"NS": [], "EW": []}
    for edge in edges.values():
        if edge.get("to") != TRAFFIC_LIGHT_ID:
            continue
        source = junctions.get(edge.get("from"))
        if source is None:
            continue
        dx = abs(float(source.get("x", "nan")) - center_x)
        dy = abs(float(source.get("y", "nan")) - center_y)
        if dx == dy:
            raise ValueError(f"Cannot determine axis for inbound edge {edge.get('id')}")
        inbound_by_axis["EW" if dx > dy else "NS"].append(edge.get("id", ""))

    if len(inbound_by_axis["NS"]) != 2 or len(inbound_by_axis["EW"]) != 2:
        raise ValueError("A1 must have two opposing NS approaches and two opposing EW approaches")
    lanes_by_axis = {axis: [] for axis in ("NS", "EW")}
    for axis, incoming_edges in inbound_by_axis.items():
        for edge_id in incoming_edges:
            edge = edges[edge_id]
            lanes_by_axis[axis].extend(lane.get("id", "") for lane in edge.findall("lane"))
    if any(not lane_ids for lane_ids in lanes_by_axis.values()):
        raise ValueError("A1 is missing inbound lanes for an approach axis")

    route_pairs: dict[str, tuple[str, str]] = {}
    for route in routes_root.findall("route"):
        route_id = route.get("id")
        edge_ids = (route.get("edges") or "").split()
        if route_id in ("NS_route", "EW_route") and len(edge_ids) == 2:
            route_pairs[route_id] = (edge_ids[0], edge_ids[1])
    if set(route_pairs) != {"NS_route", "EW_route"}:
        raise ValueError("rl_routes.rou.xml must define NS_route and EW_route through A1")

    # SUMO uses x/y coordinates with B2 north, B4 south, B1 east, and B3 west.
    # Confirm the named flows agree with those coordinates before mapping them.
    for route_id, axis in (("NS_route", "NS"), ("EW_route", "EW")):
        incoming, outgoing = route_pairs[route_id]
        incoming_edge, outgoing_edge = edges.get(incoming), edges.get(outgoing)
        if (incoming_edge is None or outgoing_edge is None
                or incoming_edge.get("to") != TRAFFIC_LIGHT_ID
                or outgoing_edge.get("from") != TRAFFIC_LIGHT_ID):
            raise ValueError(f"{route_id} is not a connected route through A1")
        source = junctions[incoming_edge.get("from")]
        dx = abs(float(source.get("x", "nan")) - center_x)
        dy = abs(float(source.get("y", "nan")) - center_y)
        actual_axis = "EW" if dx > dy else "NS"
        if actual_axis != axis or incoming not in inbound_by_axis[axis]:
            raise ValueError(f"{route_id} does not approach A1 on its named axis")

    direct_connections = [
        connection for connection in network.findall("connection")
        if connection.get("tl") == TRAFFIC_LIGHT_ID
        and connection.get("from") in edges
        and connection.get("dir") == "s"
    ]
    through_links: dict[str, set[int]] = {"NS": set(), "EW": set()}
    for connection in direct_connections:
        incoming = connection.get("from", "")
        axis = next((key for key, values in inbound_by_axis.items() if incoming in values), None)
        if axis is not None:
            through_links[axis].add(int(connection.get("linkIndex", "-1")))

    # The named straight-through route must be represented among the controlled
    # links. Opposing through movements are included so each action serves both
    # approaches on its axis.
    for route_id, axis in (("NS_route", "NS"), ("EW_route", "EW")):
        incoming, outgoing = route_pairs[route_id]
        route_links = {
            int(connection.get("linkIndex", "-1"))
            for connection in direct_connections
            if connection.get("from") == incoming and connection.get("to") == outgoing
        }
        if not route_links or not route_links.issubset(through_links[axis]):
            raise ValueError(f"{route_id} has no A1-controlled straight-through links")

    programs = [program for program in network.findall("tlLogic") if program.get("id") == TRAFFIC_LIGHT_ID]
    if len(programs) != 1 or programs[0].get("type") != "static":
        raise ValueError("A1 must have one static traffic-light program")
    phases = programs[0].findall("phase")
    if len(phases) != 4:
        raise ValueError("A1 signal program changed: expected green/yellow/green/yellow phases")
    phase_states = [phase.get("state", "") for phase in phases]
    phase_durations = [float(phase.get("duration", "0")) for phase in phases]

    def is_green(phase_index: int, links: set[int]) -> bool:
        state = phase_states[phase_index]
        return all(link < len(state) and state[link] in "Gg" for link in links)

    def is_yellow(phase_index: int, links: set[int]) -> bool:
        state = phase_states[phase_index]
        return all(link < len(state) and state[link] in "yY" for link in links)

    def is_safe_yellow(phase_index: int, axis: str) -> bool:
        other_axis = "EW" if axis == "NS" else "NS"
        state = phase_states[phase_index]
        return (
            is_yellow(phase_index, through_links[axis])
            and all(state[link] in "rR" for link in through_links[other_axis])
        )

    def find_green(axis: str) -> int:
        other_axis = "EW" if axis == "NS" else "NS"
        candidates = [
            index for index in range(len(phases))
            if is_green(index, through_links[axis])
            and not any(phase_states[index][link] in "Gg" for link in through_links[other_axis])
        ]
        if len(candidates) != 1:
            raise ValueError(f"A1 program must have one exclusive {axis} through-green phase")
        return candidates[0]

    green_ns, green_ew = find_green("NS"), find_green("EW")
    yellow_ns = (green_ns + 1) % len(phases)
    yellow_ew = (green_ew + 1) % len(phases)
    if not is_safe_yellow(yellow_ns, "NS"):
        raise ValueError("A1's NS green must transition through its configured yellow phase")
    if not is_safe_yellow(yellow_ew, "EW"):
        raise ValueError("A1's EW green must transition through its configured yellow phase")
    if (green_ew != (yellow_ns + 1) % len(phases)
            or green_ns != (yellow_ew + 1) % len(phases)):
        raise ValueError("A1 phases do not follow green-yellow-opposing-green-yellow order")

    return {
        "lanes": lanes_by_axis,
        "links": through_links,
        "green_phase": {0: green_ns, 1: green_ew},
        "yellow_phase": {0: yellow_ns, 1: yellow_ew},
        "yellow_seconds": {0: phase_durations[yellow_ns], 1: phase_durations[yellow_ew]},
    }


class SumoTrafficEnvironment:
    """Control A1 in SUMO with the original simulator's agent-facing API.

    Args:
        episode_horizon_seconds: Maximum episode duration. SUMO may finish
            earlier when all scheduled flows and active vehicles are complete.
        decision_interval_seconds: Simulated time controlled by each action.
        simulation_step_seconds: SUMO internal step length.
        sumo_binary: Optional binary path/name. If omitted, ``SUMO_BINARY``,
            ``SUMO_HOME/bin/sumo``, or ``PATH`` is used in that order.
        traci_module: Optional TraCI-like module for deterministic tests.
            Normal callers should omit it; TraCI is imported lazily on reset.
    """

    def __init__(
        self,
        episode_horizon_seconds: float = 1800.0,
        decision_interval_seconds: float = 10.0,
        simulation_step_seconds: float = 1.0,
        sumo_binary: Optional[str] = None,
        traci_module: Any = None,
        routes_file: Optional[Path | str] = None,
    ) -> None:
        """Create the adapter.

        ``routes_file`` takes precedence over ``SUMO_ROUTES_FILE``; when both
        are omitted, the original ``rl_routes.rou.xml`` remains the default.
        This keeps legacy runs unchanged while letting all SUMO runners select
        one shared comparison scenario through the environment variable.
        """
        for value, label in (
            (episode_horizon_seconds, "episode_horizon_seconds"),
            (decision_interval_seconds, "decision_interval_seconds"),
            (simulation_step_seconds, "simulation_step_seconds"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a finite positive number")
        if episode_horizon_seconds < decision_interval_seconds:
            raise ValueError("episode horizon must be at least one decision interval")
        _integer_steps(decision_interval_seconds, simulation_step_seconds, "decision interval")
        self.signal = _read_signal_configuration()
        for direction in (0, 1):
            if self.signal["yellow_seconds"][direction] > decision_interval_seconds:
                raise ValueError("decision interval must be at least as long as the signal yellow phase")
            _integer_steps(self.signal["yellow_seconds"][direction], simulation_step_seconds, "yellow duration")
        _integer_steps(episode_horizon_seconds, decision_interval_seconds, "episode horizon")

        self.episode_horizon_seconds = float(episode_horizon_seconds)
        self.decision_interval_seconds = float(decision_interval_seconds)
        self.simulation_step_seconds = float(simulation_step_seconds)
        self.sumo_binary = sumo_binary
        configured_routes = routes_file or os.environ.get(SUSTAINED_ROUTES_ENV)
        self.routes_file = Path(configured_routes).expanduser().resolve() if configured_routes else ROUTES_FILE
        if not self.routes_file.is_file():
            raise FileNotFoundError(f"SUMO route file does not exist: {self.routes_file}")
        self._validate_route_scenario(self.routes_file)
        self._traci_module = traci_module
        self._connection: Any = None
        self._label = f"sumo_env_{id(self):x}"
        self._has_reset = False
        self._done = True
        self._current_action = 0
        self._decision_count = 0
        self._sim_time = 0.0
        self._total_inserted = 0
        self._total_completed = 0
        self._completed_stopped_waits: list[float] = []
        self._vehicle_stopped_seconds: dict[str, float] = {}
        self._queue_time_sum = 0.0
        self._cumulative_reward = 0.0
        self._termination_reason: Optional[str] = None

    @staticmethod
    def _validate_route_scenario(routes_file: Path) -> None:
        """Require the scenario to preserve the verified A1 route definitions."""
        route_root = ET.parse(routes_file).getroot()
        route_map = {
            route.get("id"): (route.get("edges") or "").split()
            for route in route_root.findall("route")
        }
        if route_map.get("NS_route") != ["B2A1", "A1B4"]:
            raise ValueError("SUMO scenario must preserve NS_route through A1")
        if route_map.get("EW_route") != ["B1A1", "A1B3"]:
            raise ValueError("SUMO scenario must preserve EW_route through A1")

    def _resolve_sumo_binary(self) -> str:
        candidate = self.sumo_binary or os.environ.get("SUMO_BINARY")
        if candidate:
            return str(candidate)
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            executable = Path(sumo_home) / "bin" / "sumo"
            if executable.is_file():
                return str(executable)
        found = shutil.which("sumo")
        if found:
            return found
        raise FileNotFoundError(
            "Could not find the SUMO binary. Set SUMO_BINARY, set SUMO_HOME, "
            "or add sumo to PATH."
        )

    def _load_traci(self) -> Any:
        if self._traci_module is not None:
            return self._traci_module
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            tools_dir = str(Path(sumo_home) / "tools")
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
        try:
            return importlib.import_module("traci")
        except ImportError as error:
            raise ImportError(
                "TraCI Python tools were not found. Install the matching TraCI "
                "package or set SUMO_HOME to the SUMO installation."
            ) from error

    def _close_connection(self) -> None:
        if self._connection is None:
            return
        connection, self._connection = self._connection, None
        try:
            connection.close()
        except Exception:
            # A connection can already be closed after SUMO naturally exits.
            pass

    def _read_queues(self) -> tuple[int, int]:
        lane_domain = self._connection.lane
        totals = []
        for axis in ("NS", "EW"):
            totals.append(sum(
                int(lane_domain.getLastStepHaltingNumber(lane_id))
                for lane_id in self.signal["lanes"][axis]
            ))
        return totals[0], totals[1]

    def _state(self, queues: Optional[tuple[int, int]] = None) -> int:
        if queues is None:
            queues = self._read_queues()
        ns_bin = 0 if queues[0] <= 4 else 1 if queues[0] <= 9 else 2
        ew_bin = 0 if queues[1] <= 4 else 1 if queues[1] <= 9 else 2
        return int(self._current_action * 9 + ns_bin * 3 + ew_bin)

    def get_queue_observation(self) -> dict[str, int]:
        """Return the current green axis and raw inbound halting counts.

        This read-only observation preserves the adapter's original 18-state
        ``reset``/``step`` return values while allowing richer agents to derive
        features from the same underlying SUMO lane counts. Queue values are
        counts of halting vehicles, not the custom simulator's FIFO queues.
        """
        if not self._has_reset or self._connection is None:
            raise RuntimeError("call reset() before requesting a SUMO observation")
        queue_ns, queue_ew = self._read_queues()
        return {
            "phase": int(self._current_action),
            "queue_ns": int(queue_ns),
            "queue_ew": int(queue_ew),
        }

    def _measure_step_metrics(self) -> None:
        connection = self._connection
        active_ids = set(connection.vehicle.getIDList())
        for vehicle_id in active_ids:
            if float(connection.vehicle.getSpeed(vehicle_id)) <= HALTING_SPEED_MPS:
                self._vehicle_stopped_seconds[vehicle_id] = (
                    self._vehicle_stopped_seconds.get(vehicle_id, 0.0)
                    + self.simulation_step_seconds
                )
        arrived_ids = list(connection.simulation.getArrivedIDList())
        self._total_inserted += int(connection.simulation.getDepartedNumber())
        self._total_completed += len(arrived_ids)
        for vehicle_id in arrived_ids:
            self._completed_stopped_waits.append(
                self._vehicle_stopped_seconds.pop(vehicle_id, 0.0)
            )
        for vehicle_id in self._vehicle_stopped_seconds.keys() - active_ids:
            self._vehicle_stopped_seconds.pop(vehicle_id, None)

        ns_queue, ew_queue = self._read_queues()
        self._queue_time_sum += (ns_queue + ew_queue) * self.simulation_step_seconds
        self._sim_time = float(connection.simulation.getTime())

    def _advance_steps(self, number_of_steps: int) -> None:
        for _ in range(number_of_steps):
            self._connection.simulationStep()
            self._measure_step_metrics()

    def reset(self, seed: Optional[int] = None) -> int:
        """Start a fresh SUMO process and return its initial discrete state."""
        self._close_connection()
        traci = self._load_traci()
        binary = self._resolve_sumo_binary()
        command = [
            binary,
            "--net-file", str(NETWORK_FILE),
            "--route-files", str(self.routes_file),
            "--begin", "0",
            "--end", str(self.episode_horizon_seconds),
            "--step-length", str(self.simulation_step_seconds),
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ]
        if seed is not None:
            command.extend(("--seed", str(int(seed))))
        traci.start(command, label=self._label)
        self._connection = traci.getConnection(self._label)

        self._decision_count = 0
        self._sim_time = float(self._connection.simulation.getTime())
        self._total_inserted = 0
        self._total_completed = 0
        self._completed_stopped_waits = []
        self._vehicle_stopped_seconds = {}
        self._queue_time_sum = 0.0
        self._cumulative_reward = 0.0
        self._termination_reason = None
        self._current_action = 0
        self._done = False
        self._has_reset = True

        # Start on NS green. The initial state is not advanced, so its queues
        # are the initial A1 observations.
        self._hold_green(0)
        return self._state()

    def _hold_green(self, action: int) -> None:
        phase_index = self.signal["green_phase"][action]
        self._connection.trafficlight.setPhase(TRAFFIC_LIGHT_ID, phase_index)
        # Prevent the static program's green duration from switching phases in
        # the middle of an agent decision. The next action can still switch,
        # but only by entering the program's yellow phase first.
        self._connection.trafficlight.setPhaseDuration(
            TRAFFIC_LIGHT_ID,
            self.decision_interval_seconds + self.simulation_step_seconds,
        )

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Advance one fixed interval using action 0 (NS) or 1 (EW).

        Changing axes consumes the configured yellow time inside the interval;
        the requested green receives the interval's remaining time. If there
        are no scheduled or active vehicles, SUMO completion also ends the
        episode before the configured maximum horizon.
        """
        if not self._has_reset or self._connection is None:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("episode is finished; call reset() before stepping again")
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in (0, 1):
            raise ValueError("action must be 0 (NS green) or 1 (EW green)")

        queue_before = self._read_queues()
        start_completed = self._total_completed
        interval_steps = _integer_steps(
            self.decision_interval_seconds,
            self.simulation_step_seconds,
            "decision interval",
        )
        yellow_steps = 0
        if action != self._current_action:
            yellow_seconds = self.signal["yellow_seconds"][self._current_action]
            yellow_steps = _integer_steps(
                yellow_seconds, self.simulation_step_seconds, "yellow duration"
            )
            self._connection.trafficlight.setPhase(
                TRAFFIC_LIGHT_ID,
                self.signal["yellow_phase"][self._current_action],
            )
            self._advance_steps(yellow_steps)
            self._current_action = int(action)

        self._hold_green(self._current_action)
        self._advance_steps(interval_steps - yellow_steps)
        self._decision_count += 1

        queue_after = self._read_queues()
        queue_before_total = sum(queue_before)
        queue_after_total = sum(queue_after)
        reward = -((queue_before_total + queue_after_total) / 2.0) / 10.0
        self._cumulative_reward += reward

        min_expected = int(self._connection.simulation.getMinExpectedNumber())
        horizon_reached = self._sim_time >= self.episode_horizon_seconds - 1e-8
        self._done = horizon_reached or min_expected <= 0
        if horizon_reached:
            self._termination_reason = "horizon"
        elif min_expected <= 0:
            self._termination_reason = "sumo_no_expected_vehicles"
        info = {
            "step": self._decision_count,
            "time_seconds": self._sim_time,
            "phase": self._current_action,
            "queue_ns": queue_after[0],
            "queue_ew": queue_after[1],
            "queue_total": queue_after_total,
            "queue_before": queue_before_total,
            "queue_after": queue_after_total,
            "completed_this_step": self._total_completed - start_completed,
            "total_completed": self._total_completed,
            "yellow_clearance_seconds": (
                self.signal["yellow_seconds"][1 - action] if yellow_steps else 0.0
            ),
            "termination_reason": self._termination_reason if self._done else "",
        }
        return self._state(queue_after), float(reward), self._done, info

    def get_metrics(self) -> dict[str, Any]:
        """Return metrics with SUMO-specific stopped-wait and trip definitions."""
        elapsed_minutes = self._sim_time / 60.0
        remaining_expected = (
            int(self._connection.simulation.getMinExpectedNumber())
            if self._connection is not None else None
        )
        return {
            "avg_stopped_waiting_time_sec": (
                sum(self._completed_stopped_waits) / len(self._completed_stopped_waits)
                if self._completed_stopped_waits else float("nan")
            ),
            "total_stopped_waiting_time_sec_completed": float(
                sum(self._completed_stopped_waits)
            ),
            "time_weighted_avg_queue": (
                self._queue_time_sum / self._sim_time if self._sim_time > 0 else 0.0
            ),
            "throughput_veh_per_min": (
                self._total_completed / elapsed_minutes if elapsed_minutes > 0 else 0.0
            ),
            "completed_vehicles": float(self._total_completed),
            "sumo_inserted_vehicles": float(self._total_inserted),
            "remaining_expected_vehicles": (
                float(remaining_expected) if remaining_expected is not None else float("nan")
            ),
            "demand_completed": remaining_expected == 0,
            "final_queue": float(sum(self._read_queues()) if self._connection else 0),
            "cumulative_reward": float(self._cumulative_reward),
            "simulated_time_sec": float(self._sim_time),
        }

    def close(self) -> None:
        """Close this environment's SUMO process, if one is running."""
        self._close_connection()
        self._done = True

    def __enter__(self) -> "SumoTrafficEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
