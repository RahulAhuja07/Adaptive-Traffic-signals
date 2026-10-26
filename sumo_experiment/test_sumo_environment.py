"""Focused adapter tests; deterministic unit coverage does not need SUMO."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import shutil
import unittest

from sumo_experiment.sumo_environment import SumoTrafficEnvironment


class FakeLane:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def getLastStepHaltingNumber(self, lane_id: str) -> int:
        return self.counts.get(lane_id, 0)


class FakeVehicle:
    def getIDList(self) -> list[str]:
        return []

    def getSpeed(self, vehicle_id: str) -> float:
        del vehicle_id
        return 1.0


class FakeSimulation:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.time = 0.0

    def getTime(self) -> float:
        return self.time

    def simulationStep(self) -> None:
        self.time += self.connection.step_length
        self.connection.apply_queue_schedule(self.time)

    def getArrivedIDList(self) -> list[str]:
        return []

    def getDepartedNumber(self) -> int:
        return 0

    def getMinExpectedNumber(self) -> int:
        return self.connection.min_expected


class FakeTrafficLight:
    def __init__(self) -> None:
        self.phase_calls: list[int] = []
        self.duration_calls: list[float] = []

    def setPhase(self, tls_id: str, phase: int) -> None:
        if tls_id != "A1":
            raise AssertionError(f"unexpected traffic light {tls_id}")
        self.phase_calls.append(phase)

    def setPhaseDuration(self, tls_id: str, duration: float) -> None:
        if tls_id != "A1":
            raise AssertionError(f"unexpected traffic light {tls_id}")
        self.duration_calls.append(duration)


class FakeConnection:
    def __init__(self, step_length: float = 1.0) -> None:
        self.step_length = step_length
        self.lane = FakeLane()
        self.vehicle = FakeVehicle()
        self.trafficlight = FakeTrafficLight()
        self.simulation = FakeSimulation(self)
        self.min_expected = 100
        self.closed = False
        self.queue_schedule: dict[float, tuple[int, int]] = {}
        self.axis_lanes: dict[str, list[str]] = {"NS": [], "EW": []}

    def apply_queue_schedule(self, time: float) -> None:
        if time not in self.queue_schedule:
            return
        ns_queue, ew_queue = self.queue_schedule[time]
        self.lane.counts = {lane: 0 for lane in self.axis_lanes["NS"] + self.axis_lanes["EW"]}
        if self.axis_lanes["NS"]:
            self.lane.counts[self.axis_lanes["NS"][0]] = ns_queue
        if self.axis_lanes["EW"]:
            self.lane.counts[self.axis_lanes["EW"][0]] = ew_queue

    def simulationStep(self) -> None:
        self.simulation.simulationStep()

    def close(self) -> None:
        self.closed = True


class FakeTraCI:
    def __init__(self) -> None:
        self.connections: dict[str, FakeConnection] = {}
        self.commands: list[list[str]] = []

    def start(self, command: list[str], label: str) -> None:
        self.commands.append(command)
        connection = FakeConnection()
        self.connections[label] = connection

    def getConnection(self, label: str) -> FakeConnection:
        connection = self.connections[label]
        return connection


class SumoTrafficEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs) -> tuple[SumoTrafficEnvironment, FakeTraCI]:
        fake = FakeTraCI()
        env = SumoTrafficEnvironment(
            episode_horizon_seconds=kwargs.pop("episode_horizon_seconds", 20),
            traci_module=fake,
            sumo_binary="/mock/bin/sumo",
            **kwargs,
        )
        return env, fake

    def seed_axis_queues(self, env: SumoTrafficEnvironment, connection: FakeConnection,
                         ns: int, ew: int) -> None:
        connection.axis_lanes = env.signal["lanes"]
        connection.lane.counts = {lane: 0 for lanes in env.signal["lanes"].values() for lane in lanes}
        connection.lane.counts[env.signal["lanes"]["NS"][0]] = ns
        connection.lane.counts[env.signal["lanes"]["EW"][0]] = ew

    def test_reset_starts_fresh_simulation_with_module_relative_inputs(self) -> None:
        env, fake = self.make_env()
        try:
            first_state = env.reset(seed=17)
            first_connection = fake.connections[env._label]
            self.assertEqual(first_state, 0)
            self.assertEqual(first_connection.trafficlight.phase_calls, [0])
            command = fake.commands[0]
            net_path = Path(command[command.index("--net-file") + 1])
            route_path = Path(command[command.index("--route-files") + 1])
            self.assertEqual(net_path, Path(__file__).resolve().parent / "rl_intersection.net.xml")
            self.assertEqual(route_path, Path(__file__).resolve().parent / "rl_routes.rou.xml")
            self.assertIn(("--seed", "17"), list(zip(command, command[1:])))

            env.reset(seed=18)
            self.assertTrue(first_connection.closed)
            self.assertEqual(len(fake.commands), 2)
            self.assertEqual(fake.commands[1][-2:], ["--seed", "18"])
        finally:
            env.close()

    def test_both_actions_and_switching_use_the_configured_yellow(self) -> None:
        env, fake = self.make_env()
        try:
            env.reset()
            connection = fake.connections[env._label]
            _, _, done, ns_info = env.step(0)
            self.assertFalse(done)
            self.assertEqual(ns_info["phase"], 0)
            self.assertEqual(ns_info["time_seconds"], 10)

            next_state, _, done, ew_info = env.step(1)
            self.assertTrue(done)
            self.assertEqual(ew_info["phase"], 1)
            self.assertEqual(ew_info["yellow_clearance_seconds"], 3.0)
            self.assertEqual(ew_info["time_seconds"], 20)
            self.assertEqual(next_state // 9, 1)

            # A1's actual sequence is NS green -> NS yellow -> EW green.
            self.assertEqual(connection.trafficlight.phase_calls, [0, 0, 1, 2])
            self.assertEqual(len(connection.trafficlight.duration_calls), 3)
            with self.assertRaises(RuntimeError):
                env.step(0)
        finally:
            env.close()

    def test_halting_lane_counts_drive_bins_reward_and_queue_metric(self) -> None:
        env, fake = self.make_env(episode_horizon_seconds=10)
        try:
            initial_state = env.reset()
            connection = fake.connections[env._label]
            self.seed_axis_queues(env, connection, ns=7, ew=10)
            initial_state = env._state()
            self.assertEqual(initial_state, 5)  # phase 0, NS medium, EW high
            self.assertEqual(env.get_queue_observation(), {
                "phase": 0, "queue_ns": 7, "queue_ew": 10,
            })

            connection.queue_schedule = {float(t): (1, 2) for t in range(1, 11)}
            next_state, reward, done, info = env.step(0)
            self.assertTrue(done)
            self.assertEqual(info["queue_ns"], 1)
            self.assertEqual(info["queue_ew"], 2)
            self.assertEqual(next_state, 0)
            self.assertEqual(reward, -1.0)  # -(17 + 3) / 2 / 10
            self.assertEqual(env.get_metrics()["time_weighted_avg_queue"], 3.0)
        finally:
            env.close()

    def test_episode_horizon_and_action_validation(self) -> None:
        env, _ = self.make_env(episode_horizon_seconds=10)
        try:
            env.reset()
            with self.assertRaises(ValueError):
                env.step(True)
            _, _, done, _ = env.step(0)
            self.assertTrue(done)
        finally:
            env.close()


def _live_sumo_available() -> bool:
    sumo_home = os.environ.get("SUMO_HOME")
    binary = os.environ.get("SUMO_BINARY") or shutil.which("sumo")
    if not binary and sumo_home:
        binary = str(Path(sumo_home) / "bin" / "sumo")
    traci_spec = importlib.util.find_spec("traci")
    tools_traci = Path(sumo_home) / "tools" / "traci" / "__init__.py" if sumo_home else None
    return bool(binary and (traci_spec or (tools_traci and tools_traci.is_file())))


@unittest.skipUnless(_live_sumo_available(), "SUMO binary and matching TraCI Python tools are unavailable")
class LiveSumoSmokeTest(unittest.TestCase):
    def test_live_reset_switch_and_horizon(self) -> None:
        env = SumoTrafficEnvironment(episode_horizon_seconds=10)
        try:
            self.assertEqual(env.reset(seed=29), 0)
            state, reward, done, info = env.step(1)
            self.assertEqual(state // 9, 1)
            self.assertTrue(math.isfinite(reward))
            self.assertTrue(done)
            self.assertEqual(info["yellow_clearance_seconds"], 3.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
