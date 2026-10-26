"""Demand-duration checks for the isolated sustained comparison scenario."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from sumo_experiment.create_comparison_routes import create_route_scenario
from sumo_experiment.sumo_environment import SumoTrafficEnvironment
from sumo_experiment.webster_adapter import rates_from_route_flows


class ScenarioGenerationTests(unittest.TestCase):
    def test_route_generation_extends_both_periodic_flows_without_touching_original(self):
        original = Path(__file__).resolve().parent / "rl_routes.rou.xml"
        original_before = original.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            output = create_route_scenario(1800, Path(temp) / "comparison.rou.xml")
            root = ET.parse(output).getroot()
            flows = {flow.get("id"): flow for flow in root.findall("flow")}
            self.assertEqual(set(flows), {"NS_flow", "EW_flow"})
            self.assertEqual({flow.get("end") for flow in flows.values()}, {"1800"})
            self.assertEqual({flow.get("period") for flow in flows.values()}, {"2"})
            self.assertEqual({flow.get("route") for flow in flows.values()}, {"NS_route", "EW_route"})
            self.assertEqual(sum(int((float(flow.get("end")) - float(flow.get("begin")))
                                     / float(flow.get("period"))) for flow in flows.values()), 1800)
            self.assertEqual(rates_from_route_flows(10.0, output), (5.0, 5.0))
            self.assertEqual(original.read_bytes(), original_before)

    def test_original_route_path_is_rejected_as_generator_output(self):
        original = Path(__file__).resolve().parent / "rl_routes.rou.xml"
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            create_route_scenario(1800, original)

    def test_environment_uses_explicit_scenario_path_and_ends_at_horizon(self):
        from sumo_experiment.test_sumo_environment import FakeTraCI

        with tempfile.TemporaryDirectory() as temp:
            routes = create_route_scenario(1800, Path(temp) / "scenario.rou.xml")
            fake = FakeTraCI()
            env = SumoTrafficEnvironment(
                episode_horizon_seconds=10, routes_file=routes, traci_module=fake,
                sumo_binary="/mock/sumo",
            )
            try:
                env.reset(seed=3)
                command = fake.commands[0]
                selected = Path(command[command.index("--route-files") + 1])
                self.assertEqual(selected, routes.resolve())
                _, _, done, info = env.step(0)
                self.assertTrue(done)
                self.assertEqual(info["termination_reason"], "horizon")
            finally:
                env.close()

    def test_environment_route_scenario_can_be_selected_by_documented_variable(self):
        from sumo_experiment.test_sumo_environment import FakeTraCI

        with tempfile.TemporaryDirectory() as temp:
            routes = create_route_scenario(1800, Path(temp) / "scenario.rou.xml")
            with patch.dict(os.environ, {"SUMO_ROUTES_FILE": str(routes)}):
                env = SumoTrafficEnvironment(traci_module=FakeTraCI(), sumo_binary="/mock/sumo")
            self.assertEqual(env.routes_file, routes.resolve())
            env.close()


def _live_sumo_available() -> bool:
    home = os.environ.get("SUMO_HOME")
    binary = os.environ.get("SUMO_BINARY") or shutil.which("sumo")
    if not binary and home:
        binary = str(Path(home) / "bin" / "sumo")
    traci_spec = importlib.util.find_spec("traci")
    tools_traci = Path(home) / "tools" / "traci" / "__init__.py" if home else None
    return bool(binary and (traci_spec or (tools_traci and tools_traci.is_file())))


@unittest.skipUnless(_live_sumo_available(), "SUMO binary and matching TraCI Python tools are unavailable")
class LiveSustainedDemandSmokeTest(unittest.TestCase):
    def test_vehicles_continue_to_be_inserted_after_original_demand_ends(self):
        scenario = create_route_scenario(1800)
        env = SumoTrafficEnvironment(episode_horizon_seconds=1800, routes_file=scenario)
        snapshots: dict[int, int] = {}
        try:
            env.reset(seed=9281)
            done = False
            decision = 0
            while not done:
                action = 0 if (decision // 6) % 2 == 0 else 1
                _, _, done, info = env.step(action)
                decision += 1
                second = int(info["time_seconds"])
                if second in (600, 700, 1200, 1700, 1800):
                    snapshots[second] = int(env.get_metrics()["sumo_inserted_vehicles"])
            self.assertIn(600, snapshots)
            self.assertIn(700, snapshots)
            self.assertIn(1700, snapshots)
            self.assertGreater(snapshots[700], snapshots[600])
            self.assertGreater(snapshots[1700], snapshots[700])
            self.assertGreater(snapshots[1800], snapshots[1700])
            self.assertEqual(info["termination_reason"], "horizon")
            self.assertEqual(info["time_seconds"], 1800.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
