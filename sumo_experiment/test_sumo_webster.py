"""Focused tests for the SUMO adapter around the existing Webster controller."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from sumo_experiment.webster_adapter import SumoWebsterAdapter, rates_from_route_flows
from sumo_experiment.webster_runner import evaluate_webster


class FakeEnvironment:
    def __init__(self, *, episode_horizon_seconds: float, decision_interval_seconds: float,
                 fail_on_reset: bool = False, fail_on_step: bool = False) -> None:
        self.episode_horizon_seconds = episode_horizon_seconds
        self.decision_interval_seconds = decision_interval_seconds
        self.signal = {"yellow_seconds": {0: 3.0, 1: 3.0}}
        self.fail_on_reset = fail_on_reset
        self.fail_on_step = fail_on_step
        self.closed = False
        self.step_count = 0
        self.previous_action = 0
        self.actions: list[int] = []
        self.yellow_seconds: list[float] = []

    def reset(self, seed: int | None = None) -> int:
        self.seed = seed
        if self.fail_on_reset:
            raise RuntimeError("simulated TraCI reset failure")
        return 0

    def step(self, action: int):
        if self.fail_on_step:
            raise RuntimeError("simulated TraCI step failure")
        self.actions.append(action)
        yellow = 3.0 if action != self.previous_action else 0.0
        self.yellow_seconds.append(yellow)
        self.previous_action = action
        self.step_count += 1
        done = self.step_count == 13
        return 0, -1.0, done, {
            "yellow_clearance_seconds": yellow,
            "termination_reason": "horizon" if done else "",
        }

    def get_metrics(self) -> dict[str, float]:
        return {
            "total_stopped_waiting_time_sec_completed": 20.0,
            "avg_stopped_waiting_time_sec": 10.0,
            "time_weighted_avg_queue": 7.0,
            "throughput_veh_per_min": 1.2,
            "completed_vehicles": 3.0,
            "final_queue": 12.0,
            "simulated_time_sec": 130.0,
            "cumulative_reward": -13.0,
        }

    def close(self) -> None:
        self.closed = True


class SumoWebsterTests(unittest.TestCase):
    def test_route_flows_derive_nominal_per_interval_rates(self) -> None:
        rates = rates_from_route_flows(10.0)
        self.assertEqual(rates, (5.0, 5.0))

    def test_existing_webster_formula_uses_sumo_lost_time_and_assumed_saturation(self) -> None:
        env = FakeEnvironment(episode_horizon_seconds=120, decision_interval_seconds=10)
        adapter = SumoWebsterAdapter(env)
        self.assertEqual(adapter.arrival_rates, (5.0, 5.0))
        self.assertEqual(adapter.saturation_flows, (1800.0, 1800.0))
        self.assertEqual(adapter.lost_time_seconds, 6.0)
        self.assertEqual(adapter.timing_plan["critical_flow_ratio_ns"], 1.0)
        self.assertEqual(adapter.timing_plan["critical_flow_ratio_ew"], 1.0)
        self.assertEqual(adapter.timing_plan["cycle_length_seconds"], 120.0)
        self.assertEqual(adapter.timing_plan["green_ns_intervals"], 6.0)
        self.assertEqual(adapter.timing_plan["green_ew_intervals"], 6.0)

    def test_controller_emits_original_interval_schedule_and_runner_records_clearance(self) -> None:
        env = FakeEnvironment(episode_horizon_seconds=130, decision_interval_seconds=10)
        controller = SumoWebsterAdapter(env)
        controller.reset()
        actions = [controller.choose_action() for _ in range(13)]
        self.assertEqual(actions[:6], [0] * 6)
        self.assertEqual(actions[6:12], [1] * 6)
        self.assertEqual(actions[12], 0)

        created: list[FakeEnvironment] = []

        def env_factory(**kwargs) -> FakeEnvironment:
            instance = FakeEnvironment(**kwargs)
            created.append(instance)
            return instance

        with tempfile.TemporaryDirectory() as temporary_dir:
            rows, config = evaluate_webster(
                episodes=1,
                episode_horizon_seconds=130,
                decision_interval_seconds=10,
                output_dir=Path(temporary_dir),
                env_factory=env_factory,
            )
            self.assertEqual(rows[0]["green_intervals_ns"], 7)
            self.assertEqual(rows[0]["green_intervals_ew"], 6)
            self.assertEqual(rows[0]["yellow_clearance_seconds_total"], 6.0)
            self.assertEqual(rows[0]["time_weighted_avg_queue_vehicles"], 7.0)
            self.assertEqual(rows[0]["throughput_completed_veh_per_min"], 1.2)
            self.assertEqual(rows[0]["termination_reason"], "horizon")
            self.assertEqual(config["lost_time_seconds_per_cycle"], 6.0)
            self.assertEqual(config["sumo_actuation_plan"]["steady_state_effective_green_ns_seconds"], 57.0)
            self.assertEqual(config["sumo_actuation_plan"]["steady_state_effective_green_ew_seconds"], 57.0)
            self.assertEqual(config["sumo_actuation_plan"]["cycle_wall_clock_seconds"], 120.0)
            self.assertTrue(created[0].closed)
            with (Path(temporary_dir) / "evaluation_metrics.csv").open(encoding="utf-8", newline="") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 1)
            saved_config = json.loads((Path(temporary_dir) / "webster_config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["arrival_rates_expected_per_interval_ns_ew"], [5.0, 5.0])

    def test_sumo_environment_closes_if_reset_or_step_raises(self) -> None:
        for failure in ("reset", "step"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary_dir:
                created: list[FakeEnvironment] = []

                def env_factory(**kwargs) -> FakeEnvironment:
                    env = FakeEnvironment(
                        **kwargs,
                        fail_on_reset=failure == "reset",
                        fail_on_step=failure == "step",
                    )
                    created.append(env)
                    return env

                with self.assertRaises(RuntimeError):
                    evaluate_webster(
                        episodes=1,
                        episode_horizon_seconds=130,
                        output_dir=Path(temporary_dir),
                        env_factory=env_factory,
                    )
                self.assertTrue(created[0].closed)


if __name__ == "__main__":
    unittest.main()
