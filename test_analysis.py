"""Deterministic tests for Part 4 analysis helpers; fixtures are synthetic only."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from analyze_results import (
    IMPROVEMENT_FIELDS,
    REQUIRED_COLUMNS,
    analyze_project,
    improvement_rows,
    ranking_rows,
    summarize_rows,
    validate_data,
)
from statistical_analysis import _holm_adjust, compare_controllers
from utils import read_csv, write_csv


def fixture_rows():
    """Return explicitly synthetic rows for unit tests, never project findings."""
    rows = []
    samples = {
        "Q-Learning": [(2.0, 1.0, 5.0, 0), (4.0, 3.0, 7.0, 2), (float("nan"), 2.0, 6.0, 1)],
        "DQN": [(3.0, 2.0, 6.0, 1), (5.0, 4.0, 8.0, 3), (6.0, 3.0, 7.0, 2)],
        "Webster": [(5.0, 4.0, 8.0, 3), (7.0, 6.0, 10.0, 5), (9.0, 5.0, 9.0, 4)],
    }
    for controller, values in samples.items():
        for index, (wait, queue, departed, final_queue) in enumerate(values):
            arrivals = departed + final_queue
            rows.append({
                "scenario": "low",
                "controller": controller,
                "episode": index + 1,
                "seed": 100 + index,
                "avg_waiting_time_sec": wait,
                "time_weighted_avg_queue": queue,
                "throughput_veh_per_min": departed / 3.0,
                "total_arrived": arrivals,
                "total_departed": departed,
                "final_queue": final_queue,
                "cumulative_reward": -queue,
            })
    return rows


class ValidationTests(unittest.TestCase):
    def test_missing_columns_are_detected(self):
        report, rows, _ = validate_data([], ["scenario", "controller"])
        self.assertTrue(any("Missing required columns" in line for line in report))
        self.assertEqual(rows, [])

    def test_nan_waiting_time_is_allowed_and_not_zeroed(self):
        source = fixture_rows()
        report, kept, _ = validate_data(source, REQUIRED_COLUMNS, expected_episodes=3)
        self.assertEqual(len(kept), len(source))
        summary = summarize_rows(kept)
        q = next(row for row in summary if row["controller"] == "Q-Learning")
        self.assertEqual(q["valid_avg_waiting_time_observations"], 2)
        self.assertEqual(q["mean_avg_waiting_time_sec"], 3.0)
        self.assertTrue(any("undefined waiting time" in line for line in report))

    def test_duplicate_episode_is_reported_and_excluded(self):
        rows = fixture_rows()
        rows.append(dict(rows[0]))
        report, kept, excluded = validate_data(rows, REQUIRED_COLUMNS)
        self.assertEqual(len(kept), len(rows) - 1)
        self.assertTrue(excluded)
        self.assertTrue(any("Duplicate episode record" in line for line in report))

    def test_bad_values_are_reported(self):
        rows = fixture_rows()
        rows[0]["final_queue"] = -1
        report, _, _ = validate_data(rows, REQUIRED_COLUMNS)
        self.assertTrue(any("is negative" in line for line in report))
        self.assertTrue(any("traffic accounting mismatch" in line for line in report))

    def test_misaligned_episode_numbers_are_reported(self):
        rows = fixture_rows()
        next(row for row in rows if row["controller"] == "Webster" and row["episode"] == 1)["episode"] = 10
        report, _, _ = validate_data(rows, REQUIRED_COLUMNS)
        self.assertTrue(any("episode-number set" in line for line in report))


class SummaryTests(unittest.TestCase):
    def test_small_summary_statistics_and_episode_ratio(self):
        rows = fixture_rows()
        summary = summarize_rows(rows)
        q = next(row for row in summary if row["controller"] == "Q-Learning")
        self.assertEqual(q["mean_avg_waiting_time_sec"], 3.0)
        self.assertEqual(q["median_avg_waiting_time_sec"], 3.0)
        self.assertEqual(q["std_avg_waiting_time_sec"], 1.0)
        self.assertEqual(q["min_avg_waiting_time_sec"], 2.0)
        self.assertEqual(q["max_avg_waiting_time_sec"], 4.0)
        self.assertEqual(q["mean_time_weighted_avg_queue"], 2.0)
        self.assertAlmostEqual(q["mean_departure_to_arrival_ratio"], (1 + 7 / 9 + 6 / 7) / 3)
        self.assertEqual(q["episodes_with_nonzero_final_queue_percent"], 2 / 3 * 100)

    def test_improvement_formula_and_zero_baseline(self):
        summary = [
            {"scenario": "low", "controller": "Q-Learning", "mean_avg_waiting_time_sec": 6,
             "mean_time_weighted_avg_queue": 2, "mean_final_queue": 1,
             "mean_throughput_veh_per_min": 12},
            {"scenario": "low", "controller": "DQN", "mean_avg_waiting_time_sec": 5,
             "mean_time_weighted_avg_queue": 1, "mean_final_queue": 0,
             "mean_throughput_veh_per_min": 10},
            {"scenario": "low", "controller": "Webster", "mean_avg_waiting_time_sec": 10,
             "mean_time_weighted_avg_queue": 4, "mean_final_queue": 2,
             "mean_throughput_veh_per_min": 8},
        ]
        improvements = improvement_rows(summary)
        q_wait = next(row for row in improvements if row["rl_controller"] == "Q-Learning" and row["metric"] == "avg_waiting_time_sec")
        self.assertEqual(q_wait["absolute_difference"], -4)
        self.assertEqual(q_wait["improvement_percent"], 40)
        webster_zero = [dict(row) for row in summary]
        next(row for row in webster_zero if row["controller"] == "Webster")["mean_avg_waiting_time_sec"] = 0
        undefined = next(row for row in improvement_rows(webster_zero) if row["metric"] == "avg_waiting_time_sec")
        self.assertTrue(math.isnan(undefined["improvement_percent"]))
        self.assertEqual(set(improvements[0]), set(IMPROVEMENT_FIELDS))

    def test_rankings_follow_each_metric_direction(self):
        summaries = summarize_rows(fixture_rows())
        ranks = ranking_rows(summaries)
        wait_ranks = {r["controller"]: r["rank"] for r in ranks if r["metric"] == "avg_waiting_time_sec"}
        throughput_ranks = {r["controller"]: r["rank"] for r in ranks if r["metric"] == "throughput_veh_per_min"}
        self.assertEqual(wait_ranks["Q-Learning"], 1)
        self.assertEqual(wait_ranks["Webster"], 3)
        self.assertEqual(throughput_ranks["Webster"], 1)


class StatisticalTests(unittest.TestCase):
    def test_pairs_are_matched_by_seed_and_sample_size_is_reported(self):
        rows = fixture_rows()
        for row in rows:
            if row["controller"] == "Webster":
                row["episode"] = int(row["episode"]) + 20
        tests = compare_controllers(rows)
        result = next(row for row in tests if row["metric"] == "final_queue" and row["controller_a"] == "Q-Learning" and row["controller_b"] == "Webster")
        self.assertEqual(result["sample_size"], 3)
        self.assertEqual(result["comparison_design"], "paired by seed")
        self.assertTrue(math.isfinite(result["raw_p_value"]))
        self.assertTrue(math.isfinite(result["holm_adjusted_p_value"]))

    def test_holm_adjustment_is_monotone_and_correct(self):
        rows = [
            {"scenario": "low", "metric": "x", "raw_p_value": 0.01},
            {"scenario": "low", "metric": "x", "raw_p_value": 0.04},
            {"scenario": "low", "metric": "x", "raw_p_value": 0.03},
        ]
        _holm_adjust(rows)
        self.assertEqual(rows[0]["holm_adjusted_p_value"], 0.03)
        self.assertEqual(rows[2]["holm_adjusted_p_value"], 0.06)
        self.assertEqual(rows[1]["holm_adjusted_p_value"], 0.06)

    def test_nonmatched_records_are_not_mislabeled_as_paired(self):
        rows = fixture_rows()
        webster_seeds = iter((102, 200, 201))
        for row in rows:
            if row["controller"] == "Webster":
                row["seed"] = next(webster_seeds)
        tests = compare_controllers(rows)
        result = next(row for row in tests if row["metric"] == "final_queue" and row["controller_a"] == "Q-Learning" and row["controller_b"] == "Webster")
        self.assertEqual(result["comparison_design"], "independent samples")


class ProjectAnalysisTests(unittest.TestCase):
    def test_missing_real_data_writes_template_without_findings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome = analyze_project(root)
            report = (root / "results" / "analysis_report.md").read_text()
            self.assertFalse(outcome["has_evaluation"])
            self.assertIn("No Part 3", report)
            self.assertNotIn("had the lowest observed", report)
            self.assertTrue((root / "results" / "data_validation_report.txt").is_file())
            self.assertTrue((root / "results" / "statistical_tests.csv").is_file())

    def test_valid_synthetic_fixture_generates_tables_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            write_csv(results / "evaluation_results.csv", fixture_rows(), REQUIRED_COLUMNS)
            outcome = analyze_project(root, expected_episodes=3)
            self.assertEqual(outcome["records"], 9)
            for filename in ("summary_results.csv", "improvement_summary.csv", "controller_rankings.csv", "statistical_tests.csv", "analysis_report.md"):
                self.assertTrue((results / filename).is_file())
            self.assertEqual(len(list((root / "plots").glob("*.png"))), 6)
            self.assertTrue(any("no usable Part 3" in message for message in outcome["plot_messages"]))


if __name__ == "__main__":
    unittest.main()
