"""Validate Part 3 results and produce Part 4 tables, tests, plots, and report."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from statistical_analysis import METRIC_COLUMNS, compare_controllers
from utils import (
    CONTROLLERS,
    PROJECT_DIR,
    SCENARIO_LABELS,
    aggregate_evaluation,
    ensure_output_dirs,
    read_csv,
    write_csv,
)


REQUIRED_COLUMNS = (
    "scenario", "controller", "episode", "seed", "avg_waiting_time_sec",
    "time_weighted_avg_queue", "throughput_veh_per_min", "total_arrived",
    "total_departed", "final_queue", "cumulative_reward",
)
METRIC_FIELDS = {
    "avg_waiting_time_sec": ("avg_waiting_time_sec", True),
    "time_weighted_avg_queue": ("time_weighted_avg_queue", True),
    "throughput_veh_per_min": ("throughput_veh_per_min", True),
    "final_queue": ("final_queue", True),
    "total_arrived": ("total_arrived", True),
    "total_departed": ("total_departed", True),
    "cumulative_reward": ("cumulative_reward", False),
}
SUMMARY_FIELDS = (
    "scenario", "controller",
    "valid_avg_waiting_time_observations", "mean_avg_waiting_time_sec",
    "median_avg_waiting_time_sec", "std_avg_waiting_time_sec",
    "min_avg_waiting_time_sec", "max_avg_waiting_time_sec",
    "p25_avg_waiting_time_sec", "p75_avg_waiting_time_sec",
    "ci95_avg_waiting_time_low", "ci95_avg_waiting_time_high",
    "valid_queue_observations", "mean_time_weighted_avg_queue",
    "median_time_weighted_avg_queue", "std_time_weighted_avg_queue",
    "ci95_time_weighted_avg_queue_low", "ci95_time_weighted_avg_queue_high",
    "valid_final_queue_observations", "mean_final_queue", "median_final_queue",
    "std_final_queue", "ci95_final_queue_low", "ci95_final_queue_high",
    "episodes_with_nonzero_final_queue_percent",
    "valid_throughput_observations", "mean_throughput_veh_per_min",
    "median_throughput_veh_per_min", "std_throughput_veh_per_min",
    "mean_total_arrived", "mean_total_departed", "mean_departure_to_arrival_ratio",
    "mean_cumulative_reward",
)
IMPROVEMENT_FIELDS = (
    "scenario", "rl_controller", "baseline_controller", "metric", "direction",
    "rl_mean", "baseline_mean", "absolute_difference", "improvement_percent",
)
RANKING_FIELDS = (
    "scenario", "metric", "direction", "rank", "controller", "mean", "std",
    "valid_observations",
)
STAT_FIELDS = (
    "scenario", "metric", "controller_a", "controller_b", "sample_size",
    "n_controller_a", "n_controller_b", "comparison_design", "test_used",
    "test_statistic", "raw_p_value", "holm_adjusted_p_value",
    "mean_difference_a_minus_b", "ci95_difference_low", "ci95_difference_high", "note",
)
METRIC_PLOT_SPECS = (
    ("avg_waiting_time_sec", "mean_avg_waiting_time_sec", "std_avg_waiting_time_sec", "Average waiting time", "Average waiting time (seconds)", "waiting_time_by_scenario.png", "lower"),
    ("time_weighted_avg_queue", "mean_time_weighted_avg_queue", "std_time_weighted_avg_queue", "Average time-weighted queue", "Queue length (vehicles)", "queue_length_by_scenario.png", "lower"),
    ("throughput_veh_per_min", "mean_throughput_veh_per_min", "std_throughput_veh_per_min", "Mean throughput", "Throughput (vehicles per minute)", "throughput_by_scenario.png", "higher"),
    ("final_queue", "mean_final_queue", "std_final_queue", "Mean final queue", "Vehicles remaining", "final_queue_by_scenario.png", "lower"),
    ("cumulative_reward", "mean_cumulative_reward", "", "Mean cumulative reward", "Cumulative reward", "reward_by_scenario.png", "higher"),
)


def _float(row: Mapping[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _finite_values(rows: Sequence[Mapping[str, object]], key: str, nonnegative: bool = True) -> np.ndarray:
    values = [_float(row, key) for row in rows]
    result = np.asarray([value for value in values if math.isfinite(value) and (not nonnegative or value >= 0)], dtype=float)
    return result


def _mean_std(values: np.ndarray) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _ci95(values: np.ndarray) -> Tuple[float, float]:
    """Student-t confidence interval for a mean (requires at least two values)."""
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    if sem == 0:
        return mean, mean
    margin = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean - margin, mean + margin


def validate_data(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    expected_episodes: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, object]], List[str]]:
    """Check required schema and contents; return report, analysis rows, exclusions."""
    issues: List[str] = []
    warnings: List[str] = []
    exclusions: List[str] = []
    if not rows:
        issues.append("Evaluation CSV contains no episode records.")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        issues.append("Missing required columns: " + ", ".join(missing_columns))
        return issues + ["Row-level checks could not be completed because the schema is incomplete."], [], exclusions

    duplicate_keys = set()
    seen = set()
    scenario_counts: Dict[Tuple[str, str], set] = {}
    scenario_seeds: Dict[Tuple[str, str], set] = {}
    seen_seed_keys = set()
    excluded_indexes = set()
    undefined_wait_count = 0
    for index, row in enumerate(rows, start=2):
        scenario = str(row.get("scenario", ""))
        controller = str(row.get("controller", ""))
        episode = str(row.get("episode", ""))
        seed = str(row.get("seed", ""))
        key = (scenario, controller, episode)
        if key in seen:
            duplicate_keys.add(key)
            excluded_indexes.add(index)
        seen.add(key)
        seed_key = (scenario, controller, seed)
        if seed_key in seen_seed_keys:
            issues.append(f"Row {index}: repeated seed {seed!r} for {scenario}/{controller}; paired matching may be ambiguous.")
        seen_seed_keys.add(seed_key)
        if controller not in CONTROLLERS:
            issues.append(f"Row {index}: unexpected controller name {controller!r}.")
            excluded_indexes.add(index)
        if scenario not in ("low", "medium", "high"):
            issues.append(f"Row {index}: unexpected scenario name {scenario!r}.")
            excluded_indexes.add(index)
        if not scenario or not controller or not episode or not seed:
            issues.append(f"Row {index}: missing scenario/controller/episode/seed key.")
            excluded_indexes.add(index)
        try:
            episode_number = int(float(episode))
            int(float(seed))
            if episode_number <= 0:
                raise ValueError
        except ValueError:
            issues.append(f"Row {index}: episode and seed must be valid integers.")
            excluded_indexes.add(index)
        if controller in CONTROLLERS and scenario in ("low", "medium", "high") and episode:
            scenario_counts.setdefault((scenario, controller), set()).add(episode)
            scenario_seeds.setdefault((scenario, controller), set()).add(seed)

        for field in METRIC_FIELDS:
            value = _float(row, field)
            if field == "avg_waiting_time_sec" and math.isnan(value):
                continue  # Undefined waiting time is allowed and stays NaN.
            if not math.isfinite(value):
                issues.append(f"Row {index}: {field} is missing or non-finite.")
            elif METRIC_FIELDS[field][1] and value < 0:
                issues.append(f"Row {index}: {field} is negative ({value}).")
        if math.isnan(_float(row, "avg_waiting_time_sec")):
            undefined_wait_count += 1

    for key in sorted(duplicate_keys):
        issues.append(f"Duplicate episode record: scenario/controller/episode={key}; later row(s) excluded from summaries.")
    scenarios = sorted({key[0] for key in scenario_counts})
    for scenario in scenarios:
        controllers_here = [controller for s, controller in scenario_counts if s == scenario]
        missing_controllers = sorted(set(CONTROLLERS) - set(controllers_here))
        if missing_controllers:
            issues.append(f"{scenario}: missing controller records for {', '.join(missing_controllers)}.")
        counts = {controller: len(scenario_counts[(scenario, controller)]) for controller in controllers_here}
        if len(set(counts.values())) > 1:
            issues.append(f"{scenario}: inconsistent episode counts across controllers: {counts}.")
        episode_sets = {controller: scenario_counts[(scenario, controller)] for controller in controllers_here}
        if len(set(map(frozenset, episode_sets.values()))) > 1:
            issues.append(f"{scenario}: controllers do not share the same episode-number set.")
        seed_sets = {controller: scenario_seeds[(scenario, controller)] for controller in controllers_here}
        if len(set(map(frozenset, seed_sets.values()))) > 1:
            issues.append(f"{scenario}: controllers do not share identical scenario/seed sets.")
        elif len(controllers_here) == len(CONTROLLERS):
            warnings.append(f"{scenario}: all controllers share {len(next(iter(seed_sets.values())))} scenario/seed combinations.")
        if expected_episodes is not None:
            for controller, count in counts.items():
                if count != expected_episodes:
                    issues.append(f"{scenario}/{controller}: observed {count} episodes; expected {expected_episodes}.")
    missing_scenarios = sorted({"low", "medium", "high"} - set(scenarios))
    if missing_scenarios:
        warnings.append("No records found for scenarios: " + ", ".join(missing_scenarios) + ".")
    for row_index, row in enumerate(rows, start=2):
        arrived = _float(row, "total_arrived")
        departed = _float(row, "total_departed")
        final = _float(row, "final_queue")
        if all(math.isfinite(value) for value in (arrived, departed, final)) and not math.isclose(arrived, departed + final, abs_tol=1e-8):
            issues.append(f"Row {row_index}: traffic accounting mismatch (arrived != departed + final_queue).")
    if undefined_wait_count:
        warnings.append(f"{undefined_wait_count} record(s) have undefined waiting time; they remain NaN and are excluded only from waiting-time summaries/tests.")
    if expected_episodes is None:
        warnings.append("No intended episode count was supplied; count consistency is reported, but the requested count cannot be verified.")

    analysis_rows: List[Dict[str, object]] = []
    for index, row in enumerate(rows, start=2):
        if index not in excluded_indexes:
            analysis_rows.append(dict(row))
        elif index not in exclusions:
            exclusions.append(f"CSV row {index} excluded from grouped analyses due to invalid key or duplicate episode.")

    report = [
        f"Evaluation records read: {len(rows)}",
        f"Required columns present: {not missing_columns}",
        f"Recognized controllers: {', '.join(CONTROLLERS)}",
        "Waiting-time NaNs are allowed as undefined values and are never replaced with zero.",
        f"Rows excluded from grouped analyses: {len(exclusions)}",
    ]
    report.extend("ISSUE: " + item for item in issues)
    report.extend("NOTE: " + item for item in warnings)
    if exclusions:
        report.extend("EXCLUSION: " + item for item in exclusions)
    if not issues:
        report.append("No structural or value-range errors were detected.")
    return report, analysis_rows, exclusions


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Create descriptive statistics for the report and summary CSV."""
    groups: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["scenario"]), str(row["controller"])), []).append(row)
    summaries: List[Dict[str, object]] = []
    for (scenario, controller), group in sorted(groups.items()):
        wait = _finite_values(group, "avg_waiting_time_sec")
        queue = _finite_values(group, "time_weighted_avg_queue")
        final = _finite_values(group, "final_queue")
        throughput = _finite_values(group, "throughput_veh_per_min")
        arrivals = _finite_values(group, "total_arrived")
        departures = _finite_values(group, "total_departed")
        rewards = _finite_values(group, "cumulative_reward", nonnegative=False)
        wait_mean, wait_std = _mean_std(wait)
        queue_mean, queue_std = _mean_std(queue)
        final_mean, final_std = _mean_std(final)
        throughput_mean, throughput_std = _mean_std(throughput)
        arrival_mean, _ = _mean_std(arrivals)
        departure_mean, _ = _mean_std(departures)
        reward_mean, _ = _mean_std(rewards)
        wait_low, wait_high = _ci95(wait)
        queue_low, queue_high = _ci95(queue)
        final_low, final_high = _ci95(final)
        ratios = [
            float(row["total_departed"]) / float(row["total_arrived"])
            for row in group
            if math.isfinite(_float(row, "total_departed"))
            and math.isfinite(_float(row, "total_arrived"))
            and float(row["total_arrived"]) > 0
            and float(row["total_departed"]) >= 0
        ]
        summaries.append({
            "scenario": scenario,
            "controller": controller,
            "valid_avg_waiting_time_observations": len(wait),
            "mean_avg_waiting_time_sec": wait_mean,
            "median_avg_waiting_time_sec": float(np.median(wait)) if len(wait) else float("nan"),
            "std_avg_waiting_time_sec": wait_std,
            "min_avg_waiting_time_sec": float(np.min(wait)) if len(wait) else float("nan"),
            "max_avg_waiting_time_sec": float(np.max(wait)) if len(wait) else float("nan"),
            "p25_avg_waiting_time_sec": float(np.percentile(wait, 25)) if len(wait) else float("nan"),
            "p75_avg_waiting_time_sec": float(np.percentile(wait, 75)) if len(wait) else float("nan"),
            "ci95_avg_waiting_time_low": wait_low,
            "ci95_avg_waiting_time_high": wait_high,
            "valid_queue_observations": len(queue),
            "mean_time_weighted_avg_queue": queue_mean,
            "median_time_weighted_avg_queue": float(np.median(queue)) if len(queue) else float("nan"),
            "std_time_weighted_avg_queue": queue_std,
            "ci95_time_weighted_avg_queue_low": queue_low,
            "ci95_time_weighted_avg_queue_high": queue_high,
            "valid_final_queue_observations": len(final),
            "mean_final_queue": final_mean,
            "median_final_queue": float(np.median(final)) if len(final) else float("nan"),
            "std_final_queue": final_std,
            "ci95_final_queue_low": final_low,
            "ci95_final_queue_high": final_high,
            "episodes_with_nonzero_final_queue_percent": (
                100.0 * float(np.mean(final > 0)) if len(final) else float("nan")
            ),
            "valid_throughput_observations": len(throughput),
            "mean_throughput_veh_per_min": throughput_mean,
            "median_throughput_veh_per_min": float(np.median(throughput)) if len(throughput) else float("nan"),
            "std_throughput_veh_per_min": throughput_std,
            "mean_total_arrived": arrival_mean,
            "mean_total_departed": departure_mean,
            "mean_departure_to_arrival_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "mean_cumulative_reward": reward_mean,
        })
    return summaries


def _safe_improvement(rl_mean: float, baseline_mean: float, lower_is_better: bool) -> float:
    if not math.isfinite(rl_mean) or not math.isfinite(baseline_mean) or baseline_mean == 0:
        return float("nan")
    return (baseline_mean - rl_mean) / baseline_mean * 100 if lower_is_better else (rl_mean - baseline_mean) / baseline_mean * 100


def improvement_rows(summary: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Compute signed RL-minus-Webster differences and directional improvements."""
    mean_columns = {
        "avg_waiting_time_sec": ("mean_avg_waiting_time_sec", True),
        "time_weighted_avg_queue": ("mean_time_weighted_avg_queue", True),
        "final_queue": ("mean_final_queue", True),
        "throughput_veh_per_min": ("mean_throughput_veh_per_min", False),
    }
    results = []
    for scenario in sorted({str(row["scenario"]) for row in summary}):
        baseline = next((row for row in summary if row["scenario"] == scenario and row["controller"] == "Webster"), None)
        if baseline is None:
            continue
        for controller in ("Q-Learning", "DQN"):
            candidate = next((row for row in summary if row["scenario"] == scenario and row["controller"] == controller), None)
            if candidate is None:
                continue
            for metric, (column, lower_is_better) in mean_columns.items():
                rl_mean = _float(candidate, column)
                baseline_mean = _float(baseline, column)
                difference = rl_mean - baseline_mean if math.isfinite(rl_mean) and math.isfinite(baseline_mean) else float("nan")
                results.append({
                    "scenario": scenario,
                    "rl_controller": controller,
                    "baseline_controller": "Webster",
                    "metric": metric,
                    "direction": "lower is better" if lower_is_better else "higher is better",
                    "rl_mean": rl_mean,
                    "baseline_mean": baseline_mean,
                    "absolute_difference": difference,
                    "improvement_percent": _safe_improvement(rl_mean, baseline_mean, lower_is_better),
                })
    return results


def ranking_rows(summary: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Rank each controller independently by scenario and metric direction."""
    metrics = (
        ("avg_waiting_time_sec", "mean_avg_waiting_time_sec", "std_avg_waiting_time_sec", "valid_avg_waiting_time_observations", "lower"),
        ("time_weighted_avg_queue", "mean_time_weighted_avg_queue", "std_time_weighted_avg_queue", "valid_queue_observations", "lower"),
        ("final_queue", "mean_final_queue", "std_final_queue", "valid_final_queue_observations", "lower"),
        ("throughput_veh_per_min", "mean_throughput_veh_per_min", "std_throughput_veh_per_min", "valid_throughput_observations", "higher"),
        ("cumulative_reward", "mean_cumulative_reward", "", "", "higher"),
    )
    result: List[Dict[str, object]] = []
    scenarios = sorted({str(row["scenario"]) for row in summary})
    for scenario in scenarios:
        group = [row for row in summary if row["scenario"] == scenario]
        for metric, mean_key, std_key, n_key, direction in metrics:
            vals = [(row, _float(row, mean_key)) for row in group]
            valid = [(row, value) for row, value in vals if math.isfinite(value)]
            valid.sort(key=lambda pair: pair[1], reverse=(direction == "higher"))
            ranks: Dict[str, float] = {}
            position = 0
            while position < len(valid):
                end = position + 1
                while end < len(valid) and valid[end][1] == valid[position][1]:
                    end += 1
                average_rank = (position + 1 + end) / 2.0
                for row, _ in valid[position:end]:
                    ranks[str(row["controller"])] = average_rank
                position = end
            for row, mean in vals:
                result.append({
                    "scenario": scenario,
                    "metric": metric,
                    "direction": direction,
                    "rank": ranks.get(str(row["controller"]), float("nan")),
                    "controller": row["controller"],
                    "mean": mean,
                    "std": _float(row, std_key) if std_key else float("nan"),
                    "valid_observations": row.get(n_key, ""),
                })
    return result


def _bar_plot(summary: Sequence[Mapping[str, object]], root: Path, spec: tuple) -> None:
    _, mean_key, std_key, title, ylabel, filename, _ = spec
    scenarios = ("low", "medium", "high")
    x = np.arange(len(scenarios), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5.2))
    plotted = False
    for controller_index, controller in enumerate(CONTROLLERS):
        means = []
        errors = []
        valid_ns = []
        for scenario in scenarios:
            match = next((row for row in summary if row["scenario"] == scenario and row["controller"] == controller), None)
            means.append(_float(match, mean_key) if match else float("nan"))
            errors.append(_float(match, std_key) if match and std_key else float("nan"))
            n_key = "valid_avg_waiting_time_observations" if "waiting" in mean_key else (
                "valid_queue_observations" if "queue" in mean_key and "final" not in mean_key else
                "valid_final_queue_observations" if "final" in mean_key else
                "valid_throughput_observations" if "throughput" in mean_key else "valid_queue_observations"
            )
            valid_ns.append(_float(match, n_key) if match else 0)
        offset = (controller_index - 1) * width
        positions = x + offset
        axis.bar(positions, means, width, label=controller)
        for pos, mean, error, n in zip(positions, means, errors, valid_ns):
            if math.isfinite(mean) and math.isfinite(error) and n >= 2:
                axis.errorbar(pos, mean, yerr=error, fmt="none", capsize=3, color="black", linewidth=1)
                plotted = True
        plotted = plotted or any(math.isfinite(value) for value in means)
    axis.set_title(title)
    axis.set_xlabel("Traffic demand scenario")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, [SCENARIO_LABELS[item] for item in scenarios])
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No valid evaluation data", ha="center", va="center", transform=axis.transAxes)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "plots" / filename, dpi=180)
    plt.close(fig)


def _heatmap(summary: Sequence[Mapping[str, object]], root: Path) -> None:
    metric_specs = (
        ("avg_waiting_time_sec", "mean_avg_waiting_time_sec", "Wait", "lower"),
        ("time_weighted_avg_queue", "mean_time_weighted_avg_queue", "Queue", "lower"),
        ("throughput_veh_per_min", "mean_throughput_veh_per_min", "Flow", "higher"),
        ("final_queue", "mean_final_queue", "Final Q", "lower"),
        ("cumulative_reward", "mean_cumulative_reward", "Reward", "higher"),
    )
    columns = [(scenario, metric) for scenario in ("low", "medium", "high") for metric in metric_specs]
    matrix = np.full((len(CONTROLLERS), len(columns)), np.nan)
    labels = []
    for column_index, (scenario, metric_spec) in enumerate(columns):
        metric, mean_key, short_label, direction = metric_spec
        labels.append(f"{SCENARIO_LABELS[scenario].replace(' demand', '')}\n{short_label}")
        raw = []
        for controller in CONTROLLERS:
            row = next((r for r in summary if r["scenario"] == scenario and r["controller"] == controller), None)
            raw.append(_float(row, mean_key) if row else float("nan"))
        valid = [value for value in raw if math.isfinite(value)]
        if not valid:
            continue
        low, high = min(valid), max(valid)
        for controller_index, value in enumerate(raw):
            if math.isfinite(value):
                score = 0.5 if high == low else (value - low) / (high - low)
                matrix[controller_index, column_index] = score if direction == "higher" else 1.0 - score
    fig, axis = plt.subplots(figsize=(17, 4.5))
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_yticks(range(len(CONTROLLERS)), CONTROLLERS)
    axis.set_xticks(range(len(labels)), labels, rotation=0, fontsize=8)
    axis.set_title("Within-scenario normalized performance (best observed = 1)")
    axis.set_xlabel("Scenario and metric (normalization is within each column)")
    if not np.isfinite(matrix).any():
        axis.text(0.5, 0.5, "No valid evaluation data", ha="center", va="center", transform=axis.transAxes, color="white")
    fig.colorbar(image, ax=axis, label="Relative score (higher is better)")
    fig.tight_layout()
    fig.savefig(root / "plots" / "performance_heatmap.png", dpi=180)
    plt.close(fig)


def _training_convergence(root: Path) -> Tuple[bool, str]:
    files = sorted((root / "results").glob("*_training.csv"))
    fig, axis = plt.subplots(figsize=(9, 5.2))
    plotted = False
    for path in files:
        if not path.name.startswith(("q_learning_", "dqn_")):
            continue
        rows = read_csv(path)
        if not rows or not {"episode", "cumulative_reward"}.issubset(rows[0]):
            continue
        episodes = np.asarray([_float(row, "episode") for row in rows])
        rewards = np.asarray([_float(row, "cumulative_reward") for row in rows])
        keep = np.isfinite(episodes) & np.isfinite(rewards)
        episodes, rewards = episodes[keep], rewards[keep]
        if not len(rewards):
            continue
        window = min(100, len(rewards))
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        algo = "Q-learning" if path.name.startswith("q_learning_") else "DQN"
        scenario = rows[0].get("scenario", "unknown")
        axis.plot(episodes[window - 1 :], rolling, label=f"{algo} / {scenario}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return False, "Skipped training_convergence.png: no usable Part 3 Q-learning/DQN training CSV logs were found."
    axis.set_title("Training reward convergence (rolling mean)")
    axis.set_xlabel("Episode")
    axis.set_ylabel("100-episode rolling mean cumulative reward")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "plots" / "training_convergence.png", dpi=180)
    plt.close(fig)
    return True, "Generated training_convergence.png from Part 3 training logs."


def _training_observations(root: Path) -> List[str]:
    """Summarize first/last rolling-mean rewards from available training logs."""
    observations = []
    for path in sorted((root / "results").glob("*_training.csv")):
        if not path.name.startswith(("q_learning_", "dqn_")):
            continue
        rows = read_csv(path)
        rewards = np.asarray([_float(row, "cumulative_reward") for row in rows])
        rewards = rewards[np.isfinite(rewards)]
        if not len(rewards):
            continue
        window = min(100, len(rewards))
        start_mean = float(np.mean(rewards[:window]))
        end_mean = float(np.mean(rewards[-window:]))
        algorithm = "Q-learning" if path.name.startswith("q_learning_") else "DQN"
        scenario = rows[0].get("scenario", "unknown") if rows else "unknown"
        observations.append(
            f"{algorithm}/{scenario}: first {window}-episode mean reward {start_mean:.3f}; "
            f"last {window}-episode mean reward {end_mean:.3f}; observed change {end_mean - start_mean:+.3f}. "
            "This trend alone does not establish stable convergence."
        )
    if not observations:
        observations.append("No usable Q-learning or DQN Part 3 training CSV logs were found; convergence could not be assessed.")
    return observations


def generate_analysis_plots(summary: Sequence[Mapping[str, object]], root: Path) -> List[str]:
    """Generate the six evaluation comparisons, heatmap, and optional convergence plot."""
    plot_messages = []
    if summary:
        for spec in METRIC_PLOT_SPECS:
            _bar_plot(summary, root, spec)
            plot_messages.append(f"Generated {spec[5]}.")
        _heatmap(summary, root)
        plot_messages.append("Generated performance_heatmap.png; scores are min-max normalized within each scenario/metric, with direction inverted for lower-is-better measures.")
    else:
        plot_messages.append("Skipped evaluation comparison plots: no valid evaluation summaries are available.")
    _, convergence_message = _training_convergence(root)
    plot_messages.append(convergence_message)
    return plot_messages


def _table(rows: Sequence[Mapping[str, object]], columns: Sequence[str], limit: Optional[int] = None) -> str:
    if not rows:
        return "_No data available._"
    shown = list(rows[:limit]) if limit else list(rows)
    header = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in shown:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                text = "undefined" if not math.isfinite(value) else f"{value:.3f}"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header] + body)


def build_analysis_report(
    validation_report: Sequence[str],
    summaries: Sequence[Mapping[str, object]],
    improvements: Sequence[Mapping[str, object]],
    tests: Sequence[Mapping[str, object]],
    rankings: Sequence[Mapping[str, object]],
    plot_messages: Sequence[str],
    has_evaluation: bool,
    training_notes: Sequence[str] = (),
) -> str:
    """Build a report with data-dependent claims only when evaluation exists."""
    lines = [
        "# Comparative Analysis Report",
        "",
        "## 1. Project objective",
        "Compare tabular Q-learning, DQN, and Webster's fixed-time baseline in a custom two-direction traffic simulation.",
        "",
        "## 2. Simulation scenarios and experimental setup",
        "The configured scenarios use expected Poisson arrivals per interval: low (NS 1.0, EW 0.8), medium (NS 2.8, EW 1.2), and high (NS 4.0, EW 3.5). These are synthetic assumptions, not real-world measurements. Part 3 uses common evaluation seeds and the same episode configuration for controllers.",
        "",
        "## 3. Data-quality findings",
        *[f"- {line}" for line in validation_report],
        "",
    ]
    if any(line.startswith("ISSUE:") or line.startswith("NOTE: No records found for scenarios") for line in validation_report):
        lines.extend([
            "**Caution:** The validation report contains data-quality issues. Numerical summaries below are based on retained records but should be treated as incomplete or potentially unreliable until the source data is corrected or regenerated.",
            "",
        ])
    if not has_evaluation:
        lines.extend([
            "## 4. Analysis status",
            "No Part 3 `results/evaluation_results.csv` was present when this analysis ran. No empirical evaluation means, improvements, statistical tests, rankings, or controller-comparison plots have been generated. Training convergence may still be plotted if its Part 3 training logs are present. Run Part 3 evaluation, then rerun `python analyze_results.py`.",
            "",
            "## 5. Training convergence",
            *[f"- {note}" for note in training_notes],
            "",
            "## 6. Metrics and interpretation plan",
            "Waiting time is the mean for vehicles that departed; it can be undefined when none departed and may look low if many vehicles remain queued. Interpret it together with queue length, throughput, departures, and final queue. Departure-to-arrival ratio is calculated per episode before averaging.",
            "",
            "## 7. Limitations",
            "The simulator uses synthetic Poisson demand, one queue per direction, fixed service capacity, and no turns, lanes, pedestrians, yellow/all-red intervals, or real traffic calibration. Webster receives configured demand as prior information while the RL agents learn from reward feedback. This experiment cannot establish real-world traffic performance without validation against realistic traffic data.",
            "",
            "## 8. Reproducibility",
            "The analysis reads saved CSVs only and does not retrain agents. Statistical comparisons use matched scenario/seed pairs when available, pairwise metric-specific exclusions, and Holm-adjusted p-values within each scenario/metric family. Mean-difference confidence intervals use a t approximation and should be interpreted with sample size and distribution assumptions in mind.",
            "",
        ])
        return "\n".join(lines)

    lines.extend([
        "## 4. Average waiting time",
        "This metric averages waits only for vehicles that departed. It is interpreted alongside throughput, queue, and final-queue results.",
        "",
        _table(summaries, ("scenario", "controller", "valid_avg_waiting_time_observations", "mean_avg_waiting_time_sec", "median_avg_waiting_time_sec", "std_avg_waiting_time_sec", "p25_avg_waiting_time_sec", "p75_avg_waiting_time_sec", "ci95_avg_waiting_time_low", "ci95_avg_waiting_time_high")),
        "",
        "The 95% confidence interval uses a Student-t interval for the mean when at least two finite observations are available; it assumes independent episode observations and an approximately suitable sampling distribution. Undefined waiting values remain excluded and the valid count is shown.",
        "",
        "## 5. Queue length",
        _table(summaries, ("scenario", "controller", "mean_time_weighted_avg_queue", "median_time_weighted_avg_queue", "std_time_weighted_avg_queue", "ci95_time_weighted_avg_queue_low", "ci95_time_weighted_avg_queue_high", "mean_final_queue", "median_final_queue", "episodes_with_nonzero_final_queue_percent")),
        "",
        "Lower waiting queues generally indicate less congestion, but queue metrics are not a standalone measure of controller quality.",
        "",
        "## 6. Throughput and service",
        _table(summaries, ("scenario", "controller", "mean_throughput_veh_per_min", "median_throughput_veh_per_min", "std_throughput_veh_per_min", "mean_total_arrived", "mean_total_departed", "mean_final_queue", "mean_departure_to_arrival_ratio")),
        "",
        "Departure-to-arrival ratio is computed episode by episode as total departures / total arrivals, excluding episodes with zero arrivals, then averaged. Throughput is interpreted with final queue and total service.",
        "",
        "## 7. Improvements relative to Webster",
        "Absolute difference is RL mean minus Webster mean. Percentage improvement uses `(Webster - RL) / Webster * 100` for lower-is-better metrics, and `(RL - Webster) / Webster * 100` for throughput. A zero or undefined Webster mean yields an undefined percentage.",
        "",
        _table(improvements, ("scenario", "rl_controller", "metric", "rl_mean", "baseline_mean", "absolute_difference", "improvement_percent")),
        "",
        "## 8. Statistical comparisons",
        "Tests match rows by scenario and seed where at least two valid pairs are available. Paired t-tests are used when Shapiro-Wilk does not flag the differences as non-normal; otherwise the Wilcoxon signed-rank test is used. Unmatched samples use Welch's t-test when both groups pass the same normality screen, otherwise Mann-Whitney U. Holm correction is applied across the three controller-pair tests within each scenario/metric family. A p-value is evidence against equal distributions under test assumptions; it does not measure practical importance.",
        "",
        _table(tests, ("scenario", "metric", "controller_a", "controller_b", "sample_size", "comparison_design", "test_used", "test_statistic", "raw_p_value", "holm_adjusted_p_value", "mean_difference_a_minus_b", "ci95_difference_low", "ci95_difference_high", "note")),
        "",
        "## 9. Controller rankings by scenario and metric",
        "Rank 1 is best for the listed metric direction; ties receive average ranks. No single overall winner is calculated.",
        "",
        _table(rankings, ("scenario", "metric", "direction", "rank", "controller", "mean", "std", "valid_observations")),
        "",
        "## 10. Scenario observations",
        *scenario_observations(summaries),
        "",
        "## 11. Training convergence",
        *[f"- {note}" for note in training_notes],
        "",
        "## 12. Limitations and interpretation",
        "Results describe this simplified, synthetic environment only. The simulator has no lanes, turning movements, pedestrian demand, yellow/all-red phases, or calibrated arrival/service data. Webster receives configured demand as prior information, whereas Q-learning and DQN learn from reward feedback. Statistical significance does not imply a practically important traffic benefit. Real-world claims require validation using realistic traffic data and a calibrated traffic simulator.",
        "",
        "## 13. Generated visualizations",
        *[f"- {message}" for message in plot_messages],
        "",
    ])
    return "\n".join(lines)


def scenario_observations(summaries: Sequence[Mapping[str, object]]) -> List[str]:
    """Describe observed sample-mean extrema without calling them superiority."""
    lines = []
    for scenario in ("low", "medium", "high"):
        group = [row for row in summaries if row["scenario"] == scenario]
        if not group:
            lines.append(f"- {SCENARIO_LABELS[scenario]}: no summary data available.")
            continue
        wait = [(str(r["controller"]), _float(r, "mean_avg_waiting_time_sec")) for r in group]
        wait = [(name, value) for name, value in wait if math.isfinite(value)]
        queue = [(str(r["controller"]), _float(r, "mean_time_weighted_avg_queue")) for r in group]
        queue = [(name, value) for name, value in queue if math.isfinite(value)]
        if wait:
            name, value = min(wait, key=lambda item: item[1])
            lines.append(f"- {SCENARIO_LABELS[scenario]}: {name} had the lowest observed mean departed-vehicle waiting time ({value:.3f} seconds); consider this with queue and service metrics.")
        else:
            lines.append(f"- {SCENARIO_LABELS[scenario]}: waiting-time means are undefined for all available controllers.")
        if queue:
            name, value = min(queue, key=lambda item: item[1])
            lines.append(f"  {name} had the lowest observed time-weighted average queue ({value:.3f} vehicles). These descriptive extrema are not claims of statistical superiority.")
    return lines


def _write_empty_outputs(results_dir: Path) -> None:
    write_csv(results_dir / "summary_results.csv", [], SUMMARY_FIELDS)
    write_csv(results_dir / "statistical_tests.csv", [], STAT_FIELDS)
    write_csv(results_dir / "improvement_summary.csv", [], IMPROVEMENT_FIELDS)
    write_csv(results_dir / "controller_rankings.csv", [], RANKING_FIELDS)


def analyze_project(root: Path = PROJECT_DIR, expected_episodes: Optional[int] = None) -> Dict[str, object]:
    """Run validation and all analyses that the available evaluation CSV permits."""
    directories = ensure_output_dirs(root)
    results_dir = directories["results"]
    evaluation_path = results_dir / "evaluation_results.csv"
    if not evaluation_path.exists():
        validation = [
            "Evaluation records read: 0",
            "ISSUE: results/evaluation_results.csv is missing.",
            "No episode count, pairing, or metric checks are possible until Part 3 evaluation runs.",
        ]
        (results_dir / "data_validation_report.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")
        _write_empty_outputs(results_dir)
        generated, convergence_message = _training_convergence(root)
        plot_messages = ["Evaluation comparison plots skipped because evaluation data is absent.", convergence_message]
        training_notes = _training_observations(root)
        report = build_analysis_report(validation, [], [], [], [], plot_messages, False, training_notes)
        (results_dir / "analysis_report.md").write_text(report, encoding="utf-8")
        return {"has_evaluation": False, "records": 0, "validation": validation, "plot_messages": plot_messages, "training_plot_generated": generated}

    with evaluation_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        rows = list(reader)
    validation, analysis_rows, exclusions = validate_data(rows, columns, expected_episodes)
    (results_dir / "data_validation_report.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")
    if not set(REQUIRED_COLUMNS).issubset(columns):
        _write_empty_outputs(results_dir)
        _, convergence_message = _training_convergence(root)
        plot_messages = ["Evaluation comparison plots skipped because required evaluation columns are missing.", convergence_message]
        training_notes = _training_observations(root)
        report = build_analysis_report(validation, [], [], [], [], plot_messages, False, training_notes)
        (results_dir / "analysis_report.md").write_text(report, encoding="utf-8")
        return {"has_evaluation": True, "records": len(rows), "validation": validation, "plot_messages": plot_messages}

    summaries = summarize_rows(analysis_rows)
    improvements = improvement_rows(summaries)
    rankings = ranking_rows(summaries)
    tests = compare_controllers(analysis_rows)
    write_csv(results_dir / "summary_results.csv", summaries, SUMMARY_FIELDS)
    write_csv(results_dir / "improvement_summary.csv", improvements, IMPROVEMENT_FIELDS)
    write_csv(results_dir / "controller_rankings.csv", rankings, RANKING_FIELDS)
    write_csv(results_dir / "statistical_tests.csv", tests, STAT_FIELDS)
    plot_messages = generate_analysis_plots(summaries, root)
    training_notes = _training_observations(root)
    if exclusions:
        training_notes.append(f"{len(exclusions)} structurally invalid or duplicate record(s) were omitted from grouped analyses; see data_validation_report.txt.")
    report = build_analysis_report(validation, summaries, improvements, tests, rankings, plot_messages, True, training_notes)
    (results_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    return {
        "has_evaluation": True,
        "records": len(rows),
        "summaries": summaries,
        "improvements": improvements,
        "rankings": rankings,
        "statistical_tests": tests,
        "plot_messages": plot_messages,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--expected-episodes", type=int, default=None)
    args = parser.parse_args()
    outcome = analyze_project(args.output_dir, args.expected_episodes)
    if not outcome["has_evaluation"]:
        print("No evaluation CSV found; wrote a validation report and analysis template without empirical findings.")
    else:
        print(f"Analyzed {outcome['records']} evaluation records; see {args.output_dir / 'results'} and {args.output_dir / 'plots'}.")


if __name__ == "__main__":
    main()
