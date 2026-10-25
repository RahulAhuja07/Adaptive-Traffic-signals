"""Paired and independent statistical comparisons for controller episodes."""

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


METRIC_COLUMNS = {
    "avg_waiting_time_sec": "Average waiting time (seconds)",
    "time_weighted_avg_queue": "Time-weighted average queue",
    "throughput_veh_per_min": "Throughput (vehicles/minute)",
    "final_queue": "Final queue (vehicles)",
}
CONTROLLER_PAIRS = (
    ("Q-Learning", "Webster"),
    ("DQN", "Webster"),
    ("Q-Learning", "DQN"),
)


def _number(row: Mapping[str, object], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    if not math.isfinite(value):
        return float("nan")
    return value


def _episode_values(
    rows: Sequence[Mapping[str, object]], scenario: str, controller: str, metric: str
) -> Dict[str, float]:
    """Collect one value per unique evaluation seed; duplicates are omitted."""
    values: Dict[str, float] = {}
    duplicate_seeds = set()
    for row in rows:
        if str(row.get("scenario")) != scenario or str(row.get("controller")) != controller:
            continue
        seed = str(row.get("seed"))
        if seed in values:
            duplicate_seeds.add(seed)
        else:
            values[seed] = _number(row, metric)
    for seed in duplicate_seeds:
        values.pop(seed, None)
    return values


def _normality_p(values: np.ndarray) -> Optional[float]:
    """Shapiro-Wilk normality p-value for a small/medium sample, if possible."""
    if len(values) < 3:
        return None
    sample = values if len(values) <= 5000 else values[:5000]
    if np.allclose(sample, sample[0]):
        return 1.0
    try:
        return float(stats.shapiro(sample).pvalue)
    except ValueError:
        return None


def _mean_difference_ci(differences: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Student-t interval for the mean difference (undefined for n < 2)."""
    if len(differences) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(differences))
    sem = float(stats.sem(differences))
    if sem == 0:
        return mean, mean
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(differences) - 1))
    return mean - critical * sem, mean + critical * sem


def _test_one(
    left: np.ndarray,
    right: np.ndarray,
    paired: bool,
) -> Tuple[str, float, float, str]:
    """Choose a t-test for compatible samples, otherwise a rank-based test."""
    if paired:
        differences = left - right
        if np.allclose(differences, 0):
            return "Wilcoxon signed-rank (all differences zero)", 0.0, 1.0, "paired by seed"
        if np.allclose(differences, differences[0]):
            result = stats.wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")
            return "Wilcoxon signed-rank", float(result.statistic), float(result.pvalue), "paired by seed"
        normality_p = _normality_p(differences)
        if normality_p is not None and normality_p >= 0.05:
            result = stats.ttest_rel(left, right)
            return "paired t-test", float(result.statistic), float(result.pvalue), "paired by seed"
        result = stats.wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")
        return "Wilcoxon signed-rank", float(result.statistic), float(result.pvalue), "paired by seed"

    left_normality = _normality_p(left)
    right_normality = _normality_p(right)
    nonconstant = np.std(left, ddof=1) > 0 and np.std(right, ddof=1) > 0
    if nonconstant and left_normality is not None and right_normality is not None and min(left_normality, right_normality) >= 0.05:
        result = stats.ttest_ind(left, right, equal_var=False)
        return "Welch independent t-test", float(result.statistic), float(result.pvalue), "independent samples"
    result = stats.mannwhitneyu(left, right, alternative="two-sided")
    return "Mann-Whitney U", float(result.statistic), float(result.pvalue), "independent samples"


def _holm_adjust(rows: List[Dict[str, object]]) -> None:
    """Apply Holm correction within each scenario/metric family of pair tests."""
    families: Dict[Tuple[str, str], List[int]] = {}
    for index, row in enumerate(rows):
        if math.isfinite(float(row["raw_p_value"])):
            families.setdefault((str(row["scenario"]), str(row["metric"])), []).append(index)
    for indices in families.values():
        ordered = sorted(indices, key=lambda index: float(rows[index]["raw_p_value"]))
        count = len(ordered)
        previous = 0.0
        for rank, index in enumerate(ordered):
            raw = float(rows[index]["raw_p_value"])
            adjusted = min(1.0, max(previous, (count - rank) * raw))
            rows[index]["holm_adjusted_p_value"] = adjusted
            previous = adjusted


def compare_controllers(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Compare each requested controller pair within scenario and metric.

    Observations are matched by scenario and evaluation seed when at least two
    unique seeds overlap. Otherwise, available unpaired samples are tested with
    an independent method. Differences are consistently left controller minus
    right controller. Holm correction is applied across controller-pair tests
    within each scenario/metric family.
    """
    scenarios = sorted({str(row.get("scenario")) for row in rows})
    output: List[Dict[str, object]] = []
    for scenario in scenarios:
        for metric in METRIC_COLUMNS:
            for left_name, right_name in CONTROLLER_PAIRS:
                left_by_seed = _episode_values(rows, scenario, left_name, metric)
                right_by_seed = _episode_values(rows, scenario, right_name, metric)
                common = sorted(set(left_by_seed) & set(right_by_seed))
                paired_values = [
                    (left_by_seed[key], right_by_seed[key])
                    for key in common
                    if math.isfinite(left_by_seed[key]) and math.isfinite(right_by_seed[key])
                ]
                if len(paired_values) >= 2:
                    left = np.asarray([pair[0] for pair in paired_values], dtype=float)
                    right = np.asarray([pair[1] for pair in paired_values], dtype=float)
                    paired = True
                else:
                    paired = False
                    left = np.asarray(
                        [value for key, value in left_by_seed.items() if key not in common and math.isfinite(value)],
                        dtype=float,
                    )
                    right = np.asarray(
                        [value for key, value in right_by_seed.items() if key not in common and math.isfinite(value)],
                        dtype=float,
                    )

                row: Dict[str, object] = {
                    "scenario": scenario,
                    "metric": metric,
                    "controller_a": left_name,
                    "controller_b": right_name,
                    "sample_size": len(paired_values) if paired else min(len(left), len(right)),
                    "n_controller_a": len(left),
                    "n_controller_b": len(right),
                    "comparison_design": "paired by seed" if paired else "independent samples",
                    "test_used": "not performed",
                    "test_statistic": float("nan"),
                    "raw_p_value": float("nan"),
                    "holm_adjusted_p_value": float("nan"),
                    "mean_difference_a_minus_b": float("nan"),
                    "ci95_difference_low": float("nan"),
                    "ci95_difference_high": float("nan"),
                    "note": "insufficient valid observations",
                }
                if paired and len(left) >= 2:
                    differences = left - right
                    row["mean_difference_a_minus_b"] = float(np.mean(differences))
                    low, high = _mean_difference_ci(differences)
                    row["ci95_difference_low"] = low
                    row["ci95_difference_high"] = high
                elif not paired and len(left) >= 2 and len(right) >= 2:
                    row["mean_difference_a_minus_b"] = float(np.mean(left) - np.mean(right))
                    # Welch interval for the difference in independent means.
                    variance = np.var(left, ddof=1) / len(left) + np.var(right, ddof=1) / len(right)
                    if variance == 0:
                        low = high = float(row["mean_difference_a_minus_b"])
                    else:
                        numerator = variance ** 2
                        denominator = (
                            (np.var(left, ddof=1) / len(left)) ** 2 / (len(left) - 1)
                            + (np.var(right, ddof=1) / len(right)) ** 2 / (len(right) - 1)
                        )
                        dof = numerator / denominator if denominator else min(len(left), len(right)) - 1
                        critical = stats.t.ppf(0.975, dof)
                        margin = critical * math.sqrt(variance)
                        low = float(row["mean_difference_a_minus_b"]) - margin
                        high = float(row["mean_difference_a_minus_b"]) + margin
                    row["ci95_difference_low"] = float(low)
                    row["ci95_difference_high"] = float(high)

                if paired and len(left) >= 2 or (not paired and len(left) >= 2 and len(right) >= 2):
                    try:
                        test_name, statistic, p_value, design = _test_one(left, right, paired)
                        row.update({
                            "test_used": test_name,
                            "test_statistic": statistic,
                            "raw_p_value": p_value,
                            "comparison_design": design,
                            "note": "",
                        })
                    except ValueError as error:
                        row["note"] = f"test unavailable: {error}"
                output.append(row)
    _holm_adjust(output)
    return output
