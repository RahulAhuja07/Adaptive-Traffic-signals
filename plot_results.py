"""Generate comparison PNGs from saved training/evaluation CSV files."""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    CONTROLLERS,
    PROJECT_DIR,
    SCENARIO_LABELS,
    ensure_output_dirs,
    read_csv,
)


def _number(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _training_plot(root: Path, algorithm: str, title: str, filename: str) -> None:
    paths = sorted((root / "results").glob(f"{algorithm}_*_training.csv"))
    fig, axis = plt.subplots(figsize=(9, 5))
    plotted = False
    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        episodes = np.array([_number(row.get("episode")) for row in rows])
        rewards = np.array([_number(row.get("cumulative_reward")) for row in rows])
        valid = np.isfinite(episodes) & np.isfinite(rewards)
        episodes, rewards = episodes[valid], rewards[valid]
        if not len(rewards):
            continue
        scenario = rows[0].get("scenario", path.stem)
        window = min(100, len(rewards))
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axis.plot(episodes, rewards, alpha=0.2, linewidth=0.7)
        axis.plot(episodes[window - 1 :], rolling, label=SCENARIO_LABELS.get(scenario, scenario))
        plotted = True
    axis.set_title(title)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Cumulative reward")
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No valid training data", ha="center", va="center", transform=axis.transAxes)
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "plots" / filename, dpi=150)
    plt.close(fig)


def _comparison_plot(
    summary: Sequence[Dict[str, str]],
    root: Path,
    mean_key: str,
    std_key: str,
    title: str,
    ylabel: str,
    filename: str,
    error_bars: bool = False,
) -> None:
    scenarios = ["low", "medium", "high"]
    x = np.arange(len(scenarios), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5))
    plotted = False
    for controller_index, controller in enumerate(CONTROLLERS):
        means: List[float] = []
        errors: List[float] = []
        for scenario in scenarios:
            match = next(
                (row for row in summary if row.get("scenario") == scenario and row.get("controller") == controller),
                None,
            )
            means.append(_number(match.get(mean_key)) if match else float("nan"))
            errors.append(_number(match.get(std_key)) if match else float("nan"))
        offset = (controller_index - 1) * width
        if any(math.isfinite(value) for value in means):
            yerr = np.array(errors) if error_bars else None
            axis.bar(x + offset, means, width, yerr=yerr, capsize=3 if error_bars else 0, label=controller)
            plotted = True
    axis.set_title(title)
    axis.set_xlabel("Traffic demand scenario")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, [SCENARIO_LABELS[scenario] for scenario in scenarios])
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No valid evaluation data", ha="center", va="center", transform=axis.transAxes)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "plots" / filename, dpi=150)
    plt.close(fig)


def generate_plots(root: Path = PROJECT_DIR) -> List[Path]:
    """Generate the six requested plots from existing CSV results only."""
    ensure_output_dirs(root)
    _training_plot(root, "q_learning", "Q-learning training reward", "q_learning_training_reward.png")
    _training_plot(root, "dqn", "DQN training reward", "dqn_training_reward.png")
    summary_path = root / "results" / "summary_results.csv"
    summary = read_csv(summary_path) if summary_path.exists() else []
    _comparison_plot(
        summary, root, "mean_avg_waiting_time_sec", "std_avg_waiting_time_sec",
        "Mean waiting time by controller and demand", "Average waiting time (seconds)",
        "waiting_time_comparison.png", error_bars=True,
    )
    _comparison_plot(
        summary, root, "mean_time_weighted_avg_queue", "std_time_weighted_avg_queue",
        "Average queue by controller and demand", "Time-weighted average queue (vehicles)",
        "average_queue_comparison.png",
    )
    _comparison_plot(
        summary, root, "mean_throughput_veh_per_min", "std_throughput_veh_per_min",
        "Throughput by controller and demand", "Throughput (vehicles per minute)",
        "throughput_comparison.png",
    )
    _comparison_plot(
        summary, root, "mean_final_queue", "std_final_queue",
        "Final queue by controller and demand", "Final queue (vehicles)",
        "final_queue_comparison.png",
    )
    return sorted((root / "plots").glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR)
    args = parser.parse_args()
    paths = generate_plots(args.output_dir)
    print(f"Generated {len(paths)} plots in {args.output_dir / 'plots'}")


if __name__ == "__main__":
    main()
