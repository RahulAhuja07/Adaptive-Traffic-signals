"""Paired-seed evaluation harness for the four SUMO controllers.

Evaluation is greedy and never calls learning/update methods. Every controller
gets a fresh SUMO process for each identical route-file/seed pair.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "runs" / "comparison"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q_learning import QLearningAgent
from dqn_agent import DQNAgent
from runs.state_only_dqn_experiment.dqn_agent import StateOnlyDQNAgent

try:
    from .sumo_environment import SumoTrafficEnvironment, NETWORK_FILE, ROUTES_FILE
    from .state_only_dqn_runner import features_from_observation, validate_features
    from .webster_adapter import SumoWebsterAdapter
except ImportError:
    from sumo_environment import SumoTrafficEnvironment, NETWORK_FILE, ROUTES_FILE
    from state_only_dqn_runner import features_from_observation, validate_features
    from webster_adapter import SumoWebsterAdapter


METRICS = (
    "time_weighted_avg_queue_vehicles",
    "total_stopped_waiting_time_sec_completed",
    "avg_stopped_waiting_time_sec_completed",
    "throughput_completed_veh_per_min",
    "completed_vehicles",
    "simulated_duration_sec",
    "cumulative_reward",
)
ROW_FIELDS = (
    "controller", "scenario", "training_seed", "checkpoint", "checkpoint_sha256",
    "evaluation_seed", "episode_horizon_sec", "decision_interval_sec",
    "completion_status", "termination_reason", "sumo_inserted_vehicles",
    "remaining_expected_vehicles", "demand_completed", "paired_inserted_count_match",
    *METRICS, "final_queue_vehicles", "decision_steps",
)
MIN_TRAINING_EPISODES = 100  # Screening label only; never claimed as a sufficiency theorem.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _sumo_version(binary: str | None) -> str | None:
    if not binary:
        binary = shutil.which("sumo")
    if not binary:
        home = __import__("os").environ.get("SUMO_HOME")
        candidate = Path(home) / "bin" / "sumo" if home else None
        binary = str(candidate) if candidate and candidate.is_file() else None
    if not binary:
        return None
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                timeout=10, check=False)
        return (result.stdout or result.stderr).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _flows_description(routes: Path) -> dict[str, Any]:
    root = ET.parse(routes).getroot()
    return {"flows": [dict(flow.attrib) for flow in root.findall("flow")],
            "routes": [dict(route.attrib) for route in root.findall("route")]}


def _training_config(checkpoint: Path) -> dict[str, Any]:
    candidates = [checkpoint.parent / "training_config.json"]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def audit_checkpoint(kind: str, checkpoint: Path | str) -> dict[str, Any]:
    """Validate architecture/artifact and classify its documented training history."""
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{kind} checkpoint does not exist: {path}")
    config = _training_config(path)
    episodes = config.get("episodes")
    return {
        "path": str(path), "sha256": _sha256(path),
        "training_episodes": episodes,
        "training_seed": config.get("seed"),
        "training_provenance": "SUMO-specific runner config present; SUMO runtime/version not recorded"
        if config else "unknown (no adjacent training_config.json)",
        "screening_class": "trained_candidate" if isinstance(episodes, int) and episodes >= MIN_TRAINING_EPISODES
        else "smoke_or_unverified",
        "training_config": config,
    }


def _new_agent(kind: str, checkpoint: Path, eval_seed: int) -> Any:
    if kind == "q_learning":
        agent = QLearningAgent(seed=eval_seed + 17)
        agent.load(str(checkpoint))
        if np.asarray(agent.q_table).shape != (18, 2):
            raise ValueError("Q-learning checkpoint must contain an 18x2 table")
        return agent
    if kind == "original_dqn":
        agent = DQNAgent(seed=eval_seed + 17)
        if agent.state_size != 18 or agent.action_size != 2:
            raise ValueError("original DQN must have 18 inputs/states and 2 actions")
        agent.load(str(checkpoint))
        return agent
    agent = StateOnlyDQNAgent(seed=eval_seed + 17)
    if agent.state_size != 6 or agent.action_size != 2:
        raise ValueError("state-only DQN must have 6 features and 2 actions")
    agent.load(str(checkpoint))
    return agent


def _controller_action(kind: str, agent: Any, observation: Any, env: Any) -> int:
    if kind == "state_only_dqn":
        feature = validate_features(features_from_observation(env.get_queue_observation()))
        action = agent.choose_action(feature, training=False)
    else:
        if isinstance(observation, bool) or not isinstance(observation, (int, np.integer)) or not 0 <= int(observation) < 18:
            raise ValueError(f"{kind} requires integer SUMO state in [0,17], got {observation!r}")
        action = agent.choose_action(int(observation), training=False)
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or int(action) not in (0, 1):
        raise ValueError(f"{kind} returned invalid action {action!r}; expected 0=NS or 1=EW")
    return int(action)


def _evaluate_one(kind: str, checkpoint: Path | None, seed: int, horizon: float,
                  interval: float, env_factory: Callable[..., Any],
                  controller_factory: Callable[..., Any], webster_options: Mapping[str, Any]
                  ) -> dict[str, Any]:
    env = env_factory(episode_horizon_seconds=horizon, decision_interval_seconds=interval)
    try:
        observation = env.reset(seed=seed)
        if kind == "webster":
            controller = controller_factory(env, **webster_options)
            controller.reset()
        else:
            assert checkpoint is not None
            agent = _new_agent(kind, checkpoint, seed)
        done = False
        reward_sum = 0.0
        steps = 0
        reason = ""
        while not done:
            action = controller.choose_action() if kind == "webster" else _controller_action(kind, agent, observation, env)
            if isinstance(action, bool) or int(action) not in (0, 1):
                raise ValueError(f"{kind} action must be 0 (NS) or 1 (EW), got {action!r}")
            observation, reward, done, info = env.step(int(action))
            reward_sum += float(reward)
            steps += 1
            if done:
                reason = str(info.get("termination_reason", "unspecified"))
        raw = env.get_metrics()
        return {
            "evaluation_seed": seed,
            "completion_status": "completed" if raw.get("demand_completed") else "horizon_or_partial_demand",
            "termination_reason": reason,
            "sumo_inserted_vehicles": int(raw.get("sumo_inserted_vehicles", 0)),
            "remaining_expected_vehicles": raw.get("remaining_expected_vehicles"),
            "demand_completed": bool(raw.get("demand_completed", False)),
            "time_weighted_avg_queue_vehicles": float(raw["time_weighted_avg_queue"]),
            "total_stopped_waiting_time_sec_completed": float(raw["total_stopped_waiting_time_sec_completed"]),
            "avg_stopped_waiting_time_sec_completed": float(raw["avg_stopped_waiting_time_sec"]),
            "throughput_completed_veh_per_min": float(raw["throughput_veh_per_min"]),
            "completed_vehicles": int(raw["completed_vehicles"]),
            "simulated_duration_sec": float(raw["simulated_time_sec"]),
            "cumulative_reward": reward_sum,
            "final_queue_vehicles": float(raw["final_queue"]),
            "decision_steps": steps,
        }
    finally:
        env.close()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["controller"]), str(row["scenario"]), str(row["training_seed"]))].append(row)
    summaries = []
    for (controller, scenario, training_seed), values in sorted(groups.items()):
        output: dict[str, Any] = {"controller": controller, "scenario": scenario,
                                  "training_seed": training_seed, "n_episodes": len(values)}
        for metric in METRICS:
            array = np.asarray([float(row[metric]) for row in values], dtype=float)
            array = array[np.isfinite(array)]
            output[f"{metric}_mean"] = float(array.mean()) if array.size else float("nan")
            output[f"{metric}_sd"] = float(array.std(ddof=1)) if array.size > 1 else float("nan")
            output[f"{metric}_n"] = int(array.size)
        output["demand_completion_rate"] = sum(bool(row["demand_completed"]) for row in values) / len(values)
        output["mean_inserted_vehicles"] = float(np.mean([float(row["sumo_inserted_vehicles"]) for row in values]))
        summaries.append(output)
    return summaries


def _plots(output: Path, summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to create comparison plots") from error
    plot_specs = (
        ("time_weighted_avg_queue_vehicles", "Time weighted average queue (vehicles)"),
        ("total_stopped_waiting_time_sec_completed", "Stopped waiting time of completed trips (s)"),
        ("throughput_completed_veh_per_min", "Completed trips per simulated minute"),
        ("simulated_duration_sec", "Simulated duration (s)"),
    )
    generated = []
    for metric, label in plot_specs:
        rows = [row for row in summaries if int(row[f"{metric}_n"]) > 0]
        labels = [str(row["controller"]) + (f" / train seed {row['training_seed']}" if row["training_seed"] != "n/a" else "") for row in rows]
        means = [float(row[f"{metric}_mean"]) for row in rows]
        sds = [float(row[f"{metric}_sd"]) if np.isfinite(float(row[f"{metric}_sd"])) else 0.0 for row in rows]
        fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.5), 4.5))
        ax.bar(np.arange(len(rows)), means, yerr=sds, capsize=4, color="#3976a8")
        ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"Descriptive comparison: {label}")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        filename = f"{metric}.png"
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        generated.append(filename)
    return generated


def run_comparison(
    q_checkpoints: Sequence[Path | str] = (HERE / "runs/q_learning/q_table.npz",),
    dqn_checkpoints: Sequence[Path | str] = (HERE / "runs/original_dqn/dqn_model.pt",),
    state_only_checkpoints: Sequence[Path | str] = (HERE / "runs/state_only_dqn/state_only_dqn_model.pt",),
    seeds: Sequence[int] = tuple(range(1000, 1030)),
    horizon: float = 1800.0,
    interval: float = 10.0,
    output_dir: Path | str = DEFAULT_OUTPUT,
    allow_smoke_checkpoints: bool = False,
    env_factory: Callable[..., Any] = SumoTrafficEnvironment,
    controller_factory: Callable[..., Any] = SumoWebsterAdapter,
    webster_options: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate each available checkpoint and Webster on identical scenario seeds."""
    output = Path(output_dir)
    checkpoint_sets = {
        "q_learning": [Path(p).resolve() for p in q_checkpoints],
        "original_dqn": [Path(p).resolve() for p in dqn_checkpoints],
        "state_only_dqn": [Path(p).resolve() for p in state_only_checkpoints],
    }
    audits = {kind: [audit_checkpoint(kind, path) for path in paths]
              for kind, paths in checkpoint_sets.items()}
    risky = [(kind, item) for kind, items in audits.items() for item in items
             if item["screening_class"] != "trained_candidate"]
    if risky and not allow_smoke_checkpoints:
        listing = "; ".join(f"{kind}: {item['path']} ({item['training_episodes']} recorded episodes)" for kind, item in risky)
        raise ValueError("refusing smoke/unverified checkpoints by default; use --allow-smoke-checkpoints for explicitly exploratory output: " + listing)
    if any(not values for values in checkpoint_sets.values()):
        raise ValueError("provide at least one checkpoint for each learned controller")
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a nonempty sequence of integers")

    scenario = "configured_periodic_routes"
    comparison_routes = Path(os.environ.get("SUMO_ROUTES_FILE", str(ROUTES_FILE))).expanduser().resolve()
    webster_options = dict(webster_options or {})
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[str, Path | None, str, str]] = []
    for kind, paths in checkpoint_sets.items():
        for path, audit in zip(paths, audits[kind]):
            jobs.extend((kind, path, str(audit.get("training_seed") if audit.get("training_seed") is not None else "unknown"), audit["sha256"]) for _ in [0])
    jobs.append(("webster", None, "n/a", ""))
    for kind, checkpoint, training_seed, digest in jobs:
        for seed in seeds:
            record = _evaluate_one(kind, checkpoint, seed, horizon, interval,
                                   env_factory, controller_factory, webster_options)
            rows.append({
                "controller": kind, "scenario": scenario, "training_seed": training_seed,
                "checkpoint": str(checkpoint) if checkpoint else "",
                "checkpoint_sha256": digest, "episode_horizon_sec": horizon,
                "decision_interval_sec": interval, **record,
            })
    paired: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        paired[int(row["evaluation_seed"])].add(int(row["sumo_inserted_vehicles"]))
    for row in rows:
        row["paired_inserted_count_match"] = len(paired[int(row["evaluation_seed"])]) == 1

    summaries = _summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episode_metrics.csv", rows, ROW_FIELDS)
    summary_fields = list(summaries[0].keys()) if summaries else []
    _write_csv(output / "summary.csv", summaries, summary_fields)
    plot_files = _plots(output, summaries)
    binary = getattr(env_factory, "sumo_binary", None)
    config = {
        "experiment": "paired four-controller SUMO evaluation",
        "label": "exploratory_smoke_only" if risky else "candidate_comparison",
        "scenario": scenario,
        "network": {"path": str(Path(NETWORK_FILE).resolve()), "sha256": _sha256(Path(NETWORK_FILE))},
        "routes": {"path": str(comparison_routes), "sha256": _sha256(comparison_routes),
                   **_flows_description(comparison_routes)},
        "episode_horizon_seconds": float(horizon), "decision_interval_seconds": float(interval),
        "evaluation_seeds": list(seeds), "controllers": audits | {"webster": [{"training": "not applicable", **webster_options}]},
        "webster_assumptions": webster_options or {"saturation_flows_veh_per_hour_ns_ew": [1800, 1800],
            "lost_time": "derived from 3s+3s yellow in network", "arrival_rate": "periodic route XML nominal rates"},
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                    "torch": _version("torch"), "traci": _version("traci"), "sumo_version": _sumo_version(binary)},
        "fairness_controls": ["same network and route XML", "same SUMO seed per controller within each paired scenario",
            "same horizon and decision interval", "fresh SUMO process per episode", "no learning/update calls during evaluation",
            "inserted vehicle counts are compared and flagged; congestion can affect actual insertion"],
        "metric_definitions": {
            "time_weighted_avg_queue_vehicles": "time average of NS+EW A1 inbound halting vehicles (speed <=0.1 m/s)",
            "total_stopped_waiting_time_sec_completed": "sum of sampled stopped seconds for trips that completed; unfinished vehicles excluded",
            "avg_stopped_waiting_time_sec_completed": "mean sampled stopped seconds among completed trips; NaN if none completed",
            "throughput_completed_veh_per_min": "completed route trips / simulated minutes",
            "simulated_duration_sec": "SUMO elapsed simulated seconds before horizon or no expected/active demand",
            "cumulative_reward": "sum of shared queue-based rewards; reported for reference, Webster reward is not its objective",
            "sumo_inserted_vehicles": "SUMO departed count accumulated each simulation step",
            "demand_completed": "SUMO min expected vehicle count is zero at termination",
        },
        "observation_reward_differences": {"q_learning": "18 discrete states from green phase and 3-bin SUMO halting counts; queue reward",
            "original_dqn": "same 18 state IDs, one-hot encoded by original agent; queue reward",
            "state_only_dqn": "six raw normalized phase/queue features, unbounded counts divided by 20; queue reward",
            "webster": "periodic-flow and assumed saturation-flow timing plan; reward is reference only"},
        "minimum_training_episode_screen": MIN_TRAINING_EPISODES,
        "training_screen_is_not_evidence_of_scientific_sufficiency": True,
        "plots": plot_files,
    }
    (output / "comparison_config.json").write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(output, summaries, config)
    return rows, summaries


def _write_report(output: Path, summaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    lines = ["# Four-controller SUMO comparison", "", f"Status: **{config['label']}**", "",
             "Descriptive summaries only; this report makes no superiority claim.", "",
             "Waiting time includes completed trips only. Queue is the time-weighted sum of inbound halting vehicles. Throughput counts completed trips per simulated minute. Webster reward is reference only.", "",
             "| Controller | Training seed | n | Avg queue (mean ± SD) | Completed-trip stopped wait (s, mean ± SD) | Throughput/min (mean ± SD) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        def show(metric: str) -> str:
            mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
            return f"{mean:.3f} ± {sd:.3f}" if np.isfinite(mean) and np.isfinite(sd) else (f"{mean:.3f} ± n/a" if np.isfinite(mean) else "n/a")
        lines.append(f"| {row['controller']} | {row['training_seed']} | {row['n_episodes']} | {show('time_weighted_avg_queue_vehicles')} | {show('total_stopped_waiting_time_sec_completed')} | {show('throughput_completed_veh_per_min')} |")
    lines += ["", "## Generated plots", "", *[f"- ![{name}]({name})" for name in config["plots"]], "",
              "## Fairness and interpretation", "",
              "Every row uses the same route file, horizon, decision interval, and evaluation seed schedule. Inserted counts are checked per paired seed and mismatch is flagged. Waiting time is limited to completed trips by the adapter; unfinished-vehicle waiting is unavailable. Learned controllers retain different state representations as documented in comparison_config.json.", ""]
    (output / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
