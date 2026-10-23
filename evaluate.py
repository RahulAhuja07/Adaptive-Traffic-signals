"""Evaluate saved learning agents and Webster across paired-seed episodes."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dqn_agent import DQNAgent
from q_learning import QLearningAgent
from utils import (
    CONTROLLERS,
    EVALUATION_FIELDS,
    PROJECT_DIR,
    aggregate_evaluation,
    ensure_output_dirs,
    make_environment,
    model_path,
    selected_scenarios,
    set_global_seed,
    write_csv,
)
from webster_controller import WebsterController


def run_greedy_episode(
    env,
    controller,
    controller_name: str,
    scenario: str,
    episode: int,
    seed: int,
) -> Dict[str, object]:
    """Run one episode without changing a learned controller's parameters."""
    state = env.reset(seed=seed)
    if controller_name == "Webster":
        controller.reset()
    done = False
    while not done:
        if controller_name == "Webster":
            action = controller.choose_action()
        else:
            action = controller.choose_action(state, training=False)
        if action not in (0, 1):
            raise RuntimeError(f"{controller_name} returned invalid action {action!r}")
        state, _, done, _ = env.step(action)
    metrics = env.get_metrics()
    return {
        "scenario": scenario,
        "controller": controller_name,
        "episode": episode,
        "seed": seed,
        **metrics,
    }


def load_learned_models(root: Path, scenario: str, seed: int) -> Tuple[QLearningAgent, DQNAgent]:
    """Load required model files or raise a clear missing-model error."""
    q_path = model_path(root, "q_learning", scenario, seed)
    dqn_path = model_path(root, "dqn", scenario, seed)
    missing = [path for path in (q_path, dqn_path) if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"trained model file(s) not found: {names}. Run train.py for this "
            f"scenario and seed before evaluation."
        )
    # Seed the agents' random tie breakers as well as the environment. Greedy
    # selection only uses randomness when multiple learned values are equal.
    q_agent = QLearningAgent(seed=seed + 101)
    q_agent.load(str(q_path))
    dqn_agent = DQNAgent(seed=seed + 202)
    dqn_agent.load(str(dqn_path))
    return q_agent, dqn_agent


def evaluate_scenario(
    scenario: str,
    rates: Sequence[float],
    episodes: int,
    seed: int,
    model_seed: int,
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
    output_root: Path = PROJECT_DIR,
) -> List[Dict[str, object]]:
    """Evaluate all three controllers under shared parameters and episode seeds."""
    q_agent, dqn_agent = load_learned_models(output_root, scenario, model_seed)
    webster = WebsterController(
        arrival_rates=rates, interval_seconds=interval_seconds
    )
    controllers = (("Q-Learning", q_agent), ("DQN", dqn_agent), ("Webster", webster))
    env = make_environment(rates, episode_steps, interval_seconds, service_capacity)
    rows: List[Dict[str, object]] = []
    for episode in range(1, episodes + 1):
        # Pair all controllers on this same deterministic arrival seed.
        episode_seed = seed + episode - 1
        for name, controller in controllers:
            rows.append(
                run_greedy_episode(env, controller, name, scenario, episode, episode_seed)
            )
    return rows


def run_evaluation(
    episodes: int = 30,
    seed: int = 42,
    model_seed: Optional[int] = None,
    scenario: str = "all",
    episode_steps: int = 180,
    interval_seconds: float = 10.0,
    service_capacity: int = 5,
    output_root: Path = PROJECT_DIR,
    make_plots: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Evaluate trained models, save per-episode/summary CSVs, and plot results."""
    if episodes <= 0 or episode_steps <= 0:
        raise ValueError("episodes and episode_steps must be positive")
    if model_seed is None:
        model_seed = seed
    ensure_output_dirs(output_root)
    set_global_seed(seed)
    rows: List[Dict[str, object]] = []
    for scenario_name, rates in selected_scenarios(scenario):
        print(f"Evaluating {scenario_name} demand: NS={rates[0]}, EW={rates[1]}")
        rows.extend(
            evaluate_scenario(
                scenario_name,
                rates,
                episodes,
                seed,
                model_seed,
                episode_steps,
                interval_seconds,
                service_capacity,
                output_root,
            )
        )
    summary = aggregate_evaluation(rows)
    results_dir = output_root / "results"
    write_csv(results_dir / "evaluation_results.csv", rows, EVALUATION_FIELDS)
    write_csv(results_dir / "summary_results.csv", summary)
    if make_plots:
        from plot_results import generate_plots

        generate_plots(output_root)
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="base seed for paired evaluation episodes")
    parser.add_argument("--model-seed", type=int, default=None, help="seed used when naming the trained model files")
    parser.add_argument("--scenario", choices=("low", "medium", "high", "all"), default="all")
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--service-capacity", type=int, default=5)
    parser.add_argument("--output-dir", dest="output_root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")
    return parser


def main() -> None:
    args = vars(build_parser().parse_args())
    args["make_plots"] = not args.pop("no_plots")
    try:
        run_evaluation(**args)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"Evaluation failed: {error}") from error


if __name__ == "__main__":
    main()
