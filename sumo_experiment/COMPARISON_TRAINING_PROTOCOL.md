# Proposed training and evaluation protocol

This protocol keeps full training and long comparison separate from short
calibration. The one-episode sustained-demand calibrations documented below
were run in fresh directories; existing smoke artifacts were preserved.

## Comparison scenario

The original `rl_routes.rou.xml` has two flows with `end=600` and
`period=2`. SUMO schedules departures at 0 through 598, or 300 vehicles per
direction (600 total). The existing one-episode calibration CSVs show all 600
completed, then `sumo_no_expected_vehicles` at different times as the
controllers' queues cleared: Q-learning at 1,010 seconds, original DQN at 900,
and state-only DQN at 850. Those times reflect different traffic clearance,
not different scheduled demand.

Generate and select a dedicated route scenario:

```sh
.venv/bin/python -m sumo_experiment.create_comparison_routes \
  --demand-duration-seconds 1800
export SUMO_ROUTES_FILE="$PWD/sumo_experiment/scenarios/periodic_demand_1800s.rou.xml"
```

The generator copies the original vehicle type and route definitions and only
extends both flow end times. Since the flow end is exclusive, `begin=0` and
`period=2` request departures at 0 through 1,798 seconds: 900 per direction.
It refuses to write over the original route file. All four controllers use
this same route through `SUMO_ROUTES_FILE`; Python users can also pass
`routes_file` to `SumoTrafficEnvironment`. Webster derives its nominal arrival
rates from the environment's selected route file. Network geometry and A1
yellow clearance are unchanged.

The simulation horizon and SUMO `--end` stay at 1,800 seconds. There is no
post-horizon drain period. Vehicles scheduled near 1,800 may still be active or
waiting for insertion when the horizon ends; they do not receive extra time to
complete, and completed-trip waiting metrics exclude them. The scheduled route
demand is identical across controllers, but actual inserted/completed totals
can vary with insertion feasibility and congestion, so record those metrics.

## Training budget

- Train Q-learning for 1,000 episodes per training seed.
- Train original DQN and six-feature state-only DQN for 500 episodes per
  training seed each, retaining their existing algorithm settings and replay
  behavior. Save a final checkpoint, not a best-of-evaluation checkpoint.
- Use training seeds 42, 43, and 44; each episode uses `seed + episode_index`.
- Use the generated comparison route file, 1,800-second horizon, and 10-second decision
  interval across all three agents. Webster remains training-free.
- Do not use metric-based early stopping or test-set model selection. Stop early
  only on execution errors or invalid/nonfinite learning state, and mark such a
  run incomplete.
- Keep per-seed checkpoints and logs under
  `sumo_experiment/runs/comparison/training/{controller}/seed_{seed}/`.

## Evaluation

After training is reviewed, evaluate each learned checkpoint on the same 30
held-out SUMO seeds (1000–1029), paired across all controllers. Webster is run
once per held-out seed. Preserve the configured periodic route demand and use
identical horizon/interval. Report per-training-seed summaries and repeated
episode standard deviations; do not pool away training-seed variation.

The checkpoint gate of 100 training episodes in the harness is a smoke-test
screen, not a claim that 100 episodes are scientifically sufficient.

## Runtime estimate (not measured)

At 1-second SUMO steps, this proposal contains up to 5.4 million simulated
steps for Q-learning, 2.7 million for each DQN, and 540,000 steps for the
comparison (300 episodes), about 11.34 million steps total. If the local setup
averages 100–1,000 SUMO steps/second, this is roughly 3.2–31.5 hours, before
Python/TraCI overhead. These rates are assumptions, not measurements; DQN's
per-step interaction may run slower. Benchmark one complete episode per agent
and extrapolate before approval of the full workload.

## Suggested staged commands

First select the comparison scenario and run the one-episode runtime calibration
below, then review its wall time.
After explicit approval for the substantial budget, the following commands
create isolated outputs for all proposed seeds; none has been executed:

## One-episode runtime calibration

These commands run one calibration episode per learned controller. Each creates
a new dedicated output directory with `mkdir` (which fails if it already
exists), so the existing `runs/q_learning`, `runs/original_dqn`, and
`runs/state_only_dqn` smoke artifacts are not overwritten. All three use seed
42, one episode, 1,800-second maximum horizon, and the 10-second decision
interval. The environment's SUMO internal step remains its source default of
1 second. They use new sustained-demand output directories, preserving the
previous `comparison/calibration/` outputs and checkpoints.

The block resolves the executable in the same order as the environment
(`SUMO_BINARY`, `SUMO_HOME/bin/sumo`, then `PATH`), requires it to be executable,
exports its path explicitly, and saves `sumo --version` before starting any
trainer. It exits before training if no binary is available. The calibrations
below used SUMO 1.28.0 from
`/Library/Frameworks/EclipseSUMO.framework/Versions/1.28.0/EclipseSUMO/bin/sumo`.
For this framework install, `SUMO_HOME` is
`/Library/Frameworks/EclipseSUMO.framework/Versions/1.28.0/EclipseSUMO/share/sumo`;
its bundled TraCI reports version 1.28.0.

```sh
set -eu
calibration_root="sumo_experiment/runs/comparison/calibration_sustained"
mkdir -p "$calibration_root"

if [ -n "${SUMO_BINARY:-}" ]; then
  sumo_bin="$SUMO_BINARY"
elif [ -n "${SUMO_HOME:-}" ]; then
  sumo_bin="$SUMO_HOME/bin/sumo"
elif command -v sumo >/dev/null 2>&1; then
  sumo_bin="$(command -v sumo)"
else
  sumo_bin=""
fi
if [ -z "$sumo_bin" ] || [ ! -x "$sumo_bin" ]; then
  echo "SUMO executable unavailable; set SUMO_BINARY or SUMO_HOME" >&2
  exit 1
fi
export SUMO_BINARY="$sumo_bin"
scenario="sumo_experiment/scenarios/periodic_demand_1800s.rou.xml"
.venv/bin/python -m sumo_experiment.create_comparison_routes \
  --demand-duration-seconds 1800 --output "$scenario"
export SUMO_ROUTES_FILE="$PWD/$scenario"
{
  echo "SUMO_BINARY=$SUMO_BINARY"
  "$SUMO_BINARY" --version
} > "$calibration_root/sumo_binary_version.txt" 2>&1

out="$calibration_root/q_learning_1800s_seed42"
mkdir "$out"
/usr/bin/time -p -o "$out/wall_clock.txt" \
  .venv/bin/python -m sumo_experiment.train_q_learning \
  --episodes 1 --seed 42 --episode-horizon-seconds 1800 \
  --decision-interval-seconds 10 --output-dir "$out"

out="$calibration_root/original_dqn_1800s_seed42"
mkdir "$out"
/usr/bin/time -p -o "$out/wall_clock.txt" \
  .venv/bin/python -m sumo_experiment.train_dqn \
  --episodes 1 --seed 42 --episode-horizon-seconds 1800 \
  --decision-interval-seconds 10 --batch-size 64 --output-dir "$out"

out="$calibration_root/state_only_dqn_1800s_seed42"
mkdir "$out"
/usr/bin/time -p -o "$out/wall_clock.txt" \
  .venv/bin/python -m sumo_experiment.train_state_only_dqn \
  --episodes 1 --seed 42 --episode-horizon-seconds 1800 \
  --decision-interval-seconds 10 --batch-size 64 --output-dir "$out"
```

For each output directory, inspect `wall_clock.txt` for real/user/system
seconds; `training_metrics.csv` for `episode_duration_sec`, `decision_steps`,
termination reason, and (for both DQNs) `learning_updates`; and
`training_config.json` for the episode horizon and agent settings. SUMO's
internal simulation step is 1 second, so simulated internal step count is
`round(episode_duration_sec / 1)`. Q-learning calls its tabular update once per
decision, so its update count equals `decision_steps`; its CSV does not have a
separate `learning_updates` column. DQN `learning_updates` counts optimizer
updates, while `decision_steps` counts transitions supplied to replay.
`dqn_model.pt` and `state_only_dqn_model.pt` are the DQN checkpoint names;
`q_table.npz` is Q-learning's checkpoint.

The earlier short-demand calibration outputs remain unchanged. The sustained
calibration outputs were written to the new directories above. If repeating a
run, its reserved directory must be new because `mkdir` prevents overwriting.

## Sustained-demand calibration results

These are one-episode diagnostics, not evidence of learned-policy quality.
Each episode used seed 42, 1,800-second horizon, 10-second decisions, 1-second
SUMO simulation steps, and the generated sustained route file.

| Controller | Wall time (s) | Simulated duration (s) | SUMO steps | Decisions | Completed vehicles | Termination | Learning updates |
|---|---:|---:|---:|---:|---:|---|---:|
| Q-learning | 5.82 | 1,800 | 1,800 | 180 | 1,432 | horizon | 180 tabular updates |
| Original DQN | 7.79 | 1,800 | 1,800 | 180 | 1,445 | horizon | 117 optimizer updates |
| State-only DQN | 7.24 | 1,800 | 1,800 | 180 | 1,431 | horizon | 117 optimizer updates |

The consolidated records are in
`sumo_experiment/runs/comparison/calibration_sustained/calibration_summary.csv`;
each run directory also contains its checkpoint, training CSV/config, and
`wall_clock.txt`. The SUMO executable/version is recorded in the parent
`sumo_binary_version.txt`.

The route schedule requests 900 departures per direction. A separate live
insertion smoke using SUMO 1.28.0 recorded cumulative inserted counts of 595,
684, 1,170, 1,648, and 1,745 at seconds 600, 700, 1,200, 1,700, and 1,800,
respectively. This confirms insertions continued well beyond the original
600-second demand cutoff. It also shows that insertion feasibility can prevent
the full nominal 1,800 scheduled vehicles from entering before the horizon.

After reviewing the calibration and obtaining approval, the following commands
create isolated outputs for all proposed seeds:

```sh
export SUMO_ROUTES_FILE="$PWD/sumo_experiment/scenarios/periodic_demand_1800s.rou.xml"
for seed in 42 43 44; do
  .venv/bin/python -m sumo_experiment.train_q_learning --episodes 1000 --seed "$seed" \
    --episode-horizon-seconds 1800 --decision-interval-seconds 10 \
    --output-dir "sumo_experiment/runs/comparison/training/q_learning/seed_$seed"
  .venv/bin/python -m sumo_experiment.train_dqn --episodes 500 --seed "$seed" \
    --episode-horizon-seconds 1800 --decision-interval-seconds 10 \
    --output-dir "sumo_experiment/runs/comparison/training/original_dqn/seed_$seed"
  .venv/bin/python -m sumo_experiment.train_state_only_dqn --episodes 500 --seed "$seed" \
    --episode-horizon-seconds 1800 --decision-interval-seconds 10 \
    --output-dir "sumo_experiment/runs/comparison/training/state_only_dqn/seed_$seed"
done
```

After reviewing checkpoints and runtime, run:

```sh
export SUMO_ROUTES_FILE="$PWD/sumo_experiment/scenarios/periodic_demand_1800s.rou.xml"
.venv/bin/python -m sumo_experiment.run_comparison \
  --q-checkpoints sumo_experiment/runs/comparison/training/q_learning/seed_42/q_table.npz \
                  sumo_experiment/runs/comparison/training/q_learning/seed_43/q_table.npz \
                  sumo_experiment/runs/comparison/training/q_learning/seed_44/q_table.npz \
  --dqn-checkpoints sumo_experiment/runs/comparison/training/original_dqn/seed_42/dqn_model.pt \
                    sumo_experiment/runs/comparison/training/original_dqn/seed_43/dqn_model.pt \
                    sumo_experiment/runs/comparison/training/original_dqn/seed_44/dqn_model.pt \
  --state-only-checkpoints sumo_experiment/runs/comparison/training/state_only_dqn/seed_42/state_only_dqn_model.pt \
                           sumo_experiment/runs/comparison/training/state_only_dqn/seed_43/state_only_dqn_model.pt \
                           sumo_experiment/runs/comparison/training/state_only_dqn/seed_44/state_only_dqn_model.pt
```

The harness writes episode metrics, grouped means/standard deviations,
configuration/runtime provenance, plots, and a descriptive Markdown report to
`sumo_experiment/runs/comparison/`. This command runs 30 episodes per listed
checkpoint plus 30 Webster episodes.
