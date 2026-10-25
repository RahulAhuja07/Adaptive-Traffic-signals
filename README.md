# Adaptive Traffic Signal Optimization

This project provides a small, reproducible traffic simulation for later
controller comparisons. It models an intersection with two directions,
North–South (NS) and East–West (EW). Part 1 implements the environment; Part 2
adds tabular Q-learning, a compact DQN, and a Webster-style fixed-time baseline.
Part 3 adds reproducible training, paired-seed evaluation, CSV output, and
comparison plots. SUMO is not integrated.

## Model

Each episode has 180 intervals by default. An interval represents 10 seconds.
NS and EW arrivals are independently sampled from Poisson distributions with
expected counts of 2.8 and 1.2 vehicles per interval. These are synthetic
assumptions for this simplified simulator, not real intersection measurements.
Each arriving vehicle is stored in a FIFO `deque` with a timestamp sampled
within its interval. The active green direction can serve up to five vehicles
per interval. Departures are assigned the interval's end time, and vehicle
waiting time is departure time minus arrival time.

There are exactly two actions: `0` gives NS green and `1` gives EW green. The
discrete state has 18 possible values. Each direction's queue is binned as low
(0–4 vehicles, encoded 0), medium (5–9, encoded 1), or high (10+, encoded 2).
The state formula is `phase * 9 + ns_bin * 3 + ew_bin`, where phase is 0 for NS
green and 1 for EW green.

The reward is `-((queue_before + queue_after) / 2.0) / 10.0`. Here,
`queue_before` is the total queue before this interval's arrivals, and
`queue_after` is the total remaining after arrivals and service. It is a simple
queue-length penalty, not a direct sum or average of individual vehicle waits.

`get_metrics()` reports mean waiting time among departed vehicles (NaN if none
have departed), the time-weighted average queue, throughput in vehicles per
minute, cumulative arrivals and departures, the current final queue, and
cumulative reward. Average queue length uses the queue after arrivals and
service in each completed interval. Throughput is zero before simulated time
has elapsed.

## Assumptions and limitations

- One queue represents each direction; there are no lanes or turn movements.
- No yellow or all-red phases, pedestrian crossings, or lost switching time.
- Every interval uses the selected green phase, with a fixed service capacity.
- Arrivals use stationary Poisson rates, and this model is not calibrated to
  real-world traffic.
- Waiting times use each served vehicle's timestamp and the interval-end
  departure convention.

These limitations keep the shared controller interface straightforward.
Controllers call `reset(seed=...)`, select actions for `step(action)`, and
compare their results through the same state, reward, and metrics.

## Part 2: Controllers

### Tabular Q-learning

`QLearningAgent` keeps a table with one row for each of the 18 states and one
column for each signal action. It chooses actions with an epsilon-greedy
policy: early training explores more, then `decay_epsilon()` reduces
exploration once per episode. The update uses the immediate reward plus the
best estimated future value, except at episode termination where there is no
future value. Save and restore a table with `agent.save("q_table.npz")` and
`agent.load("q_table.npz")`.

### Deep Q-Network

`DQNAgent` estimates the two action values with a small PyTorch neural network.
It learns from batches sampled from a replay buffer and uses a separately
updated target network to calculate stable targets. Each integer state is
encoded as an 18-element one-hot vector. This gives DQN the same state
information as tabular Q-learning. Since the state space is small, DQN may not
outperform the table-based method; that is a valid result. Save and restore a
checkpoint with `agent.save("dqn_model.pt")` and `agent.load("dqn_model.pt")`.

Example interaction for either learning agent:

```python
state = env.reset(seed=7)
done = False
while not done:
    action = agent.choose_action(state, training=True)
    next_state, reward, done, info = env.step(action)
    agent.update(state, action, reward, next_state, done)
    state = next_state
agent.decay_epsilon()
```

The DQN consumes integer states directly and one-hot encodes them internally.
The Q-learning agent updates its table directly. Both agents expose
`choose_action`, `update`, `decay_epsilon`, `save`, and `load` methods.

### Webster-style fixed-time baseline

`WebsterController` is a conventional timing baseline, not an RL agent. It
converts the configured expected arrivals per interval to vehicles per hour
using `q = lambda / interval_seconds * 3600`, then divides by assumed
saturation flow to estimate each critical flow ratio. It applies
`C0 = (1.5L + 5) / (1 - Y)` when total critical flow `Y` is below one, and
allocates effective green time in proportion to the phase ratios. The default
synthetic saturation flows are 1,800 vehicles/hour per direction, with four
seconds of lost time per cycle. These illustrative inputs are not measured
traffic values.

If demand is zero, the baseline uses its minimum cycle and splits green time
equally. If total critical flow is at least one, Webster's formula is unsuitable;
the controller uses the configured maximum cycle and allocates green in
proportion to demand. Cycle lengths are clipped to configured bounds and each
phase gets at least one whole simulation interval after rounding. Construct a
controller with the scenario's known `arrival_rates` and environment's
`interval_seconds`, call `reset()` at the start of each episode, then call
`choose_action()` once per environment interval.

Webster's controller receives configured scenario demand as prior information.
Q-learning and DQN instead learn from reward feedback. This difference matters
when interpreting comparisons: the baseline is not given the same learning
process or information constraints as the RL agents.

## Dependencies and tests

The required external packages are NumPy, PyTorch, Matplotlib, and SciPy. For the simplest
cross-platform setup, install the requirements in the active virtual
environment. PyTorch's recommended command varies by operating system and
hardware; use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
if you need a specific CPU or accelerator build. On Linux or Windows, the CPU
wheel can be installed with the CPU index when that wheel matches your Python
and platform:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On macOS, use the pip command provided for macOS in the selector. Then install
the project dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m unittest -v
```

Run all tests from VS Code's project terminal with `python -m unittest -v`.
The tests cover the environment, controllers, and a small training/evaluation
pipeline run.

## Part 3: Training and evaluation

`train.py` trains Q-learning and DQN separately for each selected demand
scenario. The default is 5,000 episodes per algorithm and scenario, with 180
intervals per episode and seed 42. The traffic rates are expected Poisson
arrivals per interval; the sampled count changes from interval to interval.
The scenarios are:

| Scenario | NS expected arrivals/interval | EW expected arrivals/interval |
| --- | ---: | ---: |
| Low | 1.0 | 0.8 |
| Medium | 2.8 | 1.2 |
| High | 4.0 | 3.5 |

Training explores to learn a policy. Evaluation in `evaluate.py` loads saved
models and uses greedy action selection; it does not update Q-values, DQN
weights, replay memory, or exploration rates. Webster receives the scenario's
configured demand as prior information to calculate its timing plan, while the
RL controllers learn from reward feedback. This information difference should
be considered when interpreting results.

Evaluation runs 30 episodes by default. It uses paired episode seeds: each
controller sees the same seed for a given scenario/episode. The current
environment samples arrivals from one seeded random stream and consumes those
arrival draws independently of actions, so those paired runs use matching
arrival counts and timestamps. Actions still change queues, departures, and
waiting times. If no vehicles depart in an episode, average waiting time is
undefined and remains NaN in the CSV. Also read it alongside final queue,
departures, and average queue because departed-vehicle waiting time alone can
look deceptively low when a controller leaves vehicles behind.

The evaluation summary reports means and population standard deviations,
ignoring undefined values; it also includes the number of valid waiting-time
observations. The high-demand scenario may produce growing queues under the
fixed per-interval service capacity. That is an outcome to report, not a reason
to alter or hide the run.

### Run a short debugging experiment

From the VS Code terminal, train 100 episodes per algorithm for the medium
scenario, then evaluate five paired episodes using the same model seed:

```bash
python train.py --episodes 100 --seed 42 --scenario medium
python evaluate.py --episodes 5 --seed 42 --model-seed 42 --scenario medium
```

You can also shorten episodes while debugging. Use the same `--episode-steps`
value for training and evaluation, for example `--episode-steps 20`.

### Run the full configured experiment

The default command trains 5,000 episodes for both learning agents in all three
scenarios. Evaluation then runs 30 episodes per controller in each scenario:

```bash
python train.py --episodes 5000 --seed 42 --scenario all
python evaluate.py --episodes 30 --seed 42 --model-seed 42 --scenario all
```

Training logs are saved under `results/`, and trained Q-tables/checkpoints are
saved under `models/`. Webster has no learned model file. Evaluation writes
`results/evaluation_results.csv` (one row per episode and controller) and
`results/summary_results.csv` (scenario/controller aggregates). Plots are saved
as PNG files under `plots/`.

Evaluation generates plots after writing the CSVs. To regenerate all six plots
from existing CSV files without retraining or reevaluating, run:

```bash
python plot_results.py
```

The training-reward plots include raw episode rewards and a rolling mean using
up to the most recent 100 episodes. The other plots compare average waiting
time (with standard-deviation error bars), time-weighted average queue,
throughput, and final queue. These initial plots do not establish that one
controller is better; interpret them with the episode-level data and the
Webster/RL information difference in mind.

## Part 4: Comparative analysis

Run the analysis after Part 3 has produced `results/evaluation_results.csv`:

```bash
python analyze_results.py
```

The evaluation CSV is the source of truth. The analysis checks required
columns, duplicate or missing episode records, controller/scenario names,
value ranges, traffic accounting, episode counts, and shared scenario/seed
sets. A specific expected evaluation count can be checked with, for example,
`python analyze_results.py --expected-episodes 30`. Data issues and any
structurally excluded rows are listed in `results/data_validation_report.txt`.
Waiting-time NaNs remain undefined; they are not changed to zero. If the
evaluation CSV is missing or its required columns are incomplete, the workflow
writes a clear report template and empty analysis tables rather than inventing
experimental results. Other quality problems are reported and flagged while
available metric summaries remain traceable to their source rows.

The analysis reports mean, median, standard deviation, range, quartiles, and
95% t confidence intervals where there are at least two valid observations.
Waiting time is measured only among vehicles that departed, so it must be read
alongside average queue, throughput, departures, and final queue. Throughput
and departure-to-arrival ratio are summarized episode by episode. RL percentage
improvement relative to Webster is `(Webster - RL) / Webster * 100` for lower
is better metrics, and `(RL - Webster) / Webster * 100` for throughput. A zero
or undefined baseline yields an undefined percentage. Absolute differences
are separately reported as RL mean minus Webster mean.

When scenario/seed records are matched, controller tests use paired differences.
Depending on the paired-difference normality check, the workflow uses a paired
t-test or Wilcoxon signed-rank test. If records are not sufficiently matched,
it uses an independent test only on unmatched seed sets. Holm correction
adjusts p-values across the three controller-pair tests within each
scenario/metric family. A p-value describes evidence under the test
assumptions; it is not the size or practical importance of a traffic benefit.
The report also includes metric-by-metric controller rankings, not one
arbitrary overall winner.

The Shapiro-Wilk check has limited power with small episode counts. The
Wilcoxon signed-rank test compares paired ranks and its location interpretation
works best when paired differences are reasonably symmetric; it can also have
low power on small samples. Confidence intervals and test results should
therefore be interpreted with the sample size and design, not as guarantees.

Outputs are written to `results/`: `data_validation_report.txt`,
`analysis_report.md`, expanded `summary_results.csv`, `statistical_tests.csv`,
`improvement_summary.csv`, and `controller_rankings.csv`. Report-ready plots
are saved to `plots/`: waiting time, average queue, throughput, final queue,
cumulative reward, a normalized performance heatmap, and training convergence
when Part 3 training logs are available. Heatmap scores are min-max normalized
within each scenario/metric column; lower-is-better metrics are inverted so
higher scores mean a better observed value. The heatmap is descriptive and is
not a standalone selection rule.

The custom simulator uses synthetic Poisson arrivals and simplified queues and
signal service. These results cannot establish real-world traffic performance
without realistic traffic data and validation against a calibrated simulator
or field observations.

## Set up in VS Code

Open this folder in VS Code, then open **Terminal → New Terminal**. Create and
activate a virtual environment from the project folder.

**Windows PowerShell:**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
```

If VS Code asks which interpreter to use, select the Python interpreter inside
`.venv`. The tests use Python's built-in `unittest` framework.
