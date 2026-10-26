# Adaptive Traffic Signal Control with RL

A reproducible, custom two-direction intersection simulator comparing tabular Q-learning, DQN, a richer-state DQN, and Webster timing. Arrivals are synthetic Poisson traffic; SUMO is not used.

## Install

Requires Python 3.10 or newer. Dependencies: NumPy, PyTorch, Matplotlib, and SciPy (`requirements.txt`). From the project root:

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
```

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v
```

If PyTorch installation fails on your system, install the appropriate build using the [PyTorch selector](https://pytorch.org/get-started/locally/).

## Run the baseline experiment

The scenarios use expected NS/EW arrivals per 10-second interval: low `(1.0, 0.8)`, medium `(2.8, 1.2)`, high `(4.0, 3.5)`. Each episode is 180 intervals; service capacity is five vehicles per green interval.

Train both RL agents for 100 episodes per scenario, then evaluate all controllers on 30 paired episodes:

```bash
python train.py --episodes 100 --seed 42 --scenario all
python evaluate.py --episodes 30 --seed 42 --model-seed 42 --scenario all
python analyze_results.py --expected-episodes 30
```

For the configured longer training run, use `--episodes 5000`. Evaluation is greedy and does not update the trained agents. Results are written to `models/`, `results/`, and `plots/`. Waiting time includes departed vehicles only; interpret it alongside queues, throughput, final queue, and directional service. The high-demand case is overloaded (45 expected arrivals/minute versus at most 30 vehicles/minute service).

## Reproduce the richer-state DQN comparison

With the baseline checkpoints available in `models/`, run:

```bash
python runs/state_only_dqn_extended/run_experiment.py \
  --train-episodes 300 --eval-episodes 30 \
  --train-seeds 7 21 42 --scenarios low medium high
```

This trains the six-feature state-only DQN and compares it with the saved Q-learning, original DQN, and Webster controllers using paired evaluation seeds. Outputs are saved under `runs/state_only_dqn_extended/`. The reported comparison uses the seed-42 state-only checkpoint; original checkpoints were trained for 100 episodes, while the state-only checkpoint was trained for 300.

To run the separate directional-reward pilot, use a distinct output directory so baseline files remain untouched:

```bash
python train.py --episodes 10 --seed 42 --scenario high \
  --reward-mode directional --output-dir runs/directional_reward_pilot
python evaluate.py --episodes 5 --seed 42 --model-seed 42 --scenario high \
  --reward-mode directional --output-dir runs/directional_reward_pilot --no-plots
python analyze_results.py --output-dir runs/directional_reward_pilot \
  --expected-episodes 5
```

## Main files

- `environment.py`: simulator and baseline reward.
- `q_learning.py`, `dqn_agent.py`, `webster_controller.py`: controllers.
- `train.py`, `evaluate.py`: baseline training and evaluation.
- `analyze_results.py`, `statistical_analysis.py`: validation and result analysis.
- `runs/state_only_dqn_extended/run_experiment.py`: richer-state comparison.
- `results/analysis_report.md`: generated baseline analysis report.
