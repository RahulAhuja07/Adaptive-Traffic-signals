# Comparative Analysis Report

## 1. Project objective
Compare tabular Q-learning, DQN, and Webster's fixed-time baseline in a custom two-direction traffic simulation.

## 2. Simulation scenarios and experimental setup
The configured scenarios use expected Poisson arrivals per interval: low (NS 1.0, EW 0.8), medium (NS 2.8, EW 1.2), and high (NS 4.0, EW 3.5). These are synthetic assumptions, not real-world measurements. Part 3 uses common evaluation seeds and the same episode configuration for controllers.

## 3. Data-quality findings
- Evaluation records read: 15
- Required columns present: True
- Recognized controllers: Q-Learning, DQN, Webster
- Waiting-time NaNs are allowed as undefined values and are never replaced with zero.
- Rows excluded from grouped analyses: 0
- NOTE: high: all controllers share 5 scenario/seed combinations.
- NOTE: No records found for scenarios: low, medium.
- No structural or value-range errors were detected.

**Caution:** The validation report contains data-quality issues. Numerical summaries below are based on retained records but should be treated as incomplete or potentially unreliable until the source data is corrected or regenerated.

## 4. Average waiting time
This metric averages waits only for vehicles that departed. It is interpreted alongside throughput, queue, and final-queue results.

| scenario | controller | valid_avg_waiting_time_observations | mean_avg_waiting_time_sec | median_avg_waiting_time_sec | std_avg_waiting_time_sec | p25_avg_waiting_time_sec | p75_avg_waiting_time_sec | ci95_avg_waiting_time_low | ci95_avg_waiting_time_high |
|---|---|---|---|---|---|---|---|---|---|
| high | DQN | 5 | 7.604 | 7.253 | 1.573 | 6.461 | 7.717 | 5.421 | 9.787 |
| high | Q-Learning | 5 | 153.699 | 145.533 | 17.828 | 139.794 | 171.045 | 128.949 | 178.448 |
| high | Webster | 5 | 306.435 | 305.573 | 16.086 | 293.169 | 313.889 | 284.103 | 328.767 |

The 95% confidence interval uses a Student-t interval for the mean when at least two finite observations are available; it assumes independent episode observations and an approximately suitable sampling distribution. Undefined waiting values remain excluded and the valid count is shown.

## 5. Queue length
| scenario | controller | mean_time_weighted_avg_queue | median_time_weighted_avg_queue | std_time_weighted_avg_queue | ci95_time_weighted_avg_queue_low | ci95_time_weighted_avg_queue_high | mean_final_queue | median_final_queue | episodes_with_nonzero_final_queue_percent |
|---|---|---|---|---|---|---|---|---|---|
| high | DQN | 307.401 | 309.500 | 12.440 | 290.131 | 324.671 | 606.800 | 604.000 | 100.000 |
| high | Q-Learning | 212.107 | 205.983 | 16.117 | 189.733 | 234.480 | 414.600 | 426.000 | 100.000 |
| high | Webster | 217.834 | 209.961 | 14.683 | 197.451 | 238.217 | 420.400 | 430.000 | 100.000 |

Lower waiting queues generally indicate less congestion, but queue metrics are not a standalone measure of controller quality.

## 6. Throughput and service
| scenario | controller | mean_throughput_veh_per_min | median_throughput_veh_per_min | std_throughput_veh_per_min | mean_total_arrived | mean_total_departed | mean_final_queue | mean_departure_to_arrival_ratio |
|---|---|---|---|---|---|---|---|---|
| high | DQN | 23.533 | 23.367 | 1.098 | 1312.800 | 706.000 | 606.800 | 0.538 |
| high | Q-Learning | 29.940 | 29.933 | 0.033 | 1312.800 | 898.200 | 414.600 | 0.684 |
| high | Webster | 29.747 | 29.767 | 0.105 | 1312.800 | 892.400 | 420.400 | 0.680 |

Departure-to-arrival ratio is computed episode by episode as total departures / total arrivals, excluding episodes with zero arrivals, then averaged. Throughput is interpreted with final queue and total service.

## 7. Improvements relative to Webster
Absolute difference is RL mean minus Webster mean. Percentage improvement uses `(Webster - RL) / Webster * 100` for lower-is-better metrics, and `(RL - Webster) / Webster * 100` for throughput. A zero or undefined Webster mean yields an undefined percentage.

| scenario | rl_controller | metric | rl_mean | baseline_mean | absolute_difference | improvement_percent |
|---|---|---|---|---|---|---|
| high | Q-Learning | avg_waiting_time_sec | 153.699 | 306.435 | -152.737 | 49.843 |
| high | Q-Learning | time_weighted_avg_queue | 212.107 | 217.834 | -5.728 | 2.629 |
| high | Q-Learning | final_queue | 414.600 | 420.400 | -5.800 | 1.380 |
| high | Q-Learning | throughput_veh_per_min | 29.940 | 29.747 | 0.193 | 0.650 |
| high | DQN | avg_waiting_time_sec | 7.604 | 306.435 | -298.831 | 97.519 |
| high | DQN | time_weighted_avg_queue | 307.401 | 217.834 | 89.567 | -41.117 |
| high | DQN | final_queue | 606.800 | 420.400 | 186.400 | -44.339 |
| high | DQN | throughput_veh_per_min | 23.533 | 29.747 | -6.213 | -20.887 |

## 8. Statistical comparisons
Tests match rows by scenario and seed where at least two valid pairs are available. Paired t-tests are used when Shapiro-Wilk does not flag the differences as non-normal; otherwise the Wilcoxon signed-rank test is used. Unmatched samples use Welch's t-test when both groups pass the same normality screen, otherwise Mann-Whitney U. Holm correction is applied across the three controller-pair tests within each scenario/metric family. A p-value is evidence against equal distributions under test assumptions; it does not measure practical importance.

| scenario | metric | controller_a | controller_b | sample_size | comparison_design | test_used | test_statistic | raw_p_value | holm_adjusted_p_value | mean_difference_a_minus_b | ci95_difference_low | ci95_difference_high | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high | avg_waiting_time_sec | Q-Learning | Webster | 5 | paired by seed | paired t-test | -21.748 | 0.000 | 0.000 | -152.737 | -172.236 | -133.237 |  |
| high | avg_waiting_time_sec | DQN | Webster | 5 | paired by seed | paired t-test | -35.797 | 0.000 | 0.000 | -298.831 | -322.009 | -275.653 |  |
| high | avg_waiting_time_sec | Q-Learning | DQN | 5 | paired by seed | paired t-test | 15.629 | 0.000 | 0.000 | 146.095 | 120.141 | 172.048 |  |
| high | time_weighted_avg_queue | Q-Learning | Webster | 5 | paired by seed | paired t-test | -3.786 | 0.019 | 0.019 | -5.728 | -9.928 | -1.527 |  |
| high | time_weighted_avg_queue | DQN | Webster | 5 | paired by seed | paired t-test | 10.180 | 0.001 | 0.002 | 89.567 | 65.139 | 113.994 |  |
| high | time_weighted_avg_queue | Q-Learning | DQN | 5 | paired by seed | paired t-test | -9.420 | 0.001 | 0.002 | -95.294 | -123.381 | -67.208 |  |
| high | throughput_veh_per_min | Q-Learning | Webster | 5 | paired by seed | paired t-test | 3.792 | 0.019 | 0.019 | 0.193 | 0.052 | 0.335 |  |
| high | throughput_veh_per_min | DQN | Webster | 5 | paired by seed | paired t-test | -12.048 | 0.000 | 0.001 | -6.213 | -7.645 | -4.781 |  |
| high | throughput_veh_per_min | Q-Learning | DQN | 5 | paired by seed | paired t-test | 11.485 | 0.000 | 0.001 | 6.407 | 4.858 | 7.955 |  |
| high | final_queue | Q-Learning | Webster | 5 | paired by seed | paired t-test | -3.792 | 0.019 | 0.019 | -5.800 | -10.047 | -1.553 |  |
| high | final_queue | DQN | Webster | 5 | paired by seed | paired t-test | 12.048 | 0.000 | 0.001 | 186.400 | 143.445 | 229.355 |  |
| high | final_queue | Q-Learning | DQN | 5 | paired by seed | paired t-test | -11.485 | 0.000 | 0.001 | -192.200 | -238.662 | -145.738 |  |

## 9. Controller rankings by scenario and metric
Rank 1 is best for the listed metric direction; ties receive average ranks. No single overall winner is calculated.

| scenario | metric | direction | rank | controller | mean | std | valid_observations |
|---|---|---|---|---|---|---|---|
| high | avg_waiting_time_sec | lower | 1.000 | DQN | 7.604 | 1.573 | 5 |
| high | avg_waiting_time_sec | lower | 2.000 | Q-Learning | 153.699 | 17.828 | 5 |
| high | avg_waiting_time_sec | lower | 3.000 | Webster | 306.435 | 16.086 | 5 |
| high | time_weighted_avg_queue | lower | 3.000 | DQN | 307.401 | 12.440 | 5 |
| high | time_weighted_avg_queue | lower | 1.000 | Q-Learning | 212.107 | 16.117 | 5 |
| high | time_weighted_avg_queue | lower | 2.000 | Webster | 217.834 | 14.683 | 5 |
| high | final_queue | lower | 3.000 | DQN | 606.800 | 15.510 | 5 |
| high | final_queue | lower | 1.000 | Q-Learning | 414.600 | 26.971 | 5 |
| high | final_queue | lower | 2.000 | Webster | 420.400 | 24.937 | 5 |
| high | throughput_veh_per_min | higher | 3.000 | DQN | 23.533 | 1.098 | 5 |
| high | throughput_veh_per_min | higher | 1.000 | Q-Learning | 29.940 | 0.033 | 5 |
| high | throughput_veh_per_min | higher | 2.000 | Webster | 29.747 | 0.105 | 5 |
| high | cumulative_reward | higher | 3.000 | DQN | -8250.730 | undefined |  |
| high | cumulative_reward | higher | 2.000 | Q-Learning | -5486.350 | undefined |  |
| high | cumulative_reward | higher | 1.000 | Webster | -4135.850 | undefined |  |

## 10. Scenario observations
- Low demand: no summary data available.
- Medium demand: no summary data available.
- High demand: DQN had the lowest observed mean departed-vehicle waiting time (7.604 seconds); consider this with queue and service metrics.
  Q-Learning had the lowest observed time-weighted average queue (212.107 vehicles). These descriptive extrema are not claims of statistical superiority.

## 11. Training convergence
- DQN/high: first 10-episode mean reward -4450.995; last 10-episode mean reward -4450.995; observed change +0.000. This trend alone does not establish stable convergence.
- Q-learning/high: first 10-episode mean reward -4662.740; last 10-episode mean reward -4662.740; observed change +0.000. This trend alone does not establish stable convergence.

## 12. Limitations and interpretation
Results describe this simplified, synthetic environment only. The simulator has no lanes, turning movements, pedestrian demand, yellow/all-red phases, or calibrated arrival/service data. Webster receives configured demand as prior information, whereas Q-learning and DQN learn from reward feedback. Statistical significance does not imply a practically important traffic benefit. Real-world claims require validation using realistic traffic data and a calibrated traffic simulator.

## 13. Generated visualizations
- Generated waiting_time_by_scenario.png.
- Generated queue_length_by_scenario.png.
- Generated throughput_by_scenario.png.
- Generated final_queue_by_scenario.png.
- Generated reward_by_scenario.png.
- Generated performance_heatmap.png; scores are min-max normalized within each scenario/metric, with direction inverted for lower-is-better measures.
- Generated training_convergence.png from Part 3 training logs.
