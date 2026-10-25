# Comparative Analysis Report

## 1. Project objective
Compare tabular Q-learning, DQN, and Webster's fixed-time baseline in a custom two-direction traffic simulation.

## 2. Simulation scenarios and experimental setup
The configured scenarios use expected Poisson arrivals per interval: low (NS 1.0, EW 0.8), medium (NS 2.8, EW 1.2), and high (NS 4.0, EW 3.5). These are synthetic assumptions, not real-world measurements. Part 3 uses common evaluation seeds and the same episode configuration for controllers.

## 3. Data-quality findings
- Evaluation records read: 45
- Required columns present: True
- Recognized controllers: Q-Learning, DQN, Webster
- Waiting-time NaNs are allowed as undefined values and are never replaced with zero.
- Rows excluded from grouped analyses: 0
- NOTE: high: all controllers share 5 scenario/seed combinations.
- NOTE: low: all controllers share 5 scenario/seed combinations.
- NOTE: medium: all controllers share 5 scenario/seed combinations.
- NOTE: No intended episode count was supplied; count consistency is reported, but the requested count cannot be verified.
- No structural or value-range errors were detected.

## 4. Average waiting time
This metric averages waits only for vehicles that departed. It is interpreted alongside throughput, queue, and final-queue results.

| scenario | controller | valid_avg_waiting_time_observations | mean_avg_waiting_time_sec | median_avg_waiting_time_sec | std_avg_waiting_time_sec | p25_avg_waiting_time_sec | p75_avg_waiting_time_sec | ci95_avg_waiting_time_low | ci95_avg_waiting_time_high |
|---|---|---|---|---|---|---|---|---|---|
| high | DQN | 5 | 7.908 | 7.253 | 1.550 | 6.981 | 7.717 | 5.756 | 10.059 |
| high | Q-Learning | 5 | 19.266 | 14.228 | 10.587 | 12.819 | 25.620 | 4.569 | 33.964 |
| high | Webster | 5 | 306.435 | 305.573 | 16.086 | 293.169 | 313.889 | 284.103 | 328.767 |
| low | DQN | 5 | 10.540 | 10.574 | 0.167 | 10.373 | 10.669 | 10.308 | 10.771 |
| low | Q-Learning | 5 | 10.540 | 10.574 | 0.167 | 10.373 | 10.669 | 10.308 | 10.771 |
| low | Webster | 5 | 10.190 | 10.061 | 0.355 | 9.952 | 10.179 | 9.697 | 10.683 |
| medium | DQN | 5 | 16.501 | 16.151 | 0.763 | 15.932 | 16.896 | 15.441 | 17.561 |
| medium | Q-Learning | 5 | 17.264 | 17.194 | 0.682 | 16.539 | 17.837 | 16.318 | 18.211 |
| medium | Webster | 5 | 18.953 | 18.948 | 1.101 | 18.270 | 19.685 | 17.425 | 20.481 |

The 95% confidence interval uses a Student-t interval for the mean when at least two finite observations are available; it assumes independent episode observations and an approximately suitable sampling distribution. Undefined waiting values remain excluded and the valid count is shown.

## 5. Queue length
| scenario | controller | mean_time_weighted_avg_queue | median_time_weighted_avg_queue | std_time_weighted_avg_queue | ci95_time_weighted_avg_queue_low | ci95_time_weighted_avg_queue_high | mean_final_queue | median_final_queue | episodes_with_nonzero_final_queue_percent |
|---|---|---|---|---|---|---|---|---|---|
| high | DQN | 302.537 | 299.783 | 12.589 | 285.060 | 320.013 | 601.800 | 595.000 | 100.000 |
| high | Q-Learning | 311.177 | 327.967 | 27.640 | 272.806 | 349.547 | 653.200 | 674.000 | 100.000 |
| high | Webster | 217.834 | 209.961 | 14.683 | 197.451 | 238.217 | 420.400 | 430.000 | 100.000 |
| low | DQN | 0.986 | 0.972 | 0.053 | 0.912 | 1.059 | 0.600 | 0.000 | 40.000 |
| low | Q-Learning | 0.986 | 0.972 | 0.053 | 0.912 | 1.059 | 0.600 | 0.000 | 40.000 |
| low | Webster | 0.926 | 0.900 | 0.082 | 0.812 | 1.039 | 1.000 | 1.000 | 80.000 |
| medium | DQN | 4.498 | 4.339 | 0.321 | 4.051 | 4.944 | 4.400 | 4.000 | 100.000 |
| medium | Q-Learning | 4.798 | 4.533 | 0.379 | 4.271 | 5.325 | 4.200 | 3.000 | 80.000 |
| medium | Webster | 5.488 | 5.206 | 0.573 | 4.692 | 6.284 | 8.400 | 10.000 | 100.000 |

Lower waiting queues generally indicate less congestion, but queue metrics are not a standalone measure of controller quality.

## 6. Throughput and service
| scenario | controller | mean_throughput_veh_per_min | median_throughput_veh_per_min | std_throughput_veh_per_min | mean_total_arrived | mean_total_departed | mean_final_queue | mean_departure_to_arrival_ratio |
|---|---|---|---|---|---|---|---|---|
| high | DQN | 23.700 | 23.367 | 0.990 | 1312.800 | 711.000 | 601.800 | 0.541 |
| high | Q-Learning | 21.987 | 21.967 | 0.998 | 1312.800 | 659.600 | 653.200 | 0.503 |
| high | Webster | 29.747 | 29.767 | 0.105 | 1312.800 | 892.400 | 420.400 | 0.680 |
| low | DQN | 10.887 | 11.000 | 0.490 | 327.200 | 326.600 | 0.600 | 0.998 |
| low | Q-Learning | 10.887 | 11.000 | 0.490 | 327.200 | 326.600 | 0.600 | 0.998 |
| low | Webster | 10.873 | 11.000 | 0.493 | 327.200 | 326.200 | 1.000 | 0.997 |
| medium | DQN | 23.280 | 23.333 | 0.668 | 702.800 | 698.400 | 4.400 | 0.994 |
| medium | Q-Learning | 23.287 | 23.233 | 0.611 | 702.800 | 698.600 | 4.200 | 0.994 |
| medium | Webster | 23.147 | 23.167 | 0.559 | 702.800 | 694.400 | 8.400 | 0.988 |

Departure-to-arrival ratio is computed episode by episode as total departures / total arrivals, excluding episodes with zero arrivals, then averaged. Throughput is interpreted with final queue and total service.

## 7. Improvements relative to Webster
Absolute difference is RL mean minus Webster mean. Percentage improvement uses `(Webster - RL) / Webster * 100` for lower-is-better metrics, and `(RL - Webster) / Webster * 100` for throughput. A zero or undefined Webster mean yields an undefined percentage.

| scenario | rl_controller | metric | rl_mean | baseline_mean | absolute_difference | improvement_percent |
|---|---|---|---|---|---|---|
| high | Q-Learning | avg_waiting_time_sec | 19.266 | 306.435 | -287.169 | 93.713 |
| high | Q-Learning | time_weighted_avg_queue | 311.177 | 217.834 | 93.342 | -42.850 |
| high | Q-Learning | final_queue | 653.200 | 420.400 | 232.800 | -55.376 |
| high | Q-Learning | throughput_veh_per_min | 21.987 | 29.747 | -7.760 | -26.087 |
| high | DQN | avg_waiting_time_sec | 7.908 | 306.435 | -298.527 | 97.419 |
| high | DQN | time_weighted_avg_queue | 302.537 | 217.834 | 84.702 | -38.884 |
| high | DQN | final_queue | 601.800 | 420.400 | 181.400 | -43.149 |
| high | DQN | throughput_veh_per_min | 23.700 | 29.747 | -6.047 | -20.327 |
| low | Q-Learning | avg_waiting_time_sec | 10.540 | 10.190 | 0.350 | -3.433 |
| low | Q-Learning | time_weighted_avg_queue | 0.986 | 0.926 | 0.060 | -6.483 |
| low | Q-Learning | final_queue | 0.600 | 1.000 | -0.400 | 40.000 |
| low | Q-Learning | throughput_veh_per_min | 10.887 | 10.873 | 0.013 | 0.123 |
| low | DQN | avg_waiting_time_sec | 10.540 | 10.190 | 0.350 | -3.433 |
| low | DQN | time_weighted_avg_queue | 0.986 | 0.926 | 0.060 | -6.483 |
| low | DQN | final_queue | 0.600 | 1.000 | -0.400 | 40.000 |
| low | DQN | throughput_veh_per_min | 10.887 | 10.873 | 0.013 | 0.123 |
| medium | Q-Learning | avg_waiting_time_sec | 17.264 | 18.953 | -1.688 | 8.909 |
| medium | Q-Learning | time_weighted_avg_queue | 4.798 | 5.488 | -0.690 | 12.573 |
| medium | Q-Learning | final_queue | 4.200 | 8.400 | -4.200 | 50.000 |
| medium | Q-Learning | throughput_veh_per_min | 23.287 | 23.147 | 0.140 | 0.605 |
| medium | DQN | avg_waiting_time_sec | 16.501 | 18.953 | -2.452 | 12.936 |
| medium | DQN | time_weighted_avg_queue | 4.498 | 5.488 | -0.990 | 18.040 |
| medium | DQN | final_queue | 4.400 | 8.400 | -4.000 | 47.619 |
| medium | DQN | throughput_veh_per_min | 23.280 | 23.147 | 0.133 | 0.576 |

## 8. Statistical comparisons
Tests match rows by scenario and seed where at least two valid pairs are available. Paired t-tests are used when Shapiro-Wilk does not flag the differences as non-normal; otherwise the Wilcoxon signed-rank test is used. Unmatched samples use Welch's t-test when both groups pass the same normality screen, otherwise Mann-Whitney U. Holm correction is applied across the three controller-pair tests within each scenario/metric family. A p-value is evidence against equal distributions under test assumptions; it does not measure practical importance.

| scenario | metric | controller_a | controller_b | sample_size | comparison_design | test_used | test_statistic | raw_p_value | holm_adjusted_p_value | mean_difference_a_minus_b | ci95_difference_low | ci95_difference_high | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high | avg_waiting_time_sec | Q-Learning | Webster | 5 | paired by seed | paired t-test | -27.267 | 0.000 | 0.000 | -287.169 | -316.409 | -257.928 |  |
| high | avg_waiting_time_sec | DQN | Webster | 5 | paired by seed | paired t-test | -35.356 | 0.000 | 0.000 | -298.527 | -321.970 | -275.085 |  |
| high | avg_waiting_time_sec | Q-Learning | DQN | 5 | paired by seed | paired t-test | 2.216 | 0.091 | 0.091 | 11.359 | -2.870 | 25.588 |  |
| high | time_weighted_avg_queue | Q-Learning | Webster | 5 | paired by seed | paired t-test | 8.957 | 0.001 | 0.002 | 93.342 | 64.408 | 122.276 |  |
| high | time_weighted_avg_queue | DQN | Webster | 5 | paired by seed | paired t-test | 12.178 | 0.000 | 0.001 | 84.702 | 65.391 | 104.013 |  |
| high | time_weighted_avg_queue | Q-Learning | DQN | 5 | paired by seed | paired t-test | 0.606 | 0.577 | 0.577 | 8.640 | -30.923 | 48.203 |  |
| high | throughput_veh_per_min | Q-Learning | Webster | 5 | paired by seed | paired t-test | -16.179 | 0.000 | 0.000 | -7.760 | -9.092 | -6.428 |  |
| high | throughput_veh_per_min | DQN | Webster | 5 | paired by seed | paired t-test | -13.113 | 0.000 | 0.000 | -6.047 | -7.327 | -4.766 |  |
| high | throughput_veh_per_min | Q-Learning | DQN | 5 | paired by seed | paired t-test | -2.194 | 0.093 | 0.093 | -1.713 | -3.881 | 0.455 |  |
| high | final_queue | Q-Learning | Webster | 5 | paired by seed | paired t-test | 16.179 | 0.000 | 0.000 | 232.800 | 192.850 | 272.750 |  |
| high | final_queue | DQN | Webster | 5 | paired by seed | paired t-test | 13.113 | 0.000 | 0.000 | 181.400 | 142.993 | 219.807 |  |
| high | final_queue | Q-Learning | DQN | 5 | paired by seed | paired t-test | 2.194 | 0.093 | 0.093 | 51.400 | -13.640 | 116.440 |  |
| low | avg_waiting_time_sec | Q-Learning | Webster | 5 | paired by seed | paired t-test | 1.439 | 0.224 | 0.671 | 0.350 | -0.325 | 1.025 |  |
| low | avg_waiting_time_sec | DQN | Webster | 5 | paired by seed | paired t-test | 1.439 | 0.224 | 0.671 | 0.350 | -0.325 | 1.025 |  |
| low | avg_waiting_time_sec | Q-Learning | DQN | 5 | paired by seed | Wilcoxon signed-rank (all differences zero) | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |  |
| low | time_weighted_avg_queue | Q-Learning | Webster | 5 | paired by seed | paired t-test | 1.318 | 0.258 | 0.774 | 0.060 | -0.066 | 0.186 |  |
| low | time_weighted_avg_queue | DQN | Webster | 5 | paired by seed | paired t-test | 1.318 | 0.258 | 0.774 | 0.060 | -0.066 | 0.186 |  |
| low | time_weighted_avg_queue | Q-Learning | DQN | 5 | paired by seed | Wilcoxon signed-rank (all differences zero) | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |  |
| low | throughput_veh_per_min | Q-Learning | Webster | 5 | paired by seed | Wilcoxon signed-rank | 0.000 | 0.500 | 1.000 | 0.013 | -0.009 | 0.036 |  |
| low | throughput_veh_per_min | DQN | Webster | 5 | paired by seed | Wilcoxon signed-rank | 0.000 | 0.500 | 1.000 | 0.013 | -0.009 | 0.036 |  |
| low | throughput_veh_per_min | Q-Learning | DQN | 5 | paired by seed | Wilcoxon signed-rank (all differences zero) | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |  |
| low | final_queue | Q-Learning | Webster | 5 | paired by seed | Wilcoxon signed-rank | 0.000 | 0.500 | 1.000 | -0.400 | -1.080 | 0.280 |  |
| low | final_queue | DQN | Webster | 5 | paired by seed | Wilcoxon signed-rank | 0.000 | 0.500 | 1.000 | -0.400 | -1.080 | 0.280 |  |
| low | final_queue | Q-Learning | DQN | 5 | paired by seed | Wilcoxon signed-rank (all differences zero) | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |  |
| medium | avg_waiting_time_sec | Q-Learning | Webster | 5 | paired by seed | paired t-test | -6.997 | 0.002 | 0.007 | -1.688 | -2.359 | -1.018 |  |
| medium | avg_waiting_time_sec | DQN | Webster | 5 | paired by seed | paired t-test | -3.976 | 0.016 | 0.033 | -2.452 | -4.164 | -0.740 |  |
| medium | avg_waiting_time_sec | Q-Learning | DQN | 5 | paired by seed | Wilcoxon signed-rank | 0.000 | 0.062 | 0.062 | 0.763 | -0.471 | 1.998 |  |
| medium | time_weighted_avg_queue | Q-Learning | Webster | 5 | paired by seed | paired t-test | -6.307 | 0.003 | 0.010 | -0.690 | -0.994 | -0.386 |  |
| medium | time_weighted_avg_queue | DQN | Webster | 5 | paired by seed | paired t-test | -3.771 | 0.020 | 0.039 | -0.990 | -1.719 | -0.261 |  |
| medium | time_weighted_avg_queue | Q-Learning | DQN | 5 | paired by seed | Wilcoxon signed-rank | 0.000 | 0.062 | 0.062 | 0.300 | -0.197 | 0.797 |  |
| medium | throughput_veh_per_min | Q-Learning | Webster | 5 | paired by seed | paired t-test | 4.332 | 0.012 | 0.037 | 0.140 | 0.050 | 0.230 |  |
| medium | throughput_veh_per_min | DQN | Webster | 5 | paired by seed | paired t-test | 2.349 | 0.079 | 0.157 | 0.133 | -0.024 | 0.291 |  |
| medium | throughput_veh_per_min | Q-Learning | DQN | 5 | paired by seed | paired t-test | 0.167 | 0.876 | 0.876 | 0.007 | -0.104 | 0.118 |  |
| medium | final_queue | Q-Learning | Webster | 5 | paired by seed | paired t-test | -4.332 | 0.012 | 0.037 | -4.200 | -6.892 | -1.508 |  |
| medium | final_queue | DQN | Webster | 5 | paired by seed | paired t-test | -2.349 | 0.079 | 0.157 | -4.000 | -8.728 | 0.728 |  |
| medium | final_queue | Q-Learning | DQN | 5 | paired by seed | paired t-test | -0.167 | 0.876 | 0.876 | -0.200 | -3.532 | 3.132 |  |

## 9. Controller rankings by scenario and metric
Rank 1 is best for the listed metric direction; ties receive average ranks. No single overall winner is calculated.

| scenario | metric | direction | rank | controller | mean | std | valid_observations |
|---|---|---|---|---|---|---|---|
| high | avg_waiting_time_sec | lower | 1.000 | DQN | 7.908 | 1.550 | 5 |
| high | avg_waiting_time_sec | lower | 2.000 | Q-Learning | 19.266 | 10.587 | 5 |
| high | avg_waiting_time_sec | lower | 3.000 | Webster | 306.435 | 16.086 | 5 |
| high | time_weighted_avg_queue | lower | 2.000 | DQN | 302.537 | 12.589 | 5 |
| high | time_weighted_avg_queue | lower | 3.000 | Q-Learning | 311.177 | 27.640 | 5 |
| high | time_weighted_avg_queue | lower | 1.000 | Webster | 217.834 | 14.683 | 5 |
| high | final_queue | lower | 2.000 | DQN | 601.800 | 16.142 | 5 |
| high | final_queue | lower | 3.000 | Q-Learning | 653.200 | 42.499 | 5 |
| high | final_queue | lower | 1.000 | Webster | 420.400 | 24.937 | 5 |
| high | throughput_veh_per_min | higher | 2.000 | DQN | 23.700 | 0.990 | 5 |
| high | throughput_veh_per_min | higher | 3.000 | Q-Learning | 21.987 | 0.998 | 5 |
| high | throughput_veh_per_min | higher | 1.000 | Webster | 29.747 | 0.105 | 5 |
| high | cumulative_reward | higher | 2.000 | DQN | -5415.570 | undefined |  |
| high | cumulative_reward | higher | 3.000 | Q-Learning | -5568.520 | undefined |  |
| high | cumulative_reward | higher | 1.000 | Webster | -3900.000 | undefined |  |
| low | avg_waiting_time_sec | lower | 2.500 | DQN | 10.540 | 0.167 | 5 |
| low | avg_waiting_time_sec | lower | 2.500 | Q-Learning | 10.540 | 0.167 | 5 |
| low | avg_waiting_time_sec | lower | 1.000 | Webster | 10.190 | 0.355 | 5 |
| low | time_weighted_avg_queue | lower | 2.500 | DQN | 0.986 | 0.053 | 5 |
| low | time_weighted_avg_queue | lower | 2.500 | Q-Learning | 0.986 | 0.053 | 5 |
| low | time_weighted_avg_queue | lower | 1.000 | Webster | 0.926 | 0.082 | 5 |
| low | final_queue | lower | 1.500 | DQN | 0.600 | 0.800 | 5 |
| low | final_queue | lower | 1.500 | Q-Learning | 0.600 | 0.800 | 5 |
| low | final_queue | lower | 3.000 | Webster | 1.000 | 0.632 | 5 |
| low | throughput_veh_per_min | higher | 1.500 | DQN | 10.887 | 0.490 | 5 |
| low | throughput_veh_per_min | higher | 1.500 | Q-Learning | 10.887 | 0.490 | 5 |
| low | throughput_veh_per_min | higher | 3.000 | Webster | 10.873 | 0.493 | 5 |
| low | cumulative_reward | higher | 2.500 | DQN | -17.710 | undefined |  |
| low | cumulative_reward | higher | 2.500 | Q-Learning | -17.710 | undefined |  |
| low | cumulative_reward | higher | 1.000 | Webster | -16.610 | undefined |  |
| medium | avg_waiting_time_sec | lower | 1.000 | DQN | 16.501 | 0.763 | 5 |
| medium | avg_waiting_time_sec | lower | 2.000 | Q-Learning | 17.264 | 0.682 | 5 |
| medium | avg_waiting_time_sec | lower | 3.000 | Webster | 18.953 | 1.101 | 5 |
| medium | time_weighted_avg_queue | lower | 1.000 | DQN | 4.498 | 0.321 | 5 |
| medium | time_weighted_avg_queue | lower | 2.000 | Q-Learning | 4.798 | 0.379 | 5 |
| medium | time_weighted_avg_queue | lower | 3.000 | Webster | 5.488 | 0.573 | 5 |
| medium | final_queue | lower | 2.000 | DQN | 4.400 | 2.059 | 5 |
| medium | final_queue | lower | 1.000 | Q-Learning | 4.200 | 3.655 | 5 |
| medium | final_queue | lower | 3.000 | Webster | 8.400 | 4.409 | 5 |
| medium | throughput_veh_per_min | higher | 2.000 | DQN | 23.280 | 0.668 | 5 |
| medium | throughput_veh_per_min | higher | 1.000 | Q-Learning | 23.287 | 0.611 | 5 |
| medium | throughput_veh_per_min | higher | 3.000 | Webster | 23.147 | 0.559 | 5 |
| medium | cumulative_reward | higher | 1.000 | DQN | -80.740 | undefined |  |
| medium | cumulative_reward | higher | 2.000 | Q-Learning | -86.150 | undefined |  |
| medium | cumulative_reward | higher | 3.000 | Webster | -98.360 | undefined |  |

## 10. Scenario observations
- Low demand: Webster had the lowest observed mean departed-vehicle waiting time (10.190 seconds); consider this with queue and service metrics.
  Webster had the lowest observed time-weighted average queue (0.926 vehicles). These descriptive extrema are not claims of statistical superiority.
- Medium demand: DQN had the lowest observed mean departed-vehicle waiting time (16.501 seconds); consider this with queue and service metrics.
  DQN had the lowest observed time-weighted average queue (4.498 vehicles). These descriptive extrema are not claims of statistical superiority.
- High demand: DQN had the lowest observed mean departed-vehicle waiting time (7.908 seconds); consider this with queue and service metrics.
  Webster had the lowest observed time-weighted average queue (217.834 vehicles). These descriptive extrema are not claims of statistical superiority.

## 11. Training convergence
- DQN/high: first 100-episode mean reward -4160.915; last 100-episode mean reward -4160.915; observed change +0.000. This trend alone does not establish stable convergence.
- DQN/low: first 100-episode mean reward -31.285; last 100-episode mean reward -31.285; observed change +0.000. This trend alone does not establish stable convergence.
- DQN/medium: first 100-episode mean reward -485.710; last 100-episode mean reward -485.710; observed change +0.000. This trend alone does not establish stable convergence.
- Q-learning/high: first 100-episode mean reward -4185.019; last 100-episode mean reward -4185.019; observed change +0.000. This trend alone does not establish stable convergence.
- Q-learning/low: first 100-episode mean reward -30.542; last 100-episode mean reward -30.542; observed change +0.000. This trend alone does not establish stable convergence.
- Q-learning/medium: first 100-episode mean reward -344.350; last 100-episode mean reward -344.350; observed change +0.000. This trend alone does not establish stable convergence.

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
