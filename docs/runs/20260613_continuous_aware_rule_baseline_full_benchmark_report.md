# Continuous-Aware Rule Baseline Full Benchmark Report

Date: 2026-06-13

## Decision

`CONTINUOUS_AWARE_RULE_BASELINE_BENCHMARK_READY`

## Scope

Full benchmark over:

- 8 rule baselines
- 8 evaluation scenarios
- 20 episodes per scenario
- 4 continuous modes
- seed `42`

Expected and produced row count:

- `8 x 8 x 20 x 4 = 5120` episode rows

## Outputs

- JSON: `reports/benchmarks/sota_pathway_20260613/continuous_aware_rule_baselines/continuous_aware_rule_baseline_report.json`
- CSV: `reports/benchmarks/sota_pathway_20260613/continuous_aware_rule_baselines/continuous_aware_rule_baseline_summary.csv`
- Episodes: `reports/benchmarks/sota_pathway_20260613/continuous_aware_rule_baselines/episode_metrics.jsonl`
- Progress: `reports/benchmarks/sota_pathway_20260613/continuous_aware_rule_baselines/progress_state.json`

## Mode Summary

| Mode | Scenario-baseline rows | Mean service | Worst service | Mean dispatch success |
| --- | ---: | ---: | ---: | ---: |
| `neutral_continuous` | 64 | 0.987866 | 0.946345 | 0.999572 |
| `heuristic_continuous` | 64 | 0.987866 | 0.946345 | 0.999572 |
| `ppo_assisted_continuous` | 64 | 0.987595 | 0.945289 | 0.999632 |
| `oracle_diagnostic_continuous` | 64 | 0.987866 | 0.946345 | 0.999572 |

## Interpretation

The full benchmark completed at the requested budget. The transparent heuristic continuous mode did not materially separate the tactical rule baselines from neutral continuous control in this scenario set. PPO-assisted continuous control stayed close to neutral/heuristic service while slightly increasing dispatch success, but it is a diagnostic hybrid because learned production PPO controls are paired with rule tactical actions.

The oracle diagnostic continuous mode is non-deployable and must not be used as an advisor-facing fair baseline.

## Fairness Caveats

- `neutral_continuous` preserves historical normalized-zero behavior. In this project, normalized zero projects to midpoint physical controls, not physical no-op.
- `heuristic_continuous` is a transparent current-state heuristic, not a learned PPO.
- `ppo_assisted_continuous` is a diagnostic hybrid, not a pure rule baseline.
- `oracle_diagnostic_continuous` is explicitly non-deployable.

## Classification

`CONTINUOUS_AWARE_RULE_BASELINE_BENCHMARK_READY`
