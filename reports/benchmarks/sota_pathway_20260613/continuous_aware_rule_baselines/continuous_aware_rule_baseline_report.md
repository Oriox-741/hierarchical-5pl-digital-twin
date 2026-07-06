# Continuous-Aware Rule Baseline Benchmark

- Decision: CONTINUOUS_AWARE_RULE_BASELINE_BENCHMARK_READY
- Scenario dir: configs\eval_scenarios
- Episodes per scenario: 20
- Episode rows: 5120

## Continuous Mode Rollups

| Mode | Rows | Mean service | Worst service | Mean lateness | Mean dispatch success | Hard blockers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic_continuous | 64 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| neutral_continuous | 64 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| oracle_diagnostic_continuous | 64 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| ppo_assisted_continuous | 64 | 0.9876 | 0.9453 |  | 0.9996 | 0 |

## Fairness Caveats

- neutral_continuous preserves historical fixed normalized-zero behavior; it is midpoint physical control, not physical no-op.
- heuristic_continuous is a transparent current-state heuristic, not a learned PPO.
- ppo_assisted_continuous pairs production learned continuous controls with rule discrete actions and is diagnostic, not a pure rule baseline.
- oracle_diagnostic_continuous is non-deployable and must not be used for advisor-facing fair-comparison claims.
