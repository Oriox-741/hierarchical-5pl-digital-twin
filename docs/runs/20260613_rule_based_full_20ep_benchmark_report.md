# Rule-Based Full 20-Episode Benchmark Report

Final classification: `RULE_BASED_FULL_BENCHMARK_READY`

Coverage: `8` baselines x `8` scenarios x `20` episodes = `1280` episode rows.

Outputs:
- `reports/benchmarks/full_completion_20260613/rule_based_full_20ep/rule_based_full_report.json`
- `reports/benchmarks/full_completion_20260613/rule_based_full_20ep/rule_based_full_summary.csv`
- `reports/benchmarks/full_completion_20260613/rule_based_full_20ep/episode_metrics.jsonl`

Important caveat: rule baselines use a neutral PPO continuous vector, so this is a tactical baseline comparison, not full PPO+DQN equivalence.

## Baseline Rollups

| Baseline | Mean service | Worst service | Mean dispatch success | Hard blockers |
| --- | ---: | ---: | ---: | ---: |
| `conservative_stock_threshold_reorder` | 0.9879 | 0.9463 | 0.9996 | 0 |
| `earliest_due_shortest_primary_none` | 0.9879 | 0.9463 | 0.9996 | 0 |
| `emergency_stockout_prevention_reorder` | 0.9879 | 0.9463 | 0.9996 | 0 |
| `fifo_shortest_primary_none` | 0.9879 | 0.9463 | 0.9996 | 0 |
| `high_holding_no_overstock` | 0.9879 | 0.9463 | 0.9996 | 0 |
| `low_congestion_under_disruption` | 0.9879 | 0.9463 | 0.9996 | 0 |
| `premium_first_shortest_primary_secondary_none` | 0.9879 | 0.9463 | 0.9996 | 0 |
| `vehicle_scarcity_primary_first_secondary_fallback` | 0.9879 | 0.9463 | 0.9996 | 0 |

Observation: all eight deterministic rule policies produced identical aggregate service in this adapter run, which suggests the current neutral-continuous rule adapter is a coarse tactical comparator rather than a rich family of distinct operational policies. The full 20-episode target is nevertheless complete.
