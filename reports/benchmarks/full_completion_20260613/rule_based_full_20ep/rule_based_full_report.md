# Full Rule-Based 20-Episode Benchmark

- Decision: RULE_BASED_FULL_BENCHMARK_READY
- Scenario dir: configs\eval_scenarios
- Episodes per scenario: 20
- Episode rows: 1280

## Baseline Rollups

| Baseline | Scenarios | Mean service | Worst service | Mean lateness | Mean dispatch success | Hard blockers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conservative_stock_threshold_reorder | 8 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| earliest_due_shortest_primary_none | 8 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| emergency_stockout_prevention_reorder | 8 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| fifo_shortest_primary_none | 8 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| high_holding_no_overstock | 8 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| low_congestion_under_disruption | 8 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| premium_first_shortest_primary_secondary_none | 8 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
| vehicle_scarcity_primary_first_secondary_fallback | 8 | 0.9879 | 0.9463 |  | 0.9996 | 0 |
