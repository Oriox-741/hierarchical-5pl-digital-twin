# Multi-Seed Robustness Benchmark Report

Final classification: `MULTISEED_ROBUSTNESS_READY`

Coverage: `800` episode rows across seeds `[42, 43, 44, 45, 46]` and `8` scenarios.
Seed-42 consistency max absolute service delta vs prior equal-budget eval: `0.0`.

Outputs:
- `reports/benchmarks/full_completion_20260613/multiseed_robustness/multiseed_summary.json`
- `reports/benchmarks/full_completion_20260613/multiseed_robustness/multiseed_summary.csv`

## Scenario Statistics

| Scenario | Service mean | Service sd | Service min | Service max | Hard blockers | Action24 total | Action32 total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_normal` | 0.9897 | 0.0082 | 0.9735 | 1.0000 | 0 | 2690 | 212 |
| `demand_spike_volatility` | 0.8590 | 0.0097 | 0.8447 | 0.8816 | 0 | 0 | 39 |
| `high_holding_cost` | 0.9668 | 0.0137 | 0.9379 | 0.9940 | 0 | 8391 | 2307 |
| `lead_time_volatility` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0 | 0 | 422 |
| `mixed_stress` | 0.9423 | 0.0101 | 0.9254 | 0.9660 | 0 | 16704 | 6912 |
| `premium_sla_pressure` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| `route_disruption_congestion` | 0.9003 | 0.0162 | 0.8760 | 0.9423 | 0 | 117 | 13806 |
| `vehicle_scarcity_capacity_shock` | 0.9467 | 0.0152 | 0.9207 | 0.9791 | 0 | 781 | 15 |

Decision: no seed-level hard-blocker regression was found. Route action 32 and mixed action 24 remain monitoring watches, not blockers.
