# SVRPBench Direct Integration Report - 2026-06-14

## Classification

`SVRPBENCH_BASE_GEOMETRY_READY_STOCHASTIC_FIELDS_MISSING`

## Evidence

JSON report:

- `reports/benchmarks/public_data_expansion_20260614/svrpbench_stochastic/svrpbench_integration_report.json`

Local data:

- `data/public/svrpbench_20260614/mbzuai_svrp_bench_test.parquet`
- Size: 495,447 bytes
- Rows: 560
- Columns: 8

Columns:

- `subset_name`
- `file_name`
- `instance_id`
- `locations`
- `demands`
- `num_vehicles`
- `vehicle_capacities`
- `appear_times`

## Summary

- Instances: 560
- Customer count mean: 271.29
- Customer count min: 11
- Customer count max: 1,020
- Stochastic fields detected in released parquet: none
- Time-window fields detected in released parquet: none

## Interpretation

The MBZUAI SVRPBench parquet is usable as public VRP geometry/demand/capacity/dynamic-appearance evidence. It is not a complete stochastic benchmark integration because the released parquet lacks persisted time windows, travel-time matrix, congestion, delay, and accident samples.

## Safe Next Step

If SVRPBench remains important, integrate repo-side stochastic instance generation in a separate goal and keep generated instances under `reports/benchmarks` or `data/public`, not protected eval/checkpoint paths.

