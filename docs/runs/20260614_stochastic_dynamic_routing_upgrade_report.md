# Stochastic/Dynamic Routing Upgrade Report - 2026-06-14

## Classification

`DYNAMIC_ROUTING_APPEAR_TIMES_READY_STOCHASTIC_RAW_FIELDS_BLOCKED`

JSON report:

- `reports/benchmarks/fleet_dispatch_upgrade_20260614/stochastic_dynamic_routing/stochastic_dynamic_routing_report.json`

## Completed Evidence

SVRPBench local parquet:

- Rows: 560
- Dynamic fields detected: `appear_times`
- Customer count mean: 271.29
- Customer count min/max: 11 / 1,020

amflorio DVRP stochastic requests repository:

- GitHub repository reachable: `https://github.com/amflorio/dvrp-stochastic-requests`
- Results CSV downloaded: 27,964 rows
- Role: supporting public algorithm/results metadata, not raw instance integration

## Blockers

- Yahias21 Hugging Face `.npz` direct file returned HTTP 401.
- SVRPBench repo is reachable, but repo-side stochastic generation is a heavier separate task.
- Released SVRPBench parquet lacks persisted congestion, delay, accident, travel-time matrix, and time-window fields.

## Score Impact

Route/stochastic evidence can improve modestly because dynamic arrival/appearance fields are now explicitly summarized. It should not jump aggressively because raw stochastic travel-time/disruption fields remain unavailable.

