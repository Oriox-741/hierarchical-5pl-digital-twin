# Fleet Dispatch Benchmark Report

Date: 2026-06-13

## Decision

`FLEET_DISPATCH_BENCHMARK_READY`

## Scope

Downloaded public FleetPy source:

- `data/public/fleetpy_20260613/source/FleetPy-main`

Attempted scenario:

- Constant config: `studies/example_study/scenarios/constant_config_ir.csv`
- Scenario config: `studies/example_study/scenarios/example_ir_only.csv`
- Result scenario: `example_pool_irsonly_sc_1`

Initial run failed on missing `pyproj`. After installing declared geospatial runtime dependencies (`pyproj`, `shapely`, `rtree`, `dill`, `tqdm`), the example scenario completed and produced standard FleetPy output files.

## Outputs

- JSON: `reports/benchmarks/sota_pathway_20260613/fleet_dispatch/fleet_dispatch_benchmark_report.json`
- CSV: `reports/benchmarks/sota_pathway_20260613/fleet_dispatch/fleet_dispatch_summary.csv`
- Source result dir: `data/public/fleetpy_20260613/source/FleetPy-main/studies/example_study/results/example_pool_irsonly_sc_1`

## Key Metrics

| Metric | Value |
| --- | ---: |
| Number users | 93 |
| Modal split | 0.93 |
| Served online users [%] | 93.00 |
| Created offers [%] | 93.00 |
| Average waiting time | 151.996 |
| 90% waiting time quantile | 274.545 |
| Fleet utilization [%] | 74.579 |
| Total vkm | 227.063 |
| Empty vkm [%] | 34.433 |
| Occupancy | 0.724 |
| Shared rides [%] | 32.258 |

## Interpretation

FleetPy gives the sprint a real public fleet/on-demand dispatch analog. The run is not a full 5PL benchmark and does not include inventory/reorder behavior, but it validates that a public fleet-dispatch simulator can be executed and summarized in this workspace.

## Classification

`FLEET_DISPATCH_BENCHMARK_READY`
