# Fleet/Dispatch Score Upgrade Report

## Decision

`FLEET_DISPATCH_SCORE_UNCHANGED_BLOCKED_BY_GUROBI`

## Evidence

Artifacts:

- Existing successful run: `reports/benchmarks/sota_pathway_20260613/fleet_dispatch/fleet_dispatch_benchmark_report.json`
- Extension blocker report: `reports/benchmarks/scorecard_upgrade_20260613/fleet_dispatch_extended/fleet_dispatch_extended_report.json`

The original FleetPy public immediate-offer scenario remains valid:

| Metric | Value |
| --- | ---: |
| number users | 93 |
| served online users | 93.0% |
| average waiting time | 151.996 |
| fleet utilization | 74.579% |
| empty vkm | 34.433% |
| shared rides | 32.258% |

## Extension Attempts

Additional built-in FleetPy scenarios were attempted for batch assignment, heuristic pooling, and repositioning. All richer scenarios initialized and then failed with:

`ModuleNotFoundError: No module named 'gurobipy'`

This affects:

- `example_pool_irsbatch_sc_1`
- `example_pool_rvheuristics_sc_2`
- `example_pool_repo_Pavone_sc_1`
- `example_pool_repo_AM_sc_1`

## Score Result

| Area | Prior | Updated | Reason |
| --- | ---: | ---: | --- |
| Fleet/dispatch external evidence | 3.0 / 5 | 3.0 / 5 | No additional successful public dispatch scenario was produced. |

## Remaining Limit

The blocker is not a model issue; it is an external optimizer/license dependency. A score upgrade needs either a configured licensed Gurobi path, an open optimizer substitution validated against FleetPy, or a different public fleet-dispatch benchmark.

