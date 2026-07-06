# Route Score Upgrade Report

## Decision

`ROUTE_SCORE_UPGRADED_AFTER_VRPTW_CONVERTER`

## Evidence

Artifacts:

- `reports/benchmarks/scorecard_upgrade_20260613/route_extended/pyvrp_benchmark_report.json`
- `reports/benchmarks/scorecard_upgrade_20260613/route_extended/pyvrp_benchmark_summary.csv`
- `reports/benchmarks/scorecard_upgrade_20260613/route_extended/route_gap_summary.json`

Code/test patch:

- `scripts/pyvrp_route_reference_benchmark.py`
- `tests/benchmarks/test_pyvrp_route_reference_benchmark.py`

## What Changed

The prior PyVRP benchmark was blocked on Solomon/VRPTW TXT files because PyVRP rejected them as non-VRPLIB. The patch adds a small Solomon parser and builds PyVRP models directly from depot/customer coordinates, demand, service time, and time windows.

Focused tests prove:

- Solomon TXT vehicle/customer parsing.
- PyVRP model construction from parsed Solomon instances.
- Existing output overwrite protection remains intact.

## Fresh Results

The extended run produced `11` successful instances and `0` blocked rows:

- 5 CVRPLIB CVRP instances.
- 6 Solomon-style VRPTW instances.

All rows were feasible. VRPTW instances `C1_2_1.TXT` through `C1_2_6.TXT` now solve successfully under the bounded two-second PyVRP run.

## Gap Caveat

The local CVRP files expose header comments used for a preliminary known-best comparison, but several computed gaps are negative. That means the local header values and PyVRP objective convention are not safe enough for a performance claim. No BKS or global SOTA claim is made from these gaps.

## Stochastic/Dynamic Routing Source Scan

SVRPBench remains the preferred stochastic route target because its public description covers congestion, delays, accidents, and stochastic urban routing instances. Public locations found during this sprint:

- `https://github.com/yehias21/vrp-benchmarks`
- `https://huggingface.co/datasets/Yahias21/vrp_benchmark`
- `https://openreview.net/forum?id=yADrJyaBJl`

A smaller dynamic-VRP-with-stochastic-requests fallback exists at:

- `https://github.com/amflorio/dvrp-stochastic-requests`

No stochastic/dynamic integration was added in this sprint because the goal was targeted scorecard evidence, not a new benchmark family with unclear adapter cost.

## Score Result

| Area | Prior | Updated | Reason |
| --- | ---: | ---: | --- |
| Route benchmark score | 3.5 / 5 | 4.1 / 5 | VRPTW blocker removed and route-only public benchmark coverage expanded to 11 feasible PyVRP instances. |

## Remaining Limit

The route score is not 5/5 because stochastic/dynamic routing is still not integrated and because local BKS comparison is not yet publication-grade.

