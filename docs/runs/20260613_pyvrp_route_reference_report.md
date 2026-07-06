# PyVRP Route Reference Benchmark Report

Date: 2026-06-13

## Decision

`PYVRP_ROUTE_REFERENCE_READY`

## Scope

Installed and used PyVRP as a stronger public route-solver reference.

Data root:

- `data/public/route_benchmarks_20260613`

## Outputs

- JSON: `reports/benchmarks/sota_pathway_20260613/pyvrp_route_reference/pyvrp_benchmark_report.json`
- CSV: `reports/benchmarks/sota_pathway_20260613/pyvrp_route_reference/pyvrp_benchmark_summary.csv`

## Results

| Instance | Family | Status | Objective | Runtime seconds | Routes |
| --- | --- | --- | ---: | ---: | ---: |
| `A-n32-k5.vrp` | CVRPLIB CVRP | success | 773.0 | 2.014 | 5 |
| `A-n33-k5.vrp` | CVRPLIB CVRP | success | 650.0 | 2.015 | 5 |
| `A-n33-k6.vrp` | CVRPLIB CVRP | success | 726.0 | 2.017 | 6 |
| `A-n34-k5.vrp` | CVRPLIB CVRP | success | 767.0 | 2.015 | 5 |
| `A-n36-k5.vrp` | CVRPLIB CVRP | success | 789.0 | 2.011 | 5 |
| `C1_2_1.TXT` | VRPTW reference | blocked |  | 0.009 |  |
| `C1_2_2.TXT` | VRPTW reference | blocked |  | 0.007 |  |
| `C1_2_3.TXT` | VRPTW reference | blocked |  | 0.009 |  |

## Blocked VRPTW Reason

The extracted `C1_2_*.TXT` files were rejected by PyVRP with:

`RuntimeError: Instance does not conform to the VRPLIB format.`

## Interpretation

PyVRP successfully provides a route-only reference on CVRPLIB instances. The VRPTW subset needs a compatible converter or source format before it can be counted as a PyVRP VRPTW benchmark.

PyVRP validates route-solving capability only. It does not validate full 5PL dispatch, fleet selection, or reorder policy quality.

## Classification

`PYVRP_ROUTE_REFERENCE_READY`
