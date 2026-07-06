# Inventory Reorder Benchmark Report

Date: 2026-06-13

## Decision

`INVENTORY_REORDER_BENCHMARK_READY`

## Scope

The original `or-gym` install path was attempted but blocked by the legacy `gym<=0.19.0` metadata build under Python 3.12. The sprint then used the maintained `gym-invmgmt` package, which provides Gymnasium-compatible multi-echelon inventory environments.

Benchmark environment:

- Env id: `GymInvMgmt/Serial-v0`
- Episodes: `20`
- Max steps: `30`
- Seed: `42`

## Outputs

- JSON: `reports/benchmarks/sota_pathway_20260613/inventory_or_gym/inventory_benchmark_report.json`
- CSV: `reports/benchmarks/sota_pathway_20260613/inventory_or_gym/inventory_benchmark_summary.csv`

## Results

| Policy | Steps | Total cost | Mean inventory | Mean backlog | No-backlog step rate | Mean order quantity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_stock_order_up_to` | 600 | 346062.82 | 16922.96 | 0.00 | 1.00 | 3715.35 |
| `conservative_reorder` | 600 | 235989.81 | 9305.88 | 0.00 | 1.00 | 1920.00 |
| `aggressive_reorder` | 600 | 801909.81 | 31625.88 | 0.00 | 1.00 | 6240.00 |
| `project_heuristic_continuous_reorder` | 600 | 943597.51 | 38916.74 | 0.00 | 1.00 | 7219.43 |

## Interpretation

This benchmark validates that the project can execute a public inventory-control analog and report reorder policy tradeoffs. Conservative reorder was cheapest in this environment, while aggressive and project-inspired policies over-ordered and carried substantially more inventory.

This result should not be interpreted as a full 5PL result. It validates the inventory/reorder component only.

## Classification

`INVENTORY_REORDER_BENCHMARK_READY`
