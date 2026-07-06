# Inventory/Reorder Score Upgrade Report

## Decision

`INVENTORY_REORDER_SCORE_UPGRADED_WITH_PUBLIC_BREADTH`

## Evidence

Artifacts:

- `reports/benchmarks/scorecard_upgrade_20260613/inventory_reorder_extended/inventory_reorder_extended_report.json`
- `reports/benchmarks/scorecard_upgrade_20260613/inventory_reorder_extended/inventory_reorder_extended_summary.csv`

Fresh evidence added:

- `gym-invmgmt` serial inventory benchmark across seeds `42..46`, `20` episodes per seed, `30` max steps.
- Five public MABIM/ReplenishmentEnv configs:
  - `sku50.single_store.standard`
  - `sku200.single_store.standard`
  - `sku200.single_store.higher_holding_cost`
  - `sku200.single_store.lower_capacity`
  - `sku50.2_stores.standard`

One MABIM config, `sku200.single_store.increase_demand`, was excluded because the source loader raised `NotImplementedError`.

## Key Results

In `gym-invmgmt`, conservative reorder remained the lowest-cost tested policy across five seeds, with very low cross-seed variation:

| Policy | Mean total cost | SD | CV |
| --- | ---: | ---: | ---: |
| conservative_reorder | 235874.25 | 78.51 | 0.00033 |
| base_stock_order_up_to | 346148.35 | 139.79 | 0.00040 |
| aggressive_reorder | 801794.25 | 78.51 | 0.00010 |
| project_heuristic_continuous_reorder | 943573.19 | 54.13 | 0.00006 |

In MABIM, no-reorder stress-floor policies produced negative final balances in all tested configs, while conservative/aggressive/project-inspired reorder policies remained positive.

## Score Result

| Area | Prior | Updated | Reason |
| --- | ---: | ---: | --- |
| Inventory/reorder external evidence | 4.0 / 5 | 4.4 / 5 | Multi-seed public inventory evidence and multiple MABIM stress regimes now exist. |

## Remaining Limit

This still validates inventory/reorder analogs only. Real company demand, cost, stockout, holding, emergency purchase, and reorder economics are required for 5/5.

