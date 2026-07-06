# Public Demand/Reorder Proxy Report - 2026-06-14

## Classification

`PUBLIC_DEMAND_REORDER_PROXY_PARTIAL_TESCO_READY`

## Evidence

JSON report:

- `reports/benchmarks/public_data_expansion_20260614/m5_tesco_demand_inventory/demand_reorder_public_proxy_report.json`

Local data:

- `data/public/tesco_grocery_20260614/year_borough_grocery.csv`

## Tesco Summary

- Rows: 33
- Columns: 202
- `num_transactions` rows: 33
- Mean `num_transactions`: 18,279,139.48
- Min `num_transactions`: 842,113
- Max `num_transactions`: 58,025,525
- Mean population: 262,634.24

## Blocked Demand Sources

- Instacart: legacy public direct links returned 404.
- M5: Zenodo direct file HEAD returned 403.

## What It Adds

Tesco adds an accessible, anonymized, area-level grocery demand proxy. It can support a broad discussion of demand-density variation and why inventory/reorder validation needs real demand fields.

## Limits

This does not validate SKU-level reorder policy, stockout cost, holding cost, supplier lead time, or action-level no-reorder optimality. Company data remains required for inventory/reorder conclusions.

