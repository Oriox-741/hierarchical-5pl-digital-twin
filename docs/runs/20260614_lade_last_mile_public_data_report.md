# LaDe Last-Mile Public Data Report - 2026-06-14

## Classification

`LADE_LAST_MILE_BOUNDED_READY_WITH_LICENSE_CAVEAT`

## Evidence

JSON report:

- `reports/benchmarks/public_data_expansion_20260614/lade_last_mile/lade_preflight_report.json`

Local data:

- `data/public/lade_20260614/delivery_jl.parquet`
- Size: 2,271,457 bytes
- Rows: 31,415
- Columns: 17
- SHA256 recorded in the JSON report

## Summary

The bounded LaDe-D `delivery_jl` split contains 31,415 delivery rows for one city (`Jilin`) and 57 couriers. Every row includes delivery timing fields, courier id, region/city fields, AOI fields, GPS fields, and order id. Accept-to-delivery latency was computed for all 31,415 rows:

- Mean: 12,211.17 seconds
- Min: 0 seconds
- Max: 214,380 seconds

## What It Adds

LaDe strengthens public route/logistics evidence for delivery timing, courier assignment pressure, and route-side plausibility checks. It can help test whether route-choice proxies and delivery timing stress behave plausibly in public last-mile data.

## Caveats

The dataset should not be used for publication-grade claims until license ambiguity is resolved. It also cannot validate secondary-fleet economics, reorder decisions, stockout cost, holding cost, or company-specific action 24/32 optimality.

