# Planned-Vs-Actual Route Dataset Report - 2026-06-14

## Classification

`PLANNED_ACTUAL_ROUTE_DATA_BLOCKED_API_403`

## Evidence

JSON report:

- `reports/benchmarks/public_data_expansion_20260614/planned_vs_actual_routes/planned_actual_route_report.json`

Public page:

- `https://data.mendeley.com/datasets/kkwgfvmtxn/1`

## Blocker

The public page was reachable, but automated API access returned HTTP 403 for the tested endpoints:

- `https://api.mendeley.com/datasets/kkwgfvmtxn/versions/1`
- `https://api.mendeley.com/datasets/kkwgfvmtxn`
- `https://data.mendeley.com/public-api/datasets/kkwgfvmtxn/versions/1`
- `https://data.mendeley.com/public-api/datasets/kkwgfvmtxn`

No route-pair files were downloaded.

## Intended Metrics If Access Is Resolved

- Planned vs actual stop-sequence deviation
- Driver/route concentration
- Stress-bucket route deviation
- Whether real routes converge on shortest-like or reliable-route behavior under constrained conditions

## Decision

This source remains useful but blocked. It is not counted as completed public-data evidence in the scorecard.

