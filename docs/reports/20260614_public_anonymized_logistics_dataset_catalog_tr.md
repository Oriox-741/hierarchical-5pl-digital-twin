# Public Anonymized Logistics Dataset Catalog - 2026-06-14

## Classification

`PUBLIC_DATASET_CATALOG_READY_WITH_SOURCE_BLOCKERS`

## Scope

This catalog supports the public-data expansion sprint. It documents which public or anonymized datasets can strengthen the benchmark evidence without touching production, registry, baselines, DB, checkpoints, or existing eval outputs.

Machine-readable catalog:

- `reports/benchmarks/public_data_expansion_20260614/public_dataset_catalog/public_dataset_catalog.json`

## Ready Or Partially Ready Sources

| Source | Local status | Validation value | Limits |
|---|---:|---|---|
| LaDe-D delivery sample | Downloaded, 2.27 MB | Last-mile delivery timing, courier/zone workload, route-side public proxy | License ambiguity must be resolved before publication reuse; no company fleet/reorder economics |
| Olist ecommerce logistics | Downloaded, 41.39 MB across selected CSVs | Lateness, freight value, order status mix | Ecommerce order data; no route-choice or action-level causality |
| Tesco Grocery 1.0 area demand | Downloaded, 85 KB | Public demand-density proxy | Area-level only; not SKU reorder/stockout economics |
| NYC TLC FHV Jan 2023 | Downloaded, 10.8 MB | Gurobi-free dispatch pressure proxy over public trips | Passenger trip proxy, not parcel/fleet economics |
| SVRPBench MBZUAI parquet | Downloaded, 495 KB | VRP geometry, demands, capacities, dynamic appear times | Released parquet lacks persisted stochastic congestion/delay/accident fields |

## Blocked Sources

| Source | Blocker | Safe next step |
|---|---|---|
| Mendeley planned-vs-actual route deviations | Public page reachable, API endpoints returned HTTP 403 | Manual browser/token access or alternate mirror under explicit approval |
| OpenMines | `pip install openmines` failed on Python 3.12 due `numpy==1.25.0` build metadata path | Isolated Python 3.10/3.11 environment if still needed |
| Instacart market basket | Legacy public direct links returned 404 | Use Kaggle/API or alternate legal mirror under separate approval |
| M5 demand | Zenodo direct file HEAD returned 403 | Manual access or alternate approved mirror |

## Publication Use

This catalog can support an evidence-maturity claim: the project now has bounded public-data probes for route/logistics, ecommerce lateness, demand, fleet-dispatch pressure, and VRP geometry. It does not support a global SOTA claim or real-world optimality claim for action 24/32.

