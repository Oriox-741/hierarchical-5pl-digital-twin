# Gurobi-Free Fleet Dispatch Report - 2026-06-14

## Classification

`NYC_TLC_DISPATCH_PROXY_READY_WITH_OPENMINES_BLOCKED`

## Evidence

JSON reports:

- `reports/benchmarks/public_data_expansion_20260614/nyc_tlc_fleet_dispatch/nyc_tlc_dispatch_report.json`
- `reports/benchmarks/public_data_expansion_20260614/openmines_dispatch/openmines_dispatch_report.json`

## NYC TLC FHV Proxy

Local data:

- `data/public/nyc_tlc_20260614/fhv_tripdata_2023-01.parquet`
- Rows in file: 1,114,320
- Rows analyzed: 5,000
- Zones in analyzed subset: 249

Three zone-local immediate-dispatch proxy variants were computed:

| Variant | Vehicles per zone | Served proxy rate | Unserved proxy |
|---|---:|---:|---:|
| Low capacity | 1 | 0.4132 | 2,934 |
| Balanced capacity | 2 | 0.6048 | 1,976 |
| High capacity | 4 | 0.7796 | 1,102 |

## OpenMines Blocker

`pip install openmines` failed under Python 3.12 because OpenMines pins `numpy==1.25.0`, and that build path fails with `pkgutil.ImpImporter` removed.

## Interpretation

NYC TLC adds a bounded, public, Gurobi-free demand replay proxy for dispatch pressure. It does not model parcel delivery, 5PL carrier acceptance, inventory, or action 24/32 economics. OpenMines remains a possible future dispatch analog only in an isolated compatible Python environment.

