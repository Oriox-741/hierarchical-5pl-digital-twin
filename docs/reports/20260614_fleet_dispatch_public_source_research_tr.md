# Fleet/Dispatch Public Source Research - 2026-06-14

## Classification

`FLEET_DISPATCH_PUBLIC_SOURCE_CATALOG_READY`

Machine-readable catalog:

- `reports/benchmarks/fleet_dispatch_upgrade_20260614/final_synthesis/fleet_dispatch_public_source_catalog.json`

## Ranked Sources

| Priority | Source | Status | Why It Matters | Limits |
|---|---|---|---|---|
| P0 | NYC High Volume FHV | Completed | Official, recent ride-hail/FHV data with request, on-scene, pickup, base, zone, trip, fare, and driver-pay fields | No declined-request stream or actual assignment alternatives |
| P0 | Chicago TNP Trips | Blocked by Socrata 503 | Strong second-city ride-hail OD/fare/demand proxy | Rounded/suppressed fields; API unavailable in this run |
| P0 | FleetPy no-Gurobi | Completed | Public simulator evidence with immediate insertion and ride-parcel-pooling insertion scenarios | Advanced optimization/repositioning still requires `gurobipy` |
| P1 | Chicago Taxi Trips | Blocked by Socrata 503 | Vehicle-level-ish taxi utilization/reposition comparator | Taxi, not ride-hail; API unavailable in this run |
| P1 | San Francisco taxi trajectories | Completed | Public vehicle movement and spatial coverage proxy | Old 2008 taxi trajectories; no request/acceptance stream |
| P1 | Porto taxi trajectories | Cataloged | Large trajectory dataset useful for route/movement behavior | 508.9 MB zip; not needed after HVFHV/SF bounded evidence in this sprint |
| P1 | SVRPBench dynamic fields | Completed partial | Dynamic routing `appear_times` evidence over 560 instances | No persisted stochastic congestion/delay/accident fields |
| P2 | OpenMines | Blocked | Gurobi-free dispatch analog in mining domain | No compatible Python 3.10/3.11 interpreter available here |
| Future | DiDi GAIA/KDD dispatch | Cataloged | Highly dispatch-specific if access/legal terms allow | Requires registration/agreement; not reproducible in this sprint |

## Score Interpretation

Fleet/dispatch evidence can move above `3.5` because the sprint added:

- A 200,000-row deterministic stride sample from an 18.48M-row NYC HVFHV month.
- A fresh no-Gurobi FleetPy ride-parcel-pooling run with two scenarios.
- A 200,000-row SF taxi trajectory movement proxy over 534 taxis.

The score should still stay below full maturity because Chicago TNP/Taxi were blocked, OpenMines was blocked, and none of the public sources validate company-specific carrier acceptance, secondary-fleet economics, or 5PL dispatch failure reasons.

