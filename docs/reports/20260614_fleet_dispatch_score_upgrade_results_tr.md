# Fleet Dispatch Score Upgrade Results - 2026-06-14

## Classification

`FLEET_DISPATCH_UPGRADE_PARTIALLY_READY_WITH_BLOCKERS`

## Completed Evidence

- NYC HVFHV: 200,000-row deterministic stride sample from 18.48M January 2023 rows.
- FleetPy: three no-Gurobi public simulation scenarios, including two fresh ride-parcel-pooling insertion runs.
- San Francisco taxi trajectories: 200,000-row movement sample over 534 taxis.
- SVRPBench dynamic routing: 560 instances with `appear_times`.

## Blocked Evidence

- Chicago TNP: Socrata HTTP 503.
- Chicago Taxi: Socrata HTTP 503.
- OpenMines: no compatible Python 3.10/3.11 isolation available.
- Yahias21 Hugging Face `.npz`: HTTP 401.
- SVRPBench stochastic raw fields: not present in released parquet.

## Interpretation

Fleet/dispatch evidence is substantially stronger than the prior 5,000-row FHV proxy plus a Gurobi-blocked FleetPy extension. The new package has one large official ride-hail/FHV source, one public simulator path with no-Gurobi scenarios, and one trajectory movement proxy.

This still does not validate company-specific dispatch failures, secondary-fleet acceptance/cost, carrier contracts, or action 24/32 economic optimality.

