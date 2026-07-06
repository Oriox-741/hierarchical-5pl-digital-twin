# Fleet Dispatch Scorecard Update - 2026-06-14

## Classification

`FLEET_DISPATCH_SCORECARD_UPDATED_WITH_BLOCKERS`

## Recommended Score Changes

| Dimension | Previous | Recommended | Reason |
|---|---:|---:|---|
| Fleet/dispatch external evidence | 3.5 / 5 | 4.1 / 5 | HVFHV 200k public ride-hail sample, FleetPy no-Gurobi RPP/IRS scenarios, SF trajectory proxy |
| Route/stochastic public evidence | 4.2 / 5 | 4.3 / 5 | SVRPBench dynamic `appear_times` summarized; stochastic raw fields still missing |
| SOTA pathway maturity | 4.2 / 5 | 4.3 / 5 | Public benchmark package is more reproducible and multi-source |
| Thesis/advisor defensibility | 4.4 / 5 | 4.5 / 5 | Fleet/dispatch weak point now has stronger public evidence and honest blockers |

## Why Fleet/Dispatch Can Exceed 4.0

The evidence now spans:

- Official large-scale HVFHV request-to-service timing.
- Public FleetPy simulation with no-Gurobi ride-pooling and ride-parcel-pooling insertion.
- Public trajectory movement coverage.

## Why It Should Not Be Higher Yet

Chicago TNP/Taxi were blocked, OpenMines was blocked, and no public source contains the company-specific 5PL fields needed for secondary fleet economics, dispatch failure causality, or action 24/32 cost optimality.

## No-Overclaim Rules

Do not claim global SOTA, real-world deployment readiness, company-data validation, or action 24/32 economic optimality from this package.

