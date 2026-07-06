# Public Data Scorecard Update - 2026-06-14

## Classification

`PUBLIC_DATA_SCORECARD_UPDATED_WITH_PARTIAL_BLOCKERS`

## Conservative Score Changes

| Dimension | Previous maturity | Updated maturity | Reason |
|---|---:|---:|---|
| Fleet/dispatch external evidence | 3.0 | 3.5 | NYC TLC public dispatch-pressure proxy added; OpenMines blocked |
| Route/stochastic public evidence | 4.1 | 4.2 | SVRPBench base geometry integrated; stochastic fields missing |
| Inventory/reorder external evidence | 4.4 | 4.5 | Tesco public demand proxy added; Instacart/M5 blocked |
| Company-data validation readiness | 4.0 | 4.2 | Synthetic replay harness now checks action 24/32 validation questions |
| SOTA pathway maturity | 3.9 | 4.2 | Public-data catalog plus ablation package improve evidence structure |
| Thesis/advisor defensibility | 4.2 | 4.4 | Public sources and blockers are now documented reproducibly |

## Why The Scores Stay Conservative

The public datasets improve evidence breadth, but they do not validate company-specific fleet acceptance, secondary-fleet cost, reorder economics, stockout penalties, or exact action-level optimality. The correct next step is company-data replay validation under separate approval, not larger blind training.

