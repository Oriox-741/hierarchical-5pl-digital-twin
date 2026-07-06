# Upgraded Benchmark Scorecard

Date: 2026-06-14 local time. Sprint folder: `reports/benchmarks/scorecard_upgrade_20260613/`.

## Final Classification

`SCORECARD_UPGRADE_PARTIALLY_READY_WITH_BLOCKERS`

## Updated Scores

| Area | Previous | Updated | Change | Rationale |
| --- | ---: | ---: | ---: | --- |
| Rule-based baseline score | 4.0 / 5 | 4.0 / 5 | 0.0 | Diagnostics show continuous controls changed but aggregate outcomes did not separate from neutral. |
| Inventory/reorder external evidence | 4.0 / 5 | 4.4 / 5 | +0.4 | Added five-seed `gym-invmgmt` evidence and five MABIM public stress configs. |
| Fleet/dispatch external evidence | 3.0 / 5 | 3.0 / 5 | 0.0 | Richer FleetPy scenarios were blocked by missing `gurobipy`; only one public scenario remains successful. |
| Route benchmark score | 3.5 / 5 | 4.1 / 5 | +0.6 | Added Solomon/VRPTW conversion and produced 11 feasible PyVRP runs with no blocked rows. |
| SOTA pathway maturity | 3.5 / 5 | 3.9 / 5 | +0.4 | Added reproducibility guide, manifest, explicit blocker accounting, and stronger route/inventory evidence. |
| Thesis-defense readiness | 4.0 / 5 | 4.2 / 5 | +0.2 | Evidence package is clearer and more defensible, but company data and stochastic routing remain limits. |

## What Actually Improved

- Route: the prior VRPTW format blocker was fixed with tested Solomon parsing and PyVRP model construction.
- Inventory: evidence now spans multiple seeds plus multiple public MABIM regimes.
- Reproducibility: the sprint now has a compact manifest and runbook.

## What Did Not Improve

- Rule baseline: no behavioral separation was observed; no heuristic v2 was justified.
- Fleet: richer public FleetPy scenarios require `gurobipy`.
- Stochastic routing: SVRPBench remains the preferred target, but it was not integrated in this sprint.

## Safe Claim

CODEX PROJE has a stronger benchmark evidence package after targeted route and inventory upgrades. The project can defend its current model maturity, simulation benchmark breadth, and production guardrails without claiming global SOTA or real-world optimality.

## Unsafe Claim

Do not claim global SOTA, company-data validation, or production economic optimality.

