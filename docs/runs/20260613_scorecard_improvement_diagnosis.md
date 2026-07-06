# Scorecard Improvement Diagnosis

Date: 2026-06-14 local time. Sprint folder: `reports/benchmarks/scorecard_upgrade_20260613/`.

## Decision

`SCORECARD_IMPROVEMENT_DIAGNOSIS_READY`

## Why Scores Were Not Higher

| Area | Prior score | Primary limiter | Upgrade feasible without company data? | Company data needed for 5/5? |
| --- | ---: | --- | --- | --- |
| Rule-based continuous baseline | 4.0 / 5 | Heuristic continuous controls changed internally but aggregate outcomes matched neutral/oracle modes. | Diagnostic only; no honest v2 formula yet. | Yes, for cost/control calibration. |
| Inventory/reorder evidence | 4.0 / 5 | Public inventory evidence existed but was narrow in seeds and stress regimes. | Yes, by adding multi-seed `gym-invmgmt` and more MABIM regimes. | Yes, for real reorder economics. |
| Fleet/dispatch evidence | 3.0 / 5 | Only one public FleetPy immediate-offer scenario completed. | Attempted; richer FleetPy scenarios blocked by Gurobi. | Yes, or a richer open dispatch benchmark. |
| Route benchmark evidence | 3.5 / 5 | PyVRP CVRP worked, but VRPTW was blocked by Solomon TXT format. | Yes, by adding a Solomon parser/converter. | Not for route-only references, but yes for full 5PL action economics. |
| SOTA pathway maturity | 3.5 / 5 | Evidence existed across tracks but lacked a single reproducibility/readiness package and had unresolved stochastic/congestion blockers. | Partially. | Yes for final operational validity. |
| Thesis-defense readiness | 4.0 / 5 | Strong narrative existed but needed upgraded evidence accounting and explicit blocker framing. | Partially. | Yes for real-world deployment claims. |

## Causal Findings

The rule baseline is not weak because the current heuristic is obviously wrong. It is limited because the current 5PL scenario/action path is not sensitive enough to separate neutral, heuristic, and oracle diagnostic continuous controls in aggregate outcomes.

Inventory evidence was under-scored because it had too little public breadth. That was the highest-yield score upgrade target.

Fleet evidence is dependency-blocked, not analysis-blocked. FleetPy richer scenarios reached initialization and then failed on `gurobipy`, so the honest outcome is to retain the blocker.

Route evidence had a fixable software gap: local Solomon VRPTW files were valid benchmark data but not VRPLIB-compatible. The added parser turns those into PyVRP models and removes that blocker.

## Upgrade Policy

Scores are increased only where fresh evidence changes the maturity level. Diagnostics alone do not justify a score bump, and blocked external dependencies are preserved as blockers rather than converted into claims.

