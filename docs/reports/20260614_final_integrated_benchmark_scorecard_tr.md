# Final Integrated Benchmark Scorecard - 2026-06-14

## Classification

`FINAL_INTEGRATED_SCORECARD_READY_WITH_LIMITATIONS`

| Dimension | Previous | Final | Strongest Evidence | Limiter |
|---|---:|---:|---|---|
| simulator-production readiness | 4.6 | 4.7 | internal gate and production audits | company telemetry still absent |
| old-production comparison | 4.5 | 4.6 | 20260611 equal-budget residual-watch eval | synthetic scenario comparator |
| multi-seed robustness | 4.1 | 4.2 | multi-seed robustness benchmark root | not company telemetry replay |
| runtime latency | 4.6 | 4.7 | runtime latency repeated benchmark and replay smoke | hardware dependent |
| rule-based baseline maturity | 4.3 | 4.5 | 5120 episode continuous-aware benchmark | simulator baselines only |
| inventory/reorder external evidence | 4.5 | 4.5 | inventory/MABIM reports | company reorder economics absent |
| route/stochastic public evidence | 4.3 | 4.4 | route reference reports | stochastic raw fields missing |
| fleet/dispatch external evidence | 4.1 | 4.2 | fleet dispatch upgrade sprint | Chicago/OpenMines blockers and no company contracts |
| public historical replay maturity | 4.35 | 4.45 | large replay reports | no propensities/rewards/full trajectories |
| company-data validation readiness | 4.35 | 4.45 | company replay bridge and request package | no private extract ingested |
| SOTA pathway maturity | 4.3 | 4.45 | SOTA pathway and final scorecard | external blockers and no company OPE |
| thesis/advisor defensibility | 4.6 | 4.75 | final advisor/thesis docs | citation placeholders remain |
| real-world deployment readiness | 3.2 | 3.25 | production handoff and ops bundles | no company replay/OPE/live telemetry |

## Interpretation

The final scorecard is stronger than the 2026-06-13 package because it integrates full public-data expansion, fleet/dispatch upgrade evidence, and large public historical replay. It remains deliberately conservative: public replay is not causal OPE, company-data validation is not yet performed, and global SOTA is not claimed.

Machine-readable scorecard: `reports/benchmarks/final_evidence_20260614/final_scorecard/final_integrated_scorecard.json`
