# Final Benchmark Appendix for Advisor - 2026-06-14

## Classification

`FINAL_BENCHMARK_APPENDIX_READY`

## Evidence Families

1. Internal simulator-production: hierarchical v1 1M, equal-budget gate PASS, residual watches acceptable.
2. Rule-based and continuous-aware simulator baselines.
3. Inventory/reorder public references: gym-invmgmt and MABIM/ReplenishmentEnv.
4. Route references: PyVRP, OR-Tools, Amazon route proxy, SVRP geometry/dynamic appear-times.
5. Fleet/dispatch proxies: HVFHV, FleetPy no-Gurobi RPP/IRS, SF taxi, NYC TLC.
6. Public historical replay: LaDe, NYC HVFHS, Olist large replay.
7. Company-data readiness: synthetic harness and company replay bridge.

| Dataset | Rows | Action 24 | Action 32 | Dispatch | Missingness |
|---|---:|---:|---:|---:|---:|
| LaDe | 31415 | 0.0 | 0.0 | 0.004106318637593506 | 0.3051177454274308 |
| NYC_HVFHS | 100000 | 0.00051 | 2e-05 | 0.04092 | 0.3013698630136986 |
| Olist | 96476 | 0.002062689166217505 | 0.0 | 0.03643393175504789 | 0.5479511690607132 |

Public replay is descriptive. OPE requires propensities, alternatives, rewards, and trajectories.
