# FleetPy No-Gurobi Report - 2026-06-14

## Classification

`FLEETPY_NO_GUROBI_RPP_AND_IRS_READY`

JSON report:

- `reports/benchmarks/fleet_dispatch_upgrade_20260614/fleetpy_no_gurobi/fleetpy_no_gurobi_report.json`

## Evidence

Local FleetPy source:

- `data/public/fleetpy_20260613/source/FleetPy-main`

Successful no-Gurobi scenarios:

| Scenario | Users | Served online users | Waiting time | Fleet utilization | Empty vkm | Shared rides |
|---|---:|---:|---:|---:|---:|---:|
| `example_pool_irsonly_sc_1` | 93 | 93.00% | 152.00 | 74.58% | 34.43% | 32.26% |
| `example_rpp_tau_80_True_1` | 127 | 97.69% | 763.20 | 50.21% | 23.98% | 60.63% |
| `example_rpp_tau_90_True_1` | 128 | 98.46% | 921.42 | 49.64% | 28.74% | 68.75% |

Fresh run:

```powershell
python -c "from run_scenarios import run_scenarios; run_scenarios('studies/example_study/scenarios/constant_config_rpp.csv', 'studies/example_study/scenarios/example_rpp.csv', n_parallel_sim=1, n_cpu_per_sim=1, evaluate=1, log_level='info')"
```

The run completed both RPP scenarios with `RPPFleetControlFullInsertion`.

## Remaining Gurobi Blockers

FleetPy paths using `AlonsoMora`, `SimonettoAssignment`, advanced repositioning, and related optimizer/repositioning components still require `gurobipy` and are not counted as ready.

