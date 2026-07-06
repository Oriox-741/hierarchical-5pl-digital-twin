# Fleet Dispatch Upgrade Sprint Audit - 2026-06-14

## Classification Before Independent Review

`FLEET_DISPATCH_UPGRADE_PARTIALLY_READY_WITH_BLOCKERS`

## Scope

This sprint upgraded public fleet/dispatch evidence with bounded public data and Gurobi-free simulator evidence. It did not train, run project offline eval, run long-run gates, register, activate, promote, update baselines, mutate DB, mutate checkpoints, or overwrite existing eval outputs.

## Main Outputs

Machine-readable:

- `reports/benchmarks/fleet_dispatch_upgrade_20260614/final_synthesis/fleet_dispatch_upgrade_summary.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/final_synthesis/fleet_dispatch_public_source_catalog.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/nyc_hvfhs_dispatch/nyc_hvfhs_dispatch_report.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/chicago_tnp_dispatch/chicago_tnp_dispatch_report.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/chicago_taxi_fleet/chicago_taxi_fleet_report.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/taxi_trajectory_fleet/taxi_trajectory_fleet_report.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/openmines_isolated/openmines_isolated_report.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/fleetpy_no_gurobi/fleetpy_no_gurobi_report.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/stochastic_dynamic_routing/stochastic_dynamic_routing_report.json`
- `reports/benchmarks/fleet_dispatch_upgrade_20260614/final_protected_state_audit.json`
- `reports/ops/20260614_fleet_dispatch_upgrade_final_ops_bundle_report.json`

Docs:

- `docs/reports/20260614_fleet_dispatch_public_source_research_tr.md`
- `docs/runs/20260614_nyc_hvfhs_dispatch_benchmark_report.md`
- `docs/runs/20260614_chicago_tnp_dispatch_benchmark_report.md`
- `docs/runs/20260614_chicago_taxi_fleet_proxy_report.md`
- `docs/runs/20260614_taxi_trajectory_fleet_proxy_report.md`
- `docs/runs/20260614_openmines_isolated_dispatch_report.md`
- `docs/runs/20260614_fleetpy_no_gurobi_report.md`
- `docs/runs/20260614_stochastic_dynamic_routing_upgrade_report.md`
- `docs/reports/20260614_fleet_dispatch_score_upgrade_results_tr.md`
- `docs/reports/20260614_fleet_dispatch_scorecard_update_tr.md`

Code and tests:

- `scripts/fleet_dispatch_upgrade_benchmarks.py`
- `tests/benchmarks/test_fleet_dispatch_upgrade_benchmarks.py`

## Completed Evidence

| Source | Result |
|---|---|
| NYC HVFHV January 2023 | 200,000-row deterministic stride sample from 18,479,031 source rows |
| FleetPy no-Gurobi | Existing IRS-only result plus two fresh RPP insertion scenarios |
| SF taxi trajectories | 200,000-row movement sample from 612,853 rows over 534 taxis |
| SVRPBench/amflorio | Dynamic `appear_times` summarized; amflorio public results metadata downloaded |

## Blockers

| Source | Blocker |
|---|---|
| Chicago TNP | Socrata HTTP 503 |
| Chicago Taxi | Socrata HTTP 503 |
| OpenMines | No Python 3.10/3.11 isolated interpreter; Python 3.12 dependency blocker remains |
| Yahias21 Hugging Face `.npz` | HTTP 401 |
| SVRPBench stochastic raw fields | Released parquet lacks persisted congestion/delay/accident/time-window fields |

## Score Recommendation

| Dimension | Previous | Recommended |
|---|---:|---:|
| Fleet/dispatch external evidence | 3.5 / 5 | 4.1 / 5 |
| Route/stochastic public evidence | 4.2 / 5 | 4.3 / 5 |
| SOTA pathway maturity | 4.2 / 5 | 4.3 / 5 |
| Thesis/advisor defensibility | 4.4 / 5 | 4.5 / 5 |

## Verification

Focused tests:

```powershell
python -m unittest tests.benchmarks.test_fleet_dispatch_upgrade_benchmarks -v
```

Result: `OK` (8 tests).

Relevant benchmark/reporting tests:

```powershell
python -m unittest tests.benchmarks.test_fleet_dispatch_upgrade_benchmarks tests.benchmarks.test_public_data_expansion_benchmarks tests.benchmarks.test_company_data_replay_validator tests.benchmarks.test_pyvrp_route_reference_benchmark tests.benchmarks.test_inventory_reorder_benchmark tests.benchmarks.test_mabim_inventory_benchmark tests.orchestration.test_production_artifact_health_report tests.orchestration.test_monitoring_report_generator -v
```

Result: `OK` (52 tests).

Compile check:

```powershell
python -m py_compile scripts\fleet_dispatch_upgrade_benchmarks.py tests\benchmarks\test_fleet_dispatch_upgrade_benchmarks.py
```

Result: passed.

Final ops bundle:

```powershell
python scripts\run_production_ops_bundle.py --output reports\ops\20260614_fleet_dispatch_upgrade_final_ops_bundle_report.json
```

Result: `OPS_BUNDLE_CLEAN`.

Protected audit:

- `reports/benchmarks/fleet_dispatch_upgrade_20260614/final_protected_state_audit.json`
- Result: `PROTECTED_STATE_CLEAN`

## No-Overclaim

This sprint supports a fleet/dispatch external evidence maturity upgrade. It does not support global SOTA, real-world deployment readiness, company-data validation, or action 24/32 economic optimality.

## Independent Review

Reviewer verdict:

`FLEET_DISPATCH_UPGRADE_APPROVED`

Reviewer noted one minor non-blocking doc drift in the SF taxi trajectory report time span. The report was corrected to match the JSON evidence.
