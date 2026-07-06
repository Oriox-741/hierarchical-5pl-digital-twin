# Public Data Expansion Sprint Audit - 2026-06-14

## Classification Before Independent Review

`PUBLIC_DATA_EXPANSION_PARTIALLY_READY_WITH_BLOCKERS`

## Scope

The sprint expanded benchmark evidence with bounded public/anonymized logistics datasets and publication-grade comparison artifacts. It did not train, run project offline eval, run long-run gates, register, activate, promote, update baselines, mutate DB, mutate checkpoints, or edit existing eval outputs.

## Files Added Or Changed

Code and tests:

- `scripts/public_data_expansion_benchmarks.py`
- `scripts/company_data_replay_validator.py`
- `tests/benchmarks/test_public_data_expansion_benchmarks.py`
- `tests/benchmarks/test_company_data_replay_validator.py`

Reports and docs:

- `docs/reports/20260614_public_anonymized_logistics_dataset_catalog_tr.md`
- `docs/runs/20260614_lade_last_mile_public_data_report.md`
- `docs/runs/20260614_planned_vs_actual_route_dataset_report.md`
- `docs/runs/20260614_olist_ecommerce_logistics_report.md`
- `docs/runs/20260614_public_demand_reorder_proxy_report.md`
- `docs/runs/20260614_gurobi_free_fleet_dispatch_report.md`
- `docs/runs/20260614_svrpbench_direct_integration_report.md`
- `docs/runbooks/20260614_company_data_replay_validation_harness_spec.md`
- `docs/runs/20260614_company_data_validation_harness_report.md`
- `docs/reports/20260614_publication_grade_comparison_and_ablation_plan_tr.md`
- `docs/runs/20260614_publication_ablation_package_report.md`
- `docs/reports/20260614_public_data_expansion_results_tr.md`
- `docs/reports/20260614_public_data_scorecard_update_tr.md`
- `docs/00_PROJECT_DASHBOARD.md`

Machine-readable outputs:

- `reports/benchmarks/public_data_expansion_20260614/public_dataset_catalog/public_dataset_catalog.json`
- `reports/benchmarks/public_data_expansion_20260614/lade_last_mile/lade_preflight_report.json`
- `reports/benchmarks/public_data_expansion_20260614/planned_vs_actual_routes/planned_actual_route_report.json`
- `reports/benchmarks/public_data_expansion_20260614/olist_ecommerce_logistics/olist_logistics_report.json`
- `reports/benchmarks/public_data_expansion_20260614/m5_tesco_demand_inventory/demand_reorder_public_proxy_report.json`
- `reports/benchmarks/public_data_expansion_20260614/nyc_tlc_fleet_dispatch/nyc_tlc_dispatch_report.json`
- `reports/benchmarks/public_data_expansion_20260614/openmines_dispatch/openmines_dispatch_report.json`
- `reports/benchmarks/public_data_expansion_20260614/svrpbench_stochastic/svrpbench_integration_report.json`
- `reports/benchmarks/public_data_expansion_20260614/company_data_validation_harness/company_data_harness_report.json`
- `reports/benchmarks/public_data_expansion_20260614/publication_ablation_package/historical_ablation_table.json`
- `reports/benchmarks/public_data_expansion_20260614/final_synthesis/public_data_expansion_summary.json`
- `reports/benchmarks/public_data_expansion_20260614/final_protected_state_audit.json`
- `reports/ops/20260614_public_data_expansion_final_ops_bundle_report.json`

## Dataset Outcomes

| Area | Outcome |
|---|---|
| LaDe last-mile | Ready as bounded public sample, with publication license caveat |
| Planned-vs-actual route data | Blocked by API 403 |
| Olist ecommerce logistics | Ready |
| Tesco demand proxy | Partial, ready as area-level demand proxy |
| NYC TLC FHV dispatch proxy | Ready as public dispatch-pressure proxy |
| OpenMines | Blocked by Python 3.12 / `numpy==1.25.0` install issue |
| SVRPBench | Base geometry ready; stochastic fields missing from released parquet |
| Company-data harness | Synthetic-only replay validator ready |
| Publication ablation package | Ready |

## Verification

Focused tests:

```powershell
python -m unittest tests.benchmarks.test_public_data_expansion_benchmarks tests.benchmarks.test_company_data_replay_validator -v
```

Result: `OK` (11 tests).

Relevant benchmark/orchestration tests:

```powershell
python -m unittest tests.benchmarks.test_public_data_expansion_benchmarks tests.benchmarks.test_company_data_replay_validator tests.benchmarks.test_pyvrp_route_reference_benchmark tests.benchmarks.test_inventory_reorder_benchmark tests.benchmarks.test_mabim_inventory_benchmark tests.orchestration.test_production_artifact_health_report tests.orchestration.test_monitoring_report_generator -v
```

Result: `OK` (44 tests).

Compile check:

```powershell
python -m py_compile scripts\public_data_expansion_benchmarks.py scripts\company_data_replay_validator.py tests\benchmarks\test_public_data_expansion_benchmarks.py tests\benchmarks\test_company_data_replay_validator.py
```

Result: passed.

Final ops bundle:

```powershell
python scripts\run_production_ops_bundle.py --output reports\ops\20260614_public_data_expansion_final_ops_bundle_report.json
```

Result: `OPS_BUNDLE_CLEAN`.

Protected audit:

- `reports/benchmarks/public_data_expansion_20260614/final_protected_state_audit.json`
- Result: `PROTECTED_STATE_CLEAN`
- Protected file hashes unchanged.
- Protected path profiles unchanged.
- No train/eval/gate/registry/AWS mutation process found.

## Independent Review

First reviewer verdict:

`PUBLIC_DATA_EXPANSION_NEEDS_FIXES`

Requested fixes:

- Do not hard-code `synthetic_only=true` for caller-supplied fixture roots.
- Reject protected output paths in `scripts/company_data_replay_validator.py`.

Fixes applied:

- `--fixture-root` now reports `synthetic_only=false` and `input_mode=caller_supplied_fixture_root`.
- Built-in fixture mode still reports `synthetic_only=true` and `input_mode=built_in_synthetic_fixture`.
- `write_report` now rejects writes under `models/registry`, `models/production`, `models/baselines`, `models/checkpoints`, `models/eval`, and `db` before directory creation or file write.
- Regression tests cover both issues.

Re-review verdict:

`PUBLIC_DATA_EXPANSION_APPROVED`

Residual classification remains partial-ready because several public sources are blocked or caveated, not because the implemented package needs fixes.

Expected reviewer verdict options:

- `PUBLIC_DATA_EXPANSION_APPROVED`
- `PUBLIC_DATA_EXPANSION_NEEDS_FIXES`
- `PUBLIC_DATA_EXPANSION_BLOCKED`
