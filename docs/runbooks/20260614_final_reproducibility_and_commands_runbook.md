# Final Reproducibility and Commands Runbook - 2026-06-14

## Classification

`FINAL_REPRODUCIBILITY_AND_COMMANDS_RUNBOOK_READY`

## Rerun Safety Rules

Do not write under `models/registry`, `models/production`, `models/baselines`, `db`, existing `models/checkpoints`, or existing `models/eval`. Do not train, register, activate, promote, update baselines, mutate DB, or create training configs.

## Final Commands

```powershell
python -m unittest tests.benchmarks.test_public_historical_replay_adapter -v
python -m unittest tests.benchmarks.test_public_historical_replay_adapter tests.benchmarks.test_company_data_replay_validator tests.benchmarks.test_public_data_expansion_benchmarks tests.benchmarks.test_fleet_dispatch_upgrade_benchmarks tests.orchestration.test_production_artifact_health_report tests.orchestration.test_monitoring_report_generator tests.orchestration.test_production_ops_bundle -v
python -m py_compile scripts\public_historical_replay_adapter.py scripts\final_evidence_integration_writer.py tests\benchmarks\test_public_historical_replay_adapter.py
python scripts\run_production_ops_bundle.py --output reports\ops\20260614_final_evidence_integration_final_ops_bundle_report.json
```

## Data Paths

- `data/public/lade_20260614/delivery_jl.parquet`
- `data/public/nyc_hvfhs_20260614/fhvhv_tripdata_2023-01.parquet`
- `data/public/olist_20260614/`

## Output Root

`reports/benchmarks/final_evidence_20260614/`

## Stale-Doc Warning

Use 2026-06-14 final truth-source index and scorecard before quoting older bounded/smoke docs.
