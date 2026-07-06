# Step 4 Company Data Intake Validator Report

Date: 2026-06-12

## Final Classification

`STEP4_COMPANY_DATA_INTAKE_VALIDATOR_READY`

## Objective

Execute `docs/goals/20260611_execute_step4_company_data_intake_validator_goal.txt`: implement a safe-now company-data intake validator skeleton using synthetic headers and fixtures only.

This implementation validates expected schemas and join-key availability for future company data. It does not ingest private company data, create DB tables, read live production/company extracts, make calibration conclusions, train, run offline eval, run a long-run gate, download datasets, create training configs, or mutate registry, production, baselines, DB, checkpoints, or existing eval outputs.

## Files Changed

- `scripts/company_data_intake_validator.py`
- `tests/orchestration/test_company_data_intake_validator.py`
- `reports/company_data/20260611_company_data_synthetic_schema_report.json`
- `docs/runs/20260611_step4_company_data_intake_validator_report.md`
- `docs/00_PROJECT_DASHBOARD.md` link/status update only

## Validator Summary

Implemented read-only synthetic-only CLI:

```powershell
python scripts/company_data_intake_validator.py --use-built-in-synthetic-fixture valid_minimal --output reports/company_data/20260611_company_data_synthetic_schema_report.json
```

Supported input modes:

- `--use-built-in-synthetic-fixture valid_minimal|missing_required_columns|bad_timestamps|duplicate_keys|invalid_labels`
- `--synthetic-fixture-root PATH`

The CLI requires exactly one synthetic input mode and `--output PATH`. It writes only the requested JSON output path.

The validator uses only Python standard library modules and does not import project runtime/model/training/eval modules.

## Table Families

Implemented synthetic schema checks for:

- `orders`
- `dispatch_attempts`
- `deliveries`
- `routes`
- `fleet`
- `inventory`
- `costs`

Join-key coverage:

- `order_id`
- `dispatch_attempt_id`
- `vehicle_id`
- `carrier_id`
- `route_id`
- `sku_id`
- `site_id`
- timestamp ordering across order, dispatch, pickup, and delivery records

## Data-Quality Checks

Implemented checks for:

- required columns,
- required table presence,
- duplicate primary keys,
- timestamp parseability,
- timezone-naive timestamp warnings,
- promised window ordering,
- `decision_timestamp <= pickup_timestamp <= delivery_timestamp`,
- `dispatch_attempt_timestamp <= pickup_timestamp`,
- non-negative costs and numeric operational quantities,
- `fleet_type` in `primary|secondary`,
- `reorder_type` in `none|conservative|aggressive|emergency`,
- `action_id` integer in `0..47`,
- required joins and warning-level carrier-only join mismatch.

## Output Schema

Generated JSON top-level blocks:

- `report_metadata`
- `table_summaries`
- `join_key_checks`
- `data_quality_checks`
- `warnings`
- `decision`

Generated report:

- Path: `reports/company_data/20260611_company_data_synthetic_schema_report.json`
- SHA256: `6C303565847105D0D30BF362050ADCE75A3ED945D782FA0AA49BBE485489BFED`
- Source kind: `built_in:valid_minimal`
- Synthetic data: `true`
- Table count: `7`
- Data-quality checks: `39`
- Join-key checks: `9`
- Warnings: `0`
- Failures/blocked checks: `0`
- Decision classification: `COMPANY_DATA_SCHEMA_READY`

## Classifications

Implemented:

- `COMPANY_DATA_SCHEMA_READY`
- `COMPANY_DATA_SCHEMA_WARNINGS_ONLY`
- `COMPANY_DATA_SCHEMA_NEEDS_FIXES`
- `COMPANY_DATA_SCHEMA_BLOCKED`

Classification priority:

1. missing required table or unreadable fixture -> blocked
2. missing required columns, duplicate primary keys, malformed timestamps, impossible ordering, invalid labels/action id, or required join failure -> needs fixes
3. warning-level join/timestamp semantics -> warnings only
4. all required checks pass -> ready

## TDD Evidence

Red run before implementation:

```powershell
python -m unittest tests.orchestration.test_company_data_intake_validator -v
```

Result:

```text
ImportError: cannot import name 'company_data_intake_validator' from 'scripts'
FAILED (errors=1)
```

Green run after implementation:

```powershell
python -m unittest tests.orchestration.test_company_data_intake_validator -v
```

Result:

```text
Ran 14 tests in 0.059s
OK
```

Covered cases:

- valid built-in minimal fixture -> `COMPANY_DATA_SCHEMA_READY`
- missing required columns -> `COMPANY_DATA_SCHEMA_NEEDS_FIXES`
- bad timestamps -> `COMPANY_DATA_SCHEMA_NEEDS_FIXES`
- duplicate primary keys -> `COMPANY_DATA_SCHEMA_NEEDS_FIXES`
- invalid fleet/reorder labels -> `COMPANY_DATA_SCHEMA_NEEDS_FIXES`
- blank required join/key values -> `COMPANY_DATA_SCHEMA_NEEDS_FIXES`
- missing required table -> `COMPANY_DATA_SCHEMA_BLOCKED`
- missing required join key -> `COMPANY_DATA_SCHEMA_NEEDS_FIXES`
- carrier-only join mismatch with valid vehicle join -> `COMPANY_DATA_SCHEMA_WARNINGS_ONLY`
- optional fields absent do not block readiness
- synthetic fixture root CSV loader
- CLI writes only requested output path
- CLI refuses missing synthetic input
- CLI refuses conflicting synthetic input modes

## Verification

Focused tests:

```powershell
python -m unittest tests.orchestration.test_company_data_intake_validator -v
```

Exit code: `0`

Compile:

```powershell
python -m py_compile scripts/company_data_intake_validator.py tests/orchestration/test_company_data_intake_validator.py
```

Exit code: `0`

Note: one earlier parallel run of tests and `py_compile` produced a Windows `__pycache__` rename access error. Sequential `py_compile` passed, confirming the issue was a parallel bytecode-write race rather than a source compile failure.

Generate report:

```powershell
python scripts/company_data_intake_validator.py --use-built-in-synthetic-fixture valid_minimal --output reports/company_data/20260611_company_data_synthetic_schema_report.json
```

Exit code: `0`

Stdout:

```text
COMPANY_DATA_SCHEMA_READY
```

Classification check:

```powershell
Select-String -Path reports\company_data\20260611_company_data_synthetic_schema_report.json -Pattern '"classification": "COMPANY_DATA_SCHEMA_READY"'
```

Result: found.

## Protected No-Mutation Proof

Pre/post protected hashes match:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Protected path profiles match:

- `models/baselines`: `2692` files, `7392576274` bytes
- `db`: `7` files, `28330` bytes
- `models/checkpoints`: `787` files, `4920830666` bytes
- `models/production`: `8` files, `328265276` bytes
- `models/eval`: `128` files, `388267659` bytes

Process scan:

```text
NO_TRAIN_EVAL_GATE_AWS_OR_PRIVATE_DATA_PROCESS
```

## Non-Goals Preserved

- No private company data read or write.
- No live production/company extract read.
- No DB table creation, DB write, or migration.
- No calibration conclusions from synthetic data.
- No training.
- No offline eval.
- No long-run gate.
- No dataset download.
- No registry update.
- No mutation to `active_models.json` or `models.jsonl`.
- No production mutation.
- No baseline mutation.
- No checkpoint mutation.
- No existing eval-output mutation.
- No training config creation.
- No 1.5M/2M/3M/5M/10M/100M start.

## Independent Review

Initial reviewer verdict: `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_NEEDS_FIXES`

Fix applied after reviewer verdict:

- Added explicit required-value validation for every required field in every table.
- Added tests proving blank `inventory.sku_id` / `inventory.site_id` values return `COMPANY_DATA_SCHEMA_NEEDS_FIXES`.
- Added tests proving a missing required table returns `COMPANY_DATA_SCHEMA_BLOCKED`.
- Regenerated the synthetic schema report after the fix.

Final reviewer verdict: `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_APPROVED`
