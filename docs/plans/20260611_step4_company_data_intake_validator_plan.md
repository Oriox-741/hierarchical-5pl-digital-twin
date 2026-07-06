# Step 4 Company Data Intake Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe-now company-data intake validator skeleton that validates expected schemas and join-key availability using synthetic fixtures only.

**Architecture:** The validator is a read-only CLI that loads synthetic CSV fixtures from a temp/synthetic fixture root or built-in synthetic fixtures, validates required columns, joins, timestamps, labels, duplicate primary keys, and cost sign constraints, then writes one JSON schema-readiness report. It must never ingest private company data, create DB tables, train, eval, gate, download datasets, or mutate protected model artifacts.

**Tech Stack:** Python standard library (`argparse`, `csv`, `json`, `datetime`, `pathlib`, `tempfile`, `unittest`), existing docs/runbooks as schema sources, JSON report output.

---

## Executive Summary

Step 4 prepares the safety rail that should exist before any real company extract is accepted. The future validator will not perform calibration and will not ingest private data. It will only prove that a synthetic fixture shaped like the requested company extract can be checked for schema, join keys, timestamp semantics, duplicate keys, controlled labels, and non-negative costs.

The desired outcome of the future implementation is a generated JSON report at `reports/company_data/20260611_company_data_synthetic_schema_report.json` with one of these classifications:

- `COMPANY_DATA_SCHEMA_READY`
- `COMPANY_DATA_SCHEMA_WARNINGS_ONLY`
- `COMPANY_DATA_SCHEMA_NEEDS_FIXES`
- `COMPANY_DATA_SCHEMA_BLOCKED`

## Why This Comes After Step 3

Step 3 created the monitoring report generator skeleton and proved that protected artifact-health state can be combined with synthetic/local telemetry fixtures. Step 4 extends that same safe-now pattern upstream into company-data readiness:

- Step 3 answers: can we produce a monitoring report from known-safe local fixture inputs?
- Step 4 answers: can we validate the shape and relational integrity of future company-data extracts before any private data is ingested?

This order matters because monitoring classifications and action 24/32 watches define which real-world fields matter. The intake validator should therefore follow the KPI dictionary, monitoring report spec, and company-data request package rather than inventing a data model.

## Input Contract From Company Data Request Package

Source document: `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`.

The future validator should model seven table families:

1. `orders`
2. `dispatch_attempts`
3. `deliveries`
4. `routes`
5. `fleet`
6. `inventory`
7. `costs`

The first implementation must use synthetic fixtures only:

- built-in synthetic fixture rows for the CLI smoke report,
- temp CSV fixtures inside tests,
- no checked-in private data,
- no database connection,
- no production extract path scanning,
- no automatic discovery of company files.

## Required And Optional Fields

### `orders`

Required:

- `order_id`
- `created_at`
- `promised_window_start`
- `promised_window_end`
- `order_priority`
- `order_size`
- `zone_id`

Optional:

- `customer_id`
- `customer_location_id`
- `latitude`
- `longitude`
- `sku_id`
- `site_id`

Primary key: `order_id`.

### `dispatch_attempts`

Required:

- `dispatch_attempt_id`
- `order_id`
- `decision_timestamp`
- `dispatch_attempt_timestamp`
- `dispatch_status`
- `dispatch_failure_reason`
- `action_id`
- `vehicle_id`
- `carrier_id`
- `fleet_type`
- `route_id`

Optional:

- `no_vehicle_flag`
- `already_assigned_flag`
- `route_failure_flag`
- `carrier_rejected_flag`
- `duplicate_assignment_flag`
- `policy_version`

Primary key: `dispatch_attempt_id`.

### `deliveries`

Required:

- `order_id`
- `dispatch_attempt_id`
- `pickup_timestamp`
- `delivery_timestamp`
- `delivery_status`
- `delivered_on_time`

Optional:

- `failed_attempt_reason`
- `lateness_minutes`
- `actual_sequence_rank`

Primary key: `dispatch_attempt_id`.

### `routes`

Required:

- `route_id`
- `order_id`
- `planned_route_type`
- `planned_distance`
- `planned_travel_time`
- `route_failure_flag`

Optional:

- `actual_route_id`
- `actual_distance`
- `actual_travel_time`
- `planned_sequence_rank`
- `actual_sequence_rank`
- `reroute_flag`
- `congestion_delay_flag`
- `route_provider`

Primary key: `route_id`.

### `fleet`

Required:

- `vehicle_id`
- `carrier_id`
- `fleet_type`
- `vehicle_capacity`
- `available_at_decision`
- `assigned_at`
- `fleet_cost`

Optional:

- `accepted_at`
- `cancelled_at`
- `fleet_reliability_score`

Primary key: `vehicle_id`.

### `inventory`

Required:

- `sku_id`
- `site_id`
- `order_id`
- `stock_on_hand_at_decision`
- `backlog_quantity`
- `stockout_flag`
- `supplier_lead_time`

Optional:

- `allocated_quantity`
- `reorder_event_id`
- `reorder_type`
- `supplier_id`

Primary key: `sku_id` plus `site_id` plus `order_id`.

### `costs`

Required:

- `order_id`
- `primary_fleet_cost`
- `secondary_fleet_cost`
- `lateness_penalty_cost`
- `stockout_cost`
- `holding_cost`

Optional:

- `route_cost`
- `emergency_reorder_cost`
- `cancellation_cost`

Primary key: `order_id`.

## Join-Key Rules

The validator should emit table-level and cross-table findings for these rules:

- `order_id` must exist in `orders`, `dispatch_attempts`, `deliveries`, `routes`, `inventory`, and `costs`.
- Every `dispatch_attempts.order_id` must exist in `orders.order_id`.
- Every `deliveries.order_id` and `deliveries.dispatch_attempt_id` must match `orders` and `dispatch_attempts`.
- Every `routes.order_id` must exist in `orders`; every `dispatch_attempts.route_id` must exist in `routes.route_id`.
- Every `dispatch_attempts.vehicle_id` must exist in `fleet.vehicle_id`.
- Every `dispatch_attempts.carrier_id` should exist in `fleet.carrier_id`; missing carrier join is a warning if `vehicle_id` joins.
- Every `inventory.order_id` must exist in `orders`.
- `inventory.sku_id` plus `inventory.site_id` must be present wherever inventory validation is claimed.
- Timestamps must carry shared timezone semantics; the skeleton should accept ISO-8601 UTC strings ending in `Z` or explicit offsets and classify timezone-naive values as schema warnings or fixes depending on severity.

## Data-Quality Rules

Required checks:

- required columns present for each table,
- all required timestamps parse,
- `promised_window_start <= promised_window_end`,
- if present, `decision_timestamp <= pickup_timestamp <= delivery_timestamp`,
- if present, `dispatch_attempt_timestamp <= pickup_timestamp`,
- non-negative numeric costs,
- non-negative route distance/travel-time/capacity/quantity fields,
- `fleet_type` in `primary|secondary`,
- `reorder_type` in `none|conservative|aggressive|emergency` when present and non-empty,
- `action_id` integer in `0..47`,
- no duplicate primary keys,
- join-key availability and cross-table coverage,
- empty optional fields do not block readiness,
- malformed required values produce table-specific issues and a global classification.

## Output Report Schema

Future report shape:

```json
{
  "report_metadata": {
    "report_timestamp_utc": "string",
    "validator_version": "company_data_intake_validator_v1",
    "synthetic_data": true,
    "source_kind": "built_in:valid_minimal",
    "table_count": 7
  },
  "table_summaries": {
    "orders": {
      "rows": 0,
      "required_columns_present": true,
      "optional_columns_present": [],
      "duplicate_primary_keys": 0,
      "issues": []
    }
  },
  "join_key_checks": [],
  "data_quality_checks": [],
  "warnings": [],
  "decision": {
    "classification": "COMPANY_DATA_SCHEMA_READY",
    "reason": "string"
  }
}
```

## Classification Rules

Priority order:

1. Missing required table or unreadable fixture -> `COMPANY_DATA_SCHEMA_BLOCKED`.
2. Missing required columns, duplicate primary keys, malformed required timestamps, impossible required ordering, invalid controlled labels, or invalid action id -> `COMPANY_DATA_SCHEMA_NEEDS_FIXES`.
3. Missing optional fields, missing non-critical carrier join where vehicle join is valid, or timezone-naive timestamps with otherwise parseable semantics -> `COMPANY_DATA_SCHEMA_WARNINGS_ONLY`.
4. All required schemas, joins, timestamps, labels, and non-negative constraints pass -> `COMPANY_DATA_SCHEMA_READY`.

No classification should authorize private data ingestion, DB mutation, calibration conclusions, training, eval, gate execution, registry mutation, production mutation, baseline mutation, checkpoint mutation, or existing eval-output mutation.

## Proposed Future Files

- Create: `scripts/company_data_intake_validator.py`
  - Owns table schema constants, CSV loaders, built-in synthetic fixtures, validators, report builder, classification logic, and CLI.
- Create: `tests/orchestration/test_company_data_intake_validator.py`
  - Owns TDD coverage for schema, joins, data quality, classifications, and CLI write scope.
- Create: `reports/company_data/20260611_company_data_synthetic_schema_report.json`
  - Generated by the future CLI from built-in synthetic valid fixture only.
- Create: `docs/runs/20260611_step4_company_data_intake_validator_report.md`
  - Execution report with TDD evidence, generated report hash, reviewer verdict, and protected no-mutation proof.
- Modify: `docs/00_PROJECT_DASHBOARD.md`
  - Link/status update only.

## Synthetic Fixture Design

All fixtures must be synthetic and small.

### `valid_minimal`

One order, one dispatch attempt, one delivery, one route, one fleet row, one inventory row, and one cost row. It should include:

- action `24` or `32` value in `0..47`,
- `fleet_type=secondary`,
- `reorder_type=none`,
- UTC timestamps with `Z`,
- non-negative costs,
- matching join keys across all tables.

Expected classification: `COMPANY_DATA_SCHEMA_READY`.

### `missing_required_columns`

Remove `promised_window_end` from `orders` and `dispatch_attempt_id` from `dispatch_attempts`.

Expected classification: `COMPANY_DATA_SCHEMA_NEEDS_FIXES`.

### `bad_timestamps`

Use an unparsable `created_at` and a promised window where start is after end.

Expected classification: `COMPANY_DATA_SCHEMA_NEEDS_FIXES`.

### `duplicate_keys`

Duplicate `orders.order_id` and `dispatch_attempts.dispatch_attempt_id`.

Expected classification: `COMPANY_DATA_SCHEMA_NEEDS_FIXES`.

### `invalid_labels`

Use `fleet_type=overflow_unknown` and `reorder_type=delay_forever`.

Expected classification: `COMPANY_DATA_SCHEMA_NEEDS_FIXES`.

### `warnings_only_optional_gaps`

Omit optional columns such as `actual_route_id`, `accepted_at`, and `route_cost`; keep all required fields valid.

Expected classification: `COMPANY_DATA_SCHEMA_WARNINGS_ONLY` only if the implementation chooses to warn on optional gaps. If optional gaps are reported as informational, the test should instead assert the explicit info-level issue shape and keep `COMPANY_DATA_SCHEMA_READY`.

## TDD Requirements

Write failing tests before implementation.

Required tests:

- clean valid synthetic schema returns `COMPANY_DATA_SCHEMA_READY`,
- missing required columns returns `COMPANY_DATA_SCHEMA_NEEDS_FIXES`,
- bad timestamps return `COMPANY_DATA_SCHEMA_NEEDS_FIXES`,
- duplicate primary keys return `COMPANY_DATA_SCHEMA_NEEDS_FIXES`,
- invalid fleet/reorder labels return `COMPANY_DATA_SCHEMA_NEEDS_FIXES`,
- missing required join key returns `COMPANY_DATA_SCHEMA_NEEDS_FIXES`,
- optional gaps do not block readiness beyond the documented warning/info behavior,
- CLI writes only the requested report output path,
- CLI refuses to run without either built-in synthetic fixture or explicit synthetic fixture root,
- no code path connects to DB, scans production extracts, reads registry for writes, or mutates protected artifacts.

## Verification Commands

Focused tests:

```powershell
python -m unittest tests.orchestration.test_company_data_intake_validator -v
```

Compile:

```powershell
python -m py_compile scripts/company_data_intake_validator.py tests/orchestration/test_company_data_intake_validator.py
```

Generate synthetic report:

```powershell
python scripts/company_data_intake_validator.py --use-built-in-synthetic-fixture valid_minimal --output reports/company_data/20260611_company_data_synthetic_schema_report.json
```

Classification check:

```powershell
Select-String -Path reports\company_data\20260611_company_data_synthetic_schema_report.json -Pattern '"classification": "COMPANY_DATA_SCHEMA_READY"'
```

Protected process scan:

```powershell
$self = $PID
$matches = Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and ($_.Name -match 'python|aws|mlflow|tensorboard' -or $_.CommandLine -match 'train_joint|offline_scenarios|long_run_gate|aws s3|private') } | Select-Object ProcessId,Name,CommandLine
if ($matches) { $matches | Format-Table -Wrap -AutoSize } else { Write-Output 'NO_TRAIN_EVAL_GATE_AWS_OR_PRIVATE_DATA_PROCESS' }
```

Protected hash/profile checks should compare the same protected files and folders used in Step 3.

## Independent Review Requirement

After the future implementation, spawn one `gpt-5.5` read-only reviewer.

Reviewer scope:

- read the Step 4 plan and executable goal,
- read the company-data request package, KPI dictionary, monitoring report spec, and lifecycle governance runbook,
- inspect `scripts/company_data_intake_validator.py`,
- inspect `tests/orchestration/test_company_data_intake_validator.py`,
- inspect the generated synthetic schema report,
- inspect the Step 4 run report and dashboard update,
- verify no private data ingestion, DB mutation, training, eval, gate, dataset download, registry mutation, production mutation, baseline mutation, checkpoint mutation, or existing eval-output mutation.

Reviewer verdict exactly one:

- `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_APPROVED`
- `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_NEEDS_FIXES`
- `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_BLOCKED`

## Non-Goals

- No private company data read or write.
- No DB tables or migrations.
- No calibration conclusions from synthetic data.
- No monitoring report changes except optional interface compatibility.
- No training.
- No offline eval.
- No long-run gate.
- No dataset download.
- No registry update.
- No mutation of `active_models.json` or `models.jsonl`.
- No production mutation.
- No baseline mutation.
- No checkpoint mutation.
- No existing eval-output mutation.
- No training configs.
- No 1.5M/2M/3M/5M/10M/100M run.

## Protected No-Mutation Proof Required

The future execution report must include pre/post evidence for:

- `models/registry/active_models.json` hash,
- `models/registry/models.jsonl` hash,
- current production joint checkpoint hash,
- current production manifest hash,
- `models/baselines` file count and bytes,
- `db` file count and bytes,
- `models/checkpoints` file count and bytes,
- `models/production` file count and bytes,
- `models/eval` file count and bytes,
- no train/eval/gate/AWS/private-data process.

## Exact Future Goal Command

```text
/goal Read docs/goals/20260611_execute_step4_company_data_intake_validator_goal.txt and execute it exactly.
```

## Implementation Tasks

### Task 1: Create Failing Tests

**Files:**

- Create: `tests/orchestration/test_company_data_intake_validator.py`

- [ ] **Step 1: Add imports and valid fixture smoke test**

Use `unittest`, `TemporaryDirectory`, `json`, `csv`, and import `scripts.company_data_intake_validator as validator`.

Test name: `test_valid_minimal_builtin_fixture_is_ready`.

Expected behavior:

- `validator.built_in_synthetic_fixture("valid_minimal")` returns seven table families.
- `validator.build_company_data_schema_report(..., source_kind="built_in:valid_minimal", synthetic_data=True)` returns top-level keys `report_metadata`, `table_summaries`, `join_key_checks`, `data_quality_checks`, `warnings`, `decision`.
- decision classification is `COMPANY_DATA_SCHEMA_READY`.

- [ ] **Step 2: Add required negative tests**

Add tests for:

- `missing_required_columns`,
- `bad_timestamps`,
- `duplicate_keys`,
- `invalid_labels`,
- missing join key by changing one `dispatch_attempts.order_id` to a non-existent order,
- CLI output write scope in a temp dir.

Expected first run:

```powershell
python -m unittest tests.orchestration.test_company_data_intake_validator -v
```

Expected result before implementation: import error or missing attribute failure.

### Task 2: Implement Validator Skeleton

**Files:**

- Create: `scripts/company_data_intake_validator.py`

- [ ] **Step 1: Define schema constants**

Define `TABLE_SCHEMAS`, `PRIMARY_KEYS`, `TIMESTAMP_FIELDS`, `CONTROLLED_LABELS`, `NON_NEGATIVE_FIELDS`, and `JOIN_RULES` using the fields in this plan.

- [ ] **Step 2: Implement synthetic fixture builder**

Implement `built_in_synthetic_fixture(name: str) -> dict[str, list[dict[str, str]]]` for:

- `valid_minimal`,
- `missing_required_columns`,
- `bad_timestamps`,
- `duplicate_keys`,
- `invalid_labels`.

- [ ] **Step 3: Implement validation helpers**

Implement:

- `validate_required_columns`,
- `validate_duplicate_primary_keys`,
- `validate_timestamps`,
- `validate_controlled_labels`,
- `validate_non_negative_fields`,
- `validate_join_keys`,
- `classify_schema_report`.

- [ ] **Step 4: Implement report builder and CLI**

Implement:

- `build_company_data_schema_report(tables, source_kind, synthetic_data)`,
- `load_fixture_root(path)`,
- `main(argv=None)`.

CLI args:

- `--use-built-in-synthetic-fixture valid_minimal|missing_required_columns|bad_timestamps|duplicate_keys|invalid_labels`,
- `--synthetic-fixture-root PATH`,
- `--output PATH`.

The CLI must require exactly one synthetic input mode and write only the explicit `--output` JSON path.

### Task 3: Verify And Generate Report

**Files:**

- Create: `reports/company_data/20260611_company_data_synthetic_schema_report.json`
- Create: `docs/runs/20260611_step4_company_data_intake_validator_report.md`
- Modify: `docs/00_PROJECT_DASHBOARD.md` link/status only

- [ ] **Step 1: Run focused tests**

```powershell
python -m unittest tests.orchestration.test_company_data_intake_validator -v
```

Expected: all tests pass.

- [ ] **Step 2: Run compile check**

```powershell
python -m py_compile scripts/company_data_intake_validator.py tests/orchestration/test_company_data_intake_validator.py
```

Expected: exit code `0`.

- [ ] **Step 3: Generate report**

```powershell
python scripts/company_data_intake_validator.py --use-built-in-synthetic-fixture valid_minimal --output reports/company_data/20260611_company_data_synthetic_schema_report.json
```

Expected stdout: `COMPANY_DATA_SCHEMA_READY`.

- [ ] **Step 4: Check classification**

```powershell
Select-String -Path reports\company_data\20260611_company_data_synthetic_schema_report.json -Pattern '"classification": "COMPANY_DATA_SCHEMA_READY"'
```

Expected: found.

- [ ] **Step 5: Write run report and dashboard links**

Run report must include:

- files changed,
- TDD red/green evidence,
- verification commands and results,
- generated schema report path and SHA256,
- final schema classification,
- reviewer verdict,
- protected no-mutation proof,
- statement that no private company data was read or written.

Dashboard update must add links/status only.

### Task 4: Independent Review And Finalization

**Files:**

- Modify only the Step 4 run report if reviewer verdict needs recording.

- [ ] **Step 1: Spawn read-only reviewer**

Use one `gpt-5.5` reviewer with the verdict options listed above.

- [ ] **Step 2: Apply only requested fixes if needed**

If verdict is `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_NEEDS_FIXES`, apply only requested fixes with tests and re-review.

- [ ] **Step 3: Stop if blocked**

If verdict is `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_BLOCKED`, stop and write the blocked finding in the run report.

- [ ] **Step 4: Final verification**

Re-run focused tests, compile check, classification check, protected hashes/profiles, and process scan.

Final classifications:

- `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_READY`
- `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_NEEDS_FIXES`
- `STEP4_COMPANY_DATA_INTAKE_VALIDATOR_BLOCKED`

