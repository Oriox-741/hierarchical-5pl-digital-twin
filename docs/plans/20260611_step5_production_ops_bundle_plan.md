# Step 5 Production Ops Bundle Plan

Date: 2026-06-11

Status: planning package only. This document does not authorize implementation,
training, offline eval, long-run gates, dataset download, private company-data
ingestion, registry mutation, production mutation, baseline mutation, DB
mutation, checkpoint mutation, existing eval-output edits, or training config
creation.

## Executive Summary

Step 5 should build a local-only production-ops bundle runner that executes the
already-safe read-only tools from Steps 2, 3, and 4, then writes one consolidated
ops summary JSON.

The runner is a convenience wrapper and summary aggregator. It must not introduce
a new validation domain, ingest private data, or modify protected model
artifacts. Its only persistent runtime output should be the requested ops bundle
report.

## Why Step 5 Follows Step 2/3/4

Step 2 established a read-only artifact-health source of truth:

- `scripts/production_artifact_health_report.py`
- `reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`
- Current classification: `ARTIFACT_HEALTH_CLEAN`

Step 3 established a synthetic/local-fixture monitoring report generator:

- `scripts/monitoring_report_generator.py`
- `reports/monitoring/20260611_hierarchical_v1_synthetic_monitoring_report.json`
- Current classification: `MONITORING_REPORT_CLEAN`

Step 4 established a synthetic-only company-data schema validator:

- `scripts/company_data_intake_validator.py`
- `reports/company_data/20260611_company_data_synthetic_schema_report.json`
- Current classification: `COMPANY_DATA_SCHEMA_READY`

Step 5 should compose those outputs into an operator-friendly report after the
individual safe tools exist and have test coverage. It should not reimplement
their domain logic.

## Inputs

The future runner should use these inputs:

- Artifact health CLI/report:
  - CLI: `scripts/production_artifact_health_report.py`
  - Existing reference report:
    `reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`
- Monitoring report generator:
  - CLI: `scripts/monitoring_report_generator.py`
  - Existing reference report:
    `reports/monitoring/20260611_hierarchical_v1_synthetic_monitoring_report.json`
  - Future runner mode: built-in synthetic fixture `clean`
- Company-data validator synthetic mode:
  - CLI: `scripts/company_data_intake_validator.py`
  - Existing reference report:
    `reports/company_data/20260611_company_data_synthetic_schema_report.json`
  - Future runner mode: built-in synthetic fixture `valid_minimal`

To preserve the single persistent-output rule, the future runner should execute
subtools into a temporary directory, parse their temporary JSON outputs, delete
or discard those intermediates, and write only the requested ops bundle JSON.

## Proposed Future Files

Implementation should be limited to:

- `scripts/run_production_ops_bundle.py`
- `tests/orchestration/test_production_ops_bundle.py`
- `reports/ops/20260611_hierarchical_v1_ops_bundle_report.json`
- `docs/runs/20260611_step5_production_ops_bundle_report.md`

Dashboard updates may add links/status only:

- `docs/00_PROJECT_DASHBOARD.md`

## Bundle Output Schema

The ops bundle JSON should contain exactly these top-level sections:

- `bundle_metadata`
- `artifact_health`
- `monitoring_report`
- `company_data_schema`
- `protected_state`
- `process_scan`
- `decision`

Recommended fields:

### `bundle_metadata`

- `generator_version`
- `generated_at_utc`
- `logical_model_id`
- `runtime_family`
- `dqn_architecture`
- `hierarchical_init_method`
- `synthetic_only`
- `private_data_ingested`
- `training_run`
- `offline_eval_run`
- `long_run_gate_run`

### `artifact_health`

- `source_tool`
- `classification`
- `report_generated_at_utc`
- `active_registry_status`
- `production_manifest_status`
- `runtime_smoke_status`
- `dashboard_consistency_status`

### `monitoring_report`

- `source_tool`
- `classification`
- `synthetic_data`
- `watched_actions`
- `alert_count`
- `residual_watch_count`

### `company_data_schema`

- `source_tool`
- `classification`
- `synthetic_data`
- `table_count`
- `data_quality_check_count`
- `join_key_check_count`
- `warning_count`

### `protected_state`

- `registry_status`
- `production_status`
- `baseline_status`
- `db_status`
- `checkpoint_status`
- `eval_output_status`
- `artifact_health_report_status`

### `process_scan`

- `train_process_detected`
- `offline_eval_process_detected`
- `long_run_gate_process_detected`
- `aws_download_process_detected`
- `private_data_process_detected`
- `raw_matches`

### `decision`

- `classification`
- `reasons`
- `recommended_action`

## Classification Priority

The future implementation must apply this priority order:

1. Missing, unreadable, malformed, or non-clean artifact-health result ->
   `OPS_BUNDLE_PROTECTED_STATE_DRIFT`
2. Monitoring production risk -> `OPS_BUNDLE_PRODUCTION_RISK_FOUND`
3. Company schema blocked -> `OPS_BUNDLE_DATA_SCHEMA_BLOCKED`
4. Any warning-only or investigation-level result -> `OPS_BUNDLE_WARNINGS_ONLY`
5. All clean/ready -> `OPS_BUNDLE_CLEAN`

Suggested mapping:

- Artifact health clean:
  - `ARTIFACT_HEALTH_CLEAN`
- Artifact health non-clean or unreadable:
  - anything other than `ARTIFACT_HEALTH_CLEAN`
  - missing artifact-health report JSON
  - unreadable or malformed artifact-health report JSON
- Monitoring production risk:
  - `MONITORING_REPORT_PRODUCTION_RISK_FOUND`
  - `MONITORING_REPORT_PROTECTED_STATE_DRIFT`
  - unreadable monitoring report JSON
  - `MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY`
- Monitoring warning/investigation:
  - `MONITORING_REPORT_WARNINGS_ONLY`
  - `MONITORING_REPORT_NEEDS_INVESTIGATION`
- Company schema ready:
  - `COMPANY_DATA_SCHEMA_READY`
- Company schema blocked:
  - `COMPANY_DATA_SCHEMA_BLOCKED`
  - unreadable company schema report JSON
- Company warning/investigation:
  - `COMPANY_DATA_SCHEMA_WARNINGS_ONLY`
  - `COMPANY_DATA_SCHEMA_NEEDS_FIXES`

If multiple issues exist, the highest-priority classification wins and all
lower-priority issues should still be listed in `decision.reasons`.

## Tests

The future implementation should use TDD and cover:

- All-clean bundle:
  - artifact health `ARTIFACT_HEALTH_CLEAN`
  - monitoring `MONITORING_REPORT_CLEAN`
  - company schema `COMPANY_DATA_SCHEMA_READY`
  - expected decision `OPS_BUNDLE_CLEAN`
- Artifact-health drift blocks:
  - any artifact-health non-clean classification escalates to
    `OPS_BUNDLE_PROTECTED_STATE_DRIFT`
- Malformed artifact-health report blocks:
  - malformed or unreadable artifact-health JSON escalates to
    `OPS_BUNDLE_PROTECTED_STATE_DRIFT`
- Monitoring production risk escalates:
  - monitoring `MONITORING_REPORT_PRODUCTION_RISK_FOUND` escalates to
    `OPS_BUNDLE_PRODUCTION_RISK_FOUND`
- Company schema blocked escalates:
  - company `COMPANY_DATA_SCHEMA_BLOCKED` escalates to
    `OPS_BUNDLE_DATA_SCHEMA_BLOCKED`
- Warnings-only aggregation:
  - monitoring warnings or company schema warnings produce
    `OPS_BUNDLE_WARNINGS_ONLY`
- CLI writes only requested output:
  - no persistent subtool reports outside the requested ops JSON
  - temporary subtool outputs are created only under a temp directory
  - temporary subtool outputs are cleaned up before completion
- No protected writes:
  - `models/registry/active_models.json` unchanged
  - `models/registry/models.jsonl` unchanged
  - `models/production/**` unchanged
  - `models/baselines/**` unchanged
  - `db/**` unchanged
  - `models/checkpoints/**` unchanged
  - existing `models/eval/**` unchanged

## Verification Commands

Future execution should run focused tests first:

```powershell
python -m unittest tests.orchestration.test_production_ops_bundle -v
```

Compile touched files:

```powershell
python -m py_compile scripts/run_production_ops_bundle.py tests/orchestration/test_production_ops_bundle.py
```

Run the local-only bundle generator:

```powershell
python scripts/run_production_ops_bundle.py --output reports/ops/20260611_hierarchical_v1_ops_bundle_report.json
```

Check the expected clean decision:

```powershell
Select-String -Path reports\ops\20260611_hierarchical_v1_ops_bundle_report.json -Pattern '"classification": "OPS_BUNDLE_CLEAN"'
```

Run protected hash/profile/process checks before and after implementation, and
record the results in the Step 5 run report.

## Independent Review Requirement

After implementation and verification, spawn one `gpt-5.5` read-only reviewer.

Reviewer scope:

- Read the Step 5 plan and goal.
- Read the implemented bundle runner, tests, generated ops JSON, and run report.
- Read Step 2/3/4 source reports and generated JSONs as needed.
- Verify no training, eval, gate, dataset download, registry mutation,
  production mutation, baseline mutation, DB mutation, checkpoint mutation, eval
  output mutation, training config creation, or private company-data ingestion.

Reviewer verdict must be exactly one line:

- `STEP5_PRODUCTION_OPS_BUNDLE_APPROVED`
- `STEP5_PRODUCTION_OPS_BUNDLE_NEEDS_FIXES`
- `STEP5_PRODUCTION_OPS_BUNDLE_BLOCKED`

## Non-Goals

- No live telemetry ingestion.
- No private company data ingestion.
- No production mutation.
- No registry mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No existing eval-output edits.
- No training.
- No offline eval.
- No long-run gate.
- No dataset download.
- No new validation domain beyond aggregating Steps 2, 3, and 4.
- No baseline update or production promotion.

## Protected No-Mutation Proof

The future run report should include:

- Pre/post SHA256 for:
  - `models/registry/active_models.json`
  - `models/registry/models.jsonl`
  - `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
  - `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`
- Pre/post file-count and byte-count profiles for:
  - `models/baselines`
  - `db`
  - `models/checkpoints`
  - `models/production`
  - `models/eval`
- Process scan proving no train/eval/gate/AWS/private-data process.
- Confirmation that the bundle runner wrote only:
  - `reports/ops/20260611_hierarchical_v1_ops_bundle_report.json`
  - `docs/runs/20260611_step5_production_ops_bundle_report.md`
  - implementation/test files explicitly allowed by the future goal.

## Exact Goal Command

```text
/goal Read docs/goals/20260611_execute_step5_production_ops_bundle_goal.txt and execute it exactly.
```
