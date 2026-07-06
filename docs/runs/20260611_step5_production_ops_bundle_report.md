# Step 5 Production Ops Bundle Report

Date: 2026-06-11

## Final Classification

`STEP5_PRODUCTION_OPS_BUNDLE_READY`

## Objective

Implemented a local-only production ops bundle runner that executes the existing
safe Step 2/3/4 tools into system-temp intermediate JSON files, then writes one
consolidated persistent ops report.

This does not introduce a new validation domain. It does not ingest private
company data.

## Files Written

- `scripts/run_production_ops_bundle.py`
- `tests/orchestration/test_production_ops_bundle.py`
- `reports/ops/20260611_hierarchical_v1_ops_bundle_report.json`
- `docs/runs/20260611_step5_production_ops_bundle_report.md`
- `docs/00_PROJECT_DASHBOARD.md`

## Generated Ops Report

- Path: `reports/ops/20260611_hierarchical_v1_ops_bundle_report.json`
- SHA256: `430D427E20D944B7B7B8FABFB3040E8D053CA0B8CD2DE8FE66AA50738B4BAD51`
- Decision classification: `OPS_BUNDLE_CLEAN`
- Artifact-health classification: `ARTIFACT_HEALTH_CLEAN`
- Monitoring classification: `MONITORING_REPORT_CLEAN`
- Company schema classification: `COMPANY_DATA_SCHEMA_READY`
- Synthetic-only: `true`
- Private data ingested: `false`
- Process scan raw matches: `0`

Top-level schema:

- `artifact_health`
- `bundle_metadata`
- `company_data_schema`
- `decision`
- `monitoring_report`
- `process_scan`
- `protected_state`

## Implementation Summary

`scripts/run_production_ops_bundle.py`:

- runs `scripts/production_artifact_health_report.py` into a system temp dir,
- runs `scripts/monitoring_report_generator.py` using built-in synthetic fixture
  `clean`,
- runs `scripts/company_data_intake_validator.py` using built-in synthetic
  fixture `valid_minimal`,
- parses the temporary Step 2/3/4 JSON outputs,
- deletes temporary intermediates through `TemporaryDirectory`,
- writes one persistent ops JSON to the requested `--output`,
- rejects output paths under protected project directories,
- scans process command lines for train/eval/gate/AWS/private-data patterns,
- applies deterministic fail-closed classification priority.

## Classification Priority

Implemented priority:

1. Missing, unreadable, malformed, or non-clean artifact-health result ->
   `OPS_BUNDLE_PROTECTED_STATE_DRIFT`
2. Monitoring production risk, protected-state drift, blocked data quality, or
   unreadable monitoring JSON -> `OPS_BUNDLE_PRODUCTION_RISK_FOUND`
3. Company schema blocked or unreadable company schema JSON ->
   `OPS_BUNDLE_DATA_SCHEMA_BLOCKED`
4. Warning/investigation-level monitoring or company schema result ->
   `OPS_BUNDLE_WARNINGS_ONLY`
5. All clean/ready -> `OPS_BUNDLE_CLEAN`

The final process scan also escalates train/eval/gate/AWS/private-data process
matches to `OPS_BUNDLE_PROTECTED_STATE_DRIFT`.

## TDD Evidence

Red phase:

```powershell
python -m unittest tests.orchestration.test_production_ops_bundle -v
```

Initial expected failure:

- `ImportError: cannot import name 'run_production_ops_bundle' from 'scripts'`

Regression red phase after real CLI run exposed direct-script import failure:

```powershell
python -m unittest tests.orchestration.test_production_ops_bundle.ProductionOpsBundleTests.test_repo_root_is_added_for_direct_script_runtime_imports -v
```

Expected failure:

- `AttributeError: module 'scripts.run_production_ops_bundle' has no attribute '_ensure_repo_root_on_path'`

Green phase:

```powershell
python -m unittest tests.orchestration.test_production_ops_bundle -v
```

Result:

- `Ran 13 tests`
- `OK`

## Tests Covered

- all-clean bundle -> `OPS_BUNDLE_CLEAN`
- artifact-health drift -> `OPS_BUNDLE_PROTECTED_STATE_DRIFT`
- monitoring production risk -> `OPS_BUNDLE_PRODUCTION_RISK_FOUND`
- company schema blocked -> `OPS_BUNDLE_DATA_SCHEMA_BLOCKED`
- warning/investigation aggregation -> `OPS_BUNDLE_WARNINGS_ONLY`
- malformed artifact-health JSON -> `OPS_BUNDLE_PROTECTED_STATE_DRIFT`
- malformed monitoring JSON -> `OPS_BUNDLE_PRODUCTION_RISK_FOUND`
- malformed company schema JSON -> `OPS_BUNDLE_DATA_SCHEMA_BLOCKED`
- final process scan flags train/AWS/private-data patterns without terminating
  processes
- protected output paths are rejected
- CLI wrapper writes only requested persistent output and removes temp
  intermediates
- protected files are not mutated by the CLI wrapper
- direct-script runtime imports add the repo root to `sys.path`

## Verification

Focused tests:

```powershell
python -m unittest tests.orchestration.test_production_ops_bundle -v
```

Result:

- `Ran 13 tests in 0.072s`
- `OK`

Compile:

```powershell
python -m py_compile scripts/run_production_ops_bundle.py tests/orchestration/test_production_ops_bundle.py
```

Result: exit code `0`.

Bundle generation:

```powershell
python scripts/run_production_ops_bundle.py --output reports/ops/20260611_hierarchical_v1_ops_bundle_report.json
```

Result:

- exit code `0`
- stdout: `OPS_BUNDLE_CLEAN`

Classification check:

```powershell
Select-String -Path reports\ops\20260611_hierarchical_v1_ops_bundle_report.json -Pattern '"classification": "OPS_BUNDLE_CLEAN"'
```

Result: matched the generated ops JSON.

## Protected No-Mutation Proof

Pre/post protected hashes match:

| Path | SHA256 |
| --- | --- |
| `models/registry/active_models.json` | `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A` |
| `models/registry/models.jsonl` | `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` |

Pre/post protected folder profiles match:

| Path | Files | Bytes |
| --- | ---: | ---: |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/production` | `8` | `328265276` |
| `models/eval` | `128` | `388267659` |

Process scan:

- `OK no train/eval/gate/AWS/private-data process patterns`

Temporary intermediates:

- No Step 5 temp intermediate artifact-health, monitoring, or company-schema
  JSON files were found under persistent `reports/**`.

## Forbidden Actions Not Performed

- No training.
- No offline eval.
- No long-run gate.
- No dataset download.
- No registry update.
- No `active_models.json` mutation.
- No `models.jsonl` append.
- No production mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No existing eval-output edit.
- No training config creation.
- No private company-data ingestion.
- No 1.5M/2M/3M/5M/10M/100M run.

## Reviewer

Reviewer verdict:

- `STEP5_PRODUCTION_OPS_BUNDLE_APPROVED`
