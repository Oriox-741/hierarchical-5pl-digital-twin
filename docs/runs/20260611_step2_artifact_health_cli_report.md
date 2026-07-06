# Step 2 Artifact Health CLI Report

Date: 2026-06-11

Final classification: `STEP2_ARTIFACT_HEALTH_CLI_READY`

## Objective

Implement and run a read-only hierarchical v1 production artifact-health CLI/report.

The CLI verifies current active registry pointers, registry history rows,
production manifest fields, production artifact hashes, protected path profiles,
dashboard consistency, forbidden process state, and a CPU runtime smoke check
for the promoted hierarchical v1 1M production checkpoint.

## Files Written

- `scripts/production_artifact_health_report.py`
- `tests/orchestration/test_production_artifact_health_report.py`
- `reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`
- `docs/runs/20260611_step2_artifact_health_cli_report.md`
- `docs/00_PROJECT_DASHBOARD.md`

No other writes are authorized for this Step 2 task.

## CLI Output

- Command:
  `python scripts/production_artifact_health_report.py --output reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`
- JSON output:
  `reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`
- JSON SHA256:
  `11F21A7FA72C1C3BA4FB62FBE5FF208B24C63224882943AAF2072B1B8CC4F1AF`
- Artifact-health classification:
  `ARTIFACT_HEALTH_CLEAN`

## Report Findings

- Active registry check: `PASS`
- `models.jsonl` candidate/active rows: `PASS`
- Production manifest check: `PASS`
- Production file hash check: `PASS`
- Protected path profile check: `PASS`
- Dashboard consistency check: `PASS`
- Process scan: `PASS`
- Runtime smoke: `PASS`

Runtime smoke loaded
`models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
on CPU and verified:

- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `global_step=1000000`
- `resume_state.exact_resume_capable=true`
- deterministic zero-observation continuous action length `5`
- continuous action finite and bounded
- discrete action valid in `0..47`

## TDD And Fixes

Initial RED:

- `python -m unittest tests.orchestration.test_production_artifact_health_report -v`
- Expected failure: missing `scripts.production_artifact_health_report` module.

GREEN implementation added:

- pure check helpers for registry, `models.jsonl`, manifest, production file
  hashes, protected path profiles, dashboard consistency, process scan, runtime
  smoke, and classification priority.
- CLI defaults for the current hierarchical v1 production paths.
- fixture-mode path and expected-value overrides for tests.

Regression RED/GREEN during real CLI run:

- UTF-8 BOM production manifest read was blocked; fixed by BOM-aware JSON reads.
- Direct `python scripts/...` runtime smoke import could not find `src`; fixed by
  inserting repo root into `sys.path` before runtime imports.
- Runtime smoke initially checked only top-level `exact_resume_capable`; fixed to
  resolve `resume_state.exact_resume_capable`, matching the actual checkpoint.

## Verification

- `python -m unittest tests.orchestration.test_production_artifact_health_report -v`
  - Result: `OK`, 11 tests.
- `python -m py_compile scripts/production_artifact_health_report.py tests/orchestration/test_production_artifact_health_report.py`
  - Result: exit `0`.
- `python scripts/production_artifact_health_report.py --output reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`
  - Result: exit `0`, `ARTIFACT_HEALTH_CLEAN`.

## Protected Baseline

Pre-run protected hashes:

- `models/registry/active_models.json`
  `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`
  `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
  `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`
  `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Pre-run protected path profiles:

- `models/baselines files=2692 bytes=7392576274`
- `db files=7 bytes=28330`
- `models/checkpoints files=787 bytes=4920830666`
- `models/production files=8 bytes=328265276`
- `models/eval files=128 bytes=388267659`

Post-run protected hashes:

- `models/registry/active_models.json`
  `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`
  `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
  `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`
  `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Post-run protected path profiles:

- `models/baselines files=2692 bytes=7392576274`
- `db files=7 bytes=28330`
- `models/checkpoints files=787 bytes=4920830666`
- `models/production files=8 bytes=328265276`
- `models/eval files=128 bytes=388267659`

## Process And Mutation Guard

Pre-run process scan:

`NO_TRAIN_EVAL_GATE_OR_AWS_DOWNLOAD_PROCESS`

Post-run process scan:

`NO_TRAIN_EVAL_GATE_OR_AWS_DOWNLOAD_PROCESS`

This task did not run training, offline eval, long-run gates, dataset downloads,
registry updates, production mutation, baseline mutation, DB mutation,
checkpoint mutation, existing eval-output edits, or training config creation.

## Review

Independent reviewer verdict:

`STEP2_ARTIFACT_HEALTH_CLI_APPROVED`
