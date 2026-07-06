# Step 1 Now-Work Completion Report

Date: 2026-06-11

Final classification: `STEP1_NOW_WORK_COMPLETE_SAFE`

## Executive Summary

Step 1 created the immediate safe documentation/specification package for the
current hierarchical v1 1M production state.

This work did not run training, offline eval, long-run gates, dataset downloads,
registry mutation, production mutation, baseline mutation, DB mutation,
checkpoint mutation, existing eval-output edits, or training config creation.

Post-write verification and independent read-only review are complete. The
reviewer-requested dashboard fix was applied and re-review returned
`STEP1_NOW_WORK_APPROVED`.

## Current Production State

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- architecture: `hierarchical_v1`
- initialization: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- obs/action: `73 / 48`
- active PPO id: `17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- active DQN id: `f87e10d6-479f-44fc-99d1-6925bc9cb346`
- production checkpoint:
  `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- production status: active registry plus copy-only production promotion
  complete
- residual watches: accepted for monitoring
- equal-budget residual-watch gate: `PASS`
- Amazon small-sample route proxy: `READY`

## Files Created

Step 1 deliverables:

1. `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
2. `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`
3. `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`
4. `docs/runbooks/20260611_hierarchical_v1_artifact_health_report_spec.md`
5. `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`
6. `docs/runs/20260611_step1_now_work_completion_report.md`

Dashboard update:

- `docs/00_PROJECT_DASHBOARD.md` should be updated with links/status for the
  Step 1 deliverables after all files exist.

## Step 1 Deliverable Summary

| Deliverable | Purpose | Classification |
| --- | --- | --- |
| Monitoring KPI dictionary | Defines action 24/32 and production monitoring metrics with denominators, thresholds, caveats, and company-data requirements. | `STEP1_MONITORING_KPI_DICTIONARY_READY` |
| Monitoring report spec | Defines a future read-only JSON/report contract for per-scenario, per-action, residual-watch, and protected-state monitoring. | `STEP1_MONITORING_REPORT_SPEC_READY` |
| Company data request package | Defines order, route, fleet, inventory, cost, dispatch failure, join-key, privacy, and validation requirements. | `STEP1_COMPANY_DATA_REQUEST_PACKAGE_READY` |
| Artifact health report spec | Defines read-only registry, manifest, production hash, runtime smoke, protected path, process scan, and dashboard checks. | `STEP1_ARTIFACT_HEALTH_REPORT_SPEC_READY` |
| Model lifecycle governance runbook | Defines approval gates, allowed mutation templates, forbidden operations, rollback path, no-blind-training policy, and trigger-based training policy. | `STEP1_MODEL_LIFECYCLE_GOVERNANCE_READY` |

## Project-Memory / Truth-Source Index

Current production truth:

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `models/registry/active_models.json`
- `models/registry/models.jsonl`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`

Monitoring truth:

- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`

Calibration truth:

- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_public_route_proxy_validation_readiness.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`
- `scripts/real_world_calibration/README.md`
- `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`

Governance truth:

- `docs/plans/20260611_step1_now_executable_work_plan.md`
- `docs/goals/20260611_execute_step1_now_work_goal.txt`
- `docs/runs/20260611_full_project_chronology_and_future_roadmap.md`
- `docs/plans/20260611_codex_project_future_strategy_plan.md`
- `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`
- `docs/runbooks/20260611_hierarchical_v1_artifact_health_report_spec.md`

## Obsolete / Superseded Warning Notes

- Old flat production remains a comparator and rollback reference, not the
  current active production model.
- Nextgen, V2/V2.1/V2.2/V2.3/V2.4/V2.5, and teacher-retention branches are
  historical evidence, not approved parents for new work.
- Old public-route readiness notes that AWS was unavailable are superseded by
  the later small-sample analysis using approved small files.
- Public Amazon data validates route-side plausibility only; it does not close
  secondary-fleet or reorder-economics questions.
- No blind 3M/5M/10M/100M training is recommended or authorized.

## Protected No-Mutation Proof

Pre-write protected hashes:

- `models/registry/active_models.json`:
  `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`:
  `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- production joint checkpoint:
  `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- production manifest:
  `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Pre-write protected path profiles:

| Path | File count | Total bytes |
| --- | ---: | ---: |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/production` | `8` | `328265276` |
| `models/eval` | `128` | `388267659` |

Process scan before writes:

```text
NO_TRAIN_EVAL_GATE_OR_AWS_DOWNLOAD_PROCESS
```

Post-write protected hashes:

- `models/registry/active_models.json`:
  `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`:
  `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- production joint checkpoint:
  `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- production manifest:
  `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Post-write protected path profiles:

| Path | File count | Total bytes |
| --- | ---: | ---: |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/production` | `8` | `328265276` |
| `models/eval` | `128` | `388267659` |

Post-write process scan:

```text
NO_TRAIN_EVAL_GATE_OR_AWS_DOWNLOAD_PROCESS
```

Result: protected registry, production, baseline, DB, checkpoint, and eval
state remained unchanged.

## Verification Commands And Results

Post-write verification completed before independent review:

- all Step 1 files exist,
- required sections exist in each Step 1 deliverable,
- dashboard links exist for all Step 1 deliverables,
- protected hashes match pre-write values,
- protected path profiles match pre-write values,
- no train/eval/gate/AWS download process is running.

Commands run:

```powershell
$expected = @(
  'docs\runbooks\20260611_hierarchical_v1_monitoring_kpi_dictionary.md',
  'docs\runbooks\20260611_hierarchical_v1_monitoring_report_spec.md',
  'docs\runbooks\20260611_hierarchical_v1_company_data_request_package.md',
  'docs\runbooks\20260611_hierarchical_v1_artifact_health_report_spec.md',
  'docs\runbooks\20260611_codex_model_lifecycle_governance_runbook.md',
  'docs\runs\20260611_step1_now_work_completion_report.md',
  'docs\00_PROJECT_DASHBOARD.md'
)
foreach ($path in $expected) {
  if (Test-Path -LiteralPath $path) { "EXISTS $path" } else { "MISSING $path" }
}
```

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath @(
  'models\registry\active_models.json',
  'models\registry\models.jsonl',
  'models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt',
  'models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\production_manifest.json'
)
```

```powershell
$paths = @('models\baselines','db','models\checkpoints','models\production','models\eval')
foreach ($path in $paths) {
  $items = Get-ChildItem -LiteralPath $path -Recurse -File -Force
  $bytes = ($items | Measure-Object -Property Length -Sum).Sum
  "$path files=$($items.Count) bytes=$bytes"
}
```

## Independent Reviewer Verdict

First reviewer verdict:

```text
STEP1_NOW_WORK_NEEDS_FIXES
```

Reviewer required fixes:

- Update `docs/00_PROJECT_DASHBOARD.md` so `Latest Research Direction` no
  longer says the goal is an experimental 1M candidate awaiting
  register/promote. Current truth is active registry plus copy-only production
  promotion complete.
- Update this completion report from pending reviewer state to the actual
  reviewer result and final classification after re-review.

Fix status:

- Dashboard stale state fixed.
- Re-review completed.

Final reviewer verdict:

```text
STEP1_NOW_WORK_APPROVED
```

Reviewer rationale:

- Dashboard no longer states the stale experimental-candidate/register-promote
  goal.
- Step 1 deliverables satisfy the goal boundaries.
- Amazon public data remains scoped to route-side plausibility, with
  secondary-fleet and reorder economics requiring company data.
- No unapproved training, eval, gate, download, or protected mutation is
  implied.
- Protected hashes/path profiles and process scan match the recorded proof.

## Deferred Step 2 Candidates

Step 2 should be selected only after Step 1 is approved. Candidate Step 2 work:

- implement a read-only artifact-health CLI from the spec,
- implement a monitoring report generator from the JSON spec,
- prepare an approved company-data intake checklist,
- request a bounded Amazon full/sample expansion only after explicit size and
  download approval,
- draft a trigger-based 1.5M/2M plan only if monitoring or company data exposes
  a concrete reproducible issue.

Step 2 should not default to training.

## Explicit No-Training / No-Eval / No-Protected-Mutation Statement

This Step 1 execution did not run training, offline eval, long-run gates,
dataset downloads, registry updates, production mutations, baseline mutations,
DB mutations, checkpoint mutations, existing eval-output edits, or training
config creation.

## Final Classification

`STEP1_NOW_WORK_COMPLETE_SAFE`
