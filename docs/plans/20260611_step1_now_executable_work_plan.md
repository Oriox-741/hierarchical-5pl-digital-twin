# Step 1 Now-Executable Work Plan

Date: 2026-06-11

Plan classification: `STEP1_NOW_EXECUTABLE_WORK_PLAN_READY`

## Executive Summary

Step 1 is the next safe work package for CODEX PROJE after hierarchical v1 1M
became active and copy-promoted production.

The current production model is already in place:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- DQN architecture: `hierarchical_v1`
- initialization: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation/action: `73 / 48`
- status: active registry plus copy-only production promotion complete
- residual watches: accepted for monitoring
- equal-budget residual-watch gate: `PASS`
- Amazon Last Mile small-sample route proxy status: `READY`

The useful work now is not more model search. It is operational hardening,
monitoring definitions, company-data readiness, artifact-health reporting
specification, real-world calibration boundaries, project memory indexing, and
governance. These can be done immediately from existing files and artifacts
without training, offline evaluation, registry mutation, production mutation,
baseline mutation, DB mutation, checkpoint mutation, or new dataset download.

This plan also creates the executable goal file:

`docs/goals/20260611_execute_step1_now_work_goal.txt`

The goal can later be run with:

```text
/goal Read docs/goals/20260611_execute_step1_now_work_goal.txt and execute it exactly.
```

## What Step 1 Is

Step 1 is a documentation, specification, and readiness package. It turns the
current production handoff, monitoring runbook, residual-watch assurance,
public-route proxy work, and future strategy roadmap into concrete operating
artifacts that a later task can execute safely.

Step 1 is useful now because it:

- reduces production drift risk,
- makes action 24/32 monitoring measurable,
- prepares company-data validation before private data arrives,
- clarifies what public route data can and cannot prove,
- creates an artifact-health reporting contract,
- records lifecycle approvals and forbidden mutations,
- keeps future training trigger-based rather than speculative.

## What Step 1 Is Not

Step 1 is not model experimentation.

It does not authorize:

- model training,
- project offline evaluation,
- long-run gate execution,
- dataset download,
- registry update,
- `active_models.json` or `models.jsonl` mutation,
- production mutation,
- baseline mutation,
- DB mutation,
- checkpoint mutation,
- editing existing eval outputs,
- training config creation,
- 1.5M/2M/3M/5M/10M/100M execution.

## Do Not Train / Do Not Eval / Do Not Mutate Protected Artifacts

The Step 1 execution goal must preserve these boundaries:

- Do not train.
- Do not run offline eval.
- Do not run long-run gate.
- Do not download datasets.
- Do not update registry.
- Do not mutate `models/registry/active_models.json`.
- Do not mutate `models/registry/models.jsonl`.
- Do not mutate `models/production/**`.
- Do not mutate `models/baselines/**`.
- Do not mutate `db/**`.
- Do not mutate `models/checkpoints/**`.
- Do not edit existing `models/eval/**` outputs.
- Do not create training configs.
- Do not start 1.5M/2M/3M/5M/10M/100M.

Allowed Step 1 execution writes are limited to the deliverable docs listed in
the executable goal, plus a dashboard link/status update.

## Prioritized Task List

### STEP1-A: Monitoring KPI Dictionary

Task id: `STEP1-A`

Objective:

Create `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`.

Why it matters now:

The hierarchical v1 model is production-promoted and has accepted residual
watches. The project needs precise metric definitions before monitoring or
company data comparisons begin.

Input files:

- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`

Allowed output:

- `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`

Exact allowed mutation scope:

- Add the one runbook file above.
- No other mutation for this task.

Tests/verifications:

- Verify the file exists.
- Verify sections for service, lateness, dispatch success, no-current,
  no-unassigned, failed-noop, route failure, no-vehicle, already-assigned, top
  action concentration, action 24, action 32, secondary-fleet rate, and
  reorder-none rate.
- Verify each metric includes name, definition, numerator, denominator,
  source, aggregation grain, warning threshold, escalation threshold, and known
  caveats.

Reviewer requirement:

- Included in the final independent read-only Step 1 package review.

Stop condition:

- Stop if the source docs conflict on equal-budget baselines or action decode.

Expected final classification:

- `STEP1_MONITORING_KPI_DICTIONARY_READY`

### STEP1-B: Monitoring Report Specification

Task id: `STEP1-B`

Objective:

Create `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`.

Why it matters now:

The project has monitoring thresholds, but not a normalized report contract.
This spec should make future reporting deterministic and reviewable.

Input files:

- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- output of `STEP1-A`

Allowed output:

- `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`

Exact allowed mutation scope:

- Add the one runbook file above.
- No report generator code unless separately approved.

Tests/verifications:

- Verify sections for report purpose, JSON schema, required metadata,
  per-scenario metrics, per-action metrics, residual-watch block, protected
  state block, alert levels, and final classification.
- Verify the JSON schema includes model id, production path, registry ids,
  observation/action contract, monitoring window, denominators, and warning
  decisions.

Reviewer requirement:

- Included in the final independent read-only Step 1 package review.

Stop condition:

- Stop if the monitoring spec would require reading private company data that
  is not present or approved.

Expected final classification:

- `STEP1_MONITORING_REPORT_SPEC_READY`

### STEP1-C: Company Data Request Package

Task id: `STEP1-C`

Objective:

Create `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`.

Why it matters now:

Company data is required to validate secondary-fleet and reorder economics.
Public Amazon data can only validate route-side proxies. A clear data package
prevents fuzzy requests and later mismatched schemas.

Input files:

- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runs/20260611_public_route_proxy_validation_readiness.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`

Allowed output:

- `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`

Exact allowed mutation scope:

- Add the one runbook file above.
- Do not ingest private data.
- Do not create DB tables.

Tests/verifications:

- Verify sections for order, route, fleet, inventory, cost, dispatch failure,
  privacy/access assumptions, join keys, validation questions, action 24
  mapping, action 32 mapping, and public-data limitations.
- Verify the package explicitly states that company data is required for
  `secondary_fleet` and `reorder` economics.

Reviewer requirement:

- Included in the final independent read-only Step 1 package review.

Stop condition:

- Stop if the package implies access to company data or creates a DB mutation
  path without explicit future approval.

Expected final classification:

- `STEP1_COMPANY_DATA_REQUEST_PACKAGE_READY`

### STEP1-D: Artifact Health Report Specification

Task id: `STEP1-D`

Objective:

Create `docs/runbooks/20260611_hierarchical_v1_artifact_health_report_spec.md`.

Why it matters now:

The project relies on a protected production/registry state. A future
read-only artifact-health report should have one consistent contract for hashes,
manifests, eval verdicts, and process scans.

Input files:

- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `models/registry/active_models.json`
- `models/registry/models.jsonl`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`

Allowed output:

- `docs/runbooks/20260611_hierarchical_v1_artifact_health_report_spec.md`

Exact allowed mutation scope:

- Add the one runbook file above.
- No CLI implementation unless separately approved.
- No registry or production writes.

Tests/verifications:

- Verify sections for active registry checks, candidate/active row checks,
  production manifest checks, file hash checks, runtime smoke checklist,
  protected path profiles, process scan, and failure classifications.
- Verify the spec names current expected protected hashes and production file
  hashes as read-only references.

Reviewer requirement:

- Included in the final independent read-only Step 1 package review.

Stop condition:

- Stop if current registry or production manifest does not match the final
  production handoff.

Expected final classification:

- `STEP1_ARTIFACT_HEALTH_REPORT_SPEC_READY`

### STEP1-E: Model Lifecycle Governance Runbook

Task id: `STEP1-E`

Objective:

Create `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`.

Why it matters now:

The project has successfully separated candidate registration, active registry
activation, production copy, baseline update, DB mutation, checkpoint mutation,
eval, and training. That lifecycle needs to be explicit so future work does not
collapse these approvals into one risky operation.

Input files:

- `docs/runs/20260611_full_project_chronology_and_future_roadmap.md`
- `docs/plans/20260611_codex_project_future_strategy_plan.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`

Allowed output:

- `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`

Exact allowed mutation scope:

- Add the one runbook file above.
- No registry, production, baseline, DB, checkpoint, or eval-output writes.

Tests/verifications:

- Verify sections for approval matrix, allowed mutation templates, forbidden
  operations, rollback path, no-blind-training policy, trigger-based training
  policy, exact-resume requirements, and independent review requirements.
- Verify it does not authorize 3M/5M/10M/100M and does not create training
  configs.

Reviewer requirement:

- Included in the final independent read-only Step 1 package review.

Stop condition:

- Stop if the governance runbook would authorize a protected mutation without a
  separate explicit user approval.

Expected final classification:

- `STEP1_MODEL_LIFECYCLE_GOVERNANCE_READY`

### STEP1-F: Obsidian / Project Memory Index

Task id: `STEP1-F`

Objective:

Embed a project-memory and truth-source index into the Step 1 completion report.
This should map where current truth lives without parsing binary checkpoint
bodies.

Why it matters now:

The project has many historical reports. Future work needs a small current
truth map so old stale reports do not steer execution.

Input files:

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/runs/20260611_full_project_chronology_and_future_roadmap.md`
- `docs/plans/20260611_codex_project_future_strategy_plan.md`
- final Step 1 runbook outputs

Allowed output:

- section inside `docs/runs/20260611_step1_now_work_completion_report.md`

Exact allowed mutation scope:

- Add or edit only the Step 1 completion report.

Tests/verifications:

- Verify the index includes current production truth sources, monitoring
  sources, calibration sources, governance sources, protected artifact sources,
  and obsolete/superseded warning notes.

Reviewer requirement:

- Included in the final independent read-only Step 1 package review.

Stop condition:

- Stop if truth-source conflicts are found that cannot be resolved from current
  docs.

Expected final classification:

- `STEP1_PROJECT_MEMORY_INDEX_READY`

### STEP1-G: Completion Report And Dashboard Update

Task id: `STEP1-G`

Objective:

Create `docs/runs/20260611_step1_now_work_completion_report.md` and update
`docs/00_PROJECT_DASHBOARD.md` with links/status.

Why it matters now:

The Step 1 execution should leave a clear audit trail for what was written,
what was not executed, what protected state was verified, and what is deferred.

Input files:

- all Step 1 deliverables
- `docs/00_PROJECT_DASHBOARD.md`
- current registry and production manifest as read-only evidence

Allowed outputs:

- `docs/runs/20260611_step1_now_work_completion_report.md`
- `docs/00_PROJECT_DASHBOARD.md` link/status update

Exact allowed mutation scope:

- Add the completion report.
- Edit dashboard only to add Step 1 deliverable links/status.

Tests/verifications:

- Verify all Step 1 files exist.
- Verify required sections exist in each deliverable.
- Verify dashboard links if updated.
- Verify protected hashes for registry and production remain unchanged.
- Verify no train/eval/gate/AWS download process is running.

Reviewer requirement:

- Final package must receive independent read-only reviewer verdict:
  `STEP1_NOW_WORK_APPROVED`, `STEP1_NOW_WORK_NEEDS_FIXES`, or
  `STEP1_NOW_WORK_BLOCKED`.

Stop condition:

- Stop if protected hashes drift, unexpected processes are running, or reviewer
  returns needs-fixes/blocked.

Expected final classification:

- `STEP1_NOW_WORK_COMPLETE_SAFE`

## Dependency Order

1. Read current production, monitoring, residual-watch, roadmap, calibration,
   public-route, registry, and manifest evidence.
2. Write `STEP1-A` monitoring KPI dictionary.
3. Write `STEP1-B` monitoring report spec, using the KPI dictionary.
4. Write `STEP1-C` company data request package.
5. Write `STEP1-D` artifact health report spec.
6. Write `STEP1-E` lifecycle governance runbook.
7. Write `STEP1-F` project-memory/truth-source index inside the completion
   report.
8. Write `STEP1-G` completion report and dashboard links.
9. Run verification.
10. Spawn independent read-only reviewer.
11. Apply only reviewer-requested documentation fixes if needed.
12. Re-run verification and re-review if fixes are made.

## Suggested Execution Order

Recommended order for the later `/goal` run:

1. Establish protected hashes and process baseline.
2. Create the monitoring KPI dictionary.
3. Create the monitoring report spec.
4. Create the company-data request package.
5. Create the artifact health report spec.
6. Create the model lifecycle governance runbook.
7. Create the Step 1 completion report with truth-source index.
8. Update dashboard links/status.
9. Verify required files and sections.
10. Verify protected hashes/path profiles/process scan.
11. Run independent read-only review.
12. Finish with one final classification.

## Explicit Non-Goals

- No code patch.
- No training config.
- No model training.
- No offline scenario eval.
- No long-run gate.
- No AWS or dataset download.
- No private data ingestion.
- No registry mutation.
- No active registry switch.
- No production copy or overwrite.
- No baseline update.
- No DB write.
- No checkpoint write.
- No edit to existing eval outputs.
- No 1.5M/2M/3M/5M/10M/100M execution.
- No attempt to validate secondary-fleet or reorder economics with public
  route-only data.

## What Will Be Deferred To Step 2

Step 2 should be chosen only after Step 1 exists. Candidate Step 2 work may
include:

- implementing a read-only artifact-health CLI from the Step 1 spec,
- implementing a monitoring report generator from the Step 1 schema,
- collecting approved company data according to the Step 1 request package,
- running bounded public Amazon expansion only after explicit size/download
  approval,
- drafting a trigger-based 1.5M/2M extension plan only if monitoring or company
  data exposes a concrete reproducible issue.

Step 2 should not default to training.

## Final Recommended Step 1 Execution Goal

Use this command in a later task:

```text
/goal Read docs/goals/20260611_execute_step1_now_work_goal.txt and execute it exactly.
```

The goal file is self-contained and restricts writes to the Step 1 deliverables:

1. `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
2. `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`
3. `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`
4. `docs/runbooks/20260611_hierarchical_v1_artifact_health_report_spec.md`
5. `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`
6. `docs/runs/20260611_step1_now_work_completion_report.md`
7. `docs/00_PROJECT_DASHBOARD.md` links/status only

## Final Classification

`STEP1_NOW_EXECUTABLE_WORK_PLAN_READY`
