# Step 2 Future Work Plan

Date: 2026-06-11

Plan classification: `STEP2_FUTURE_WORK_PLAN_READY`

## Executive Summary

Step 1 completed the immediate documentation/specification layer for the
current hierarchical v1 production state. Step 2 should turn the safest parts of
that specification into future executable work while keeping model and
production state protected.

Current production remains:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- DQN architecture: `hierarchical_v1`
- initialization: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation/action: `73 / 48`
- status: active registry plus copy-only production promotion complete
- residual watches: accepted for monitoring
- equal-budget residual-watch gate: `PASS`
- Amazon small-sample route proxy: `READY`

Recommended first Step 2 executable goal:

`read-only artifact-health CLI/report`

Why this is first:

- It directly protects the promoted production state.
- It needs no company data, telemetry feed, dataset download, offline eval, or
  model training.
- Step 1 already defined exact expected hashes, paths, registry ids, manifest
  fields, path profiles, process scans, and runtime smoke checks.
- It creates reusable evidence for later monitoring, governance, rollback, and
  approval tasks.

This task does not execute Step 2. It creates the Step 2 future-work plan and a
future goal file:

`docs/goals/20260611_execute_step2_selected_future_work_goal.txt`

Future execution command:

```text
/goal Read docs/goals/20260611_execute_step2_selected_future_work_goal.txt and execute it exactly.
```

## Step 2 Definition

Step 2 is the first layer after Step 1 documentation/specification.

It may include:

- safe-now documentation work,
- safe-now read-only scripts,
- report-generation scaffolding,
- data-intake schemas without data ingestion,
- governance templates,
- future approval packages.

It must separate those from:

- work requiring dataset download approval,
- work requiring company data,
- work requiring a monitoring or calibration trigger,
- model training,
- registry/production/baseline/DB/checkpoint mutation.

## Safe-Now Vs Future-Triggered Distinction

| Tier | Meaning | Examples | Current action |
| --- | --- | --- | --- |
| 1. Safe-now docs/tooling tasks | Can be done from current docs/artifacts, with writes limited to docs/scripts/tests/fresh reports. | artifact-health CLI/report, governance templates | choose one future goal |
| 2. Safe-now read-only scripts | Can read local metadata and protected artifacts without mutating them. | hash/profile/process-scan report | recommended first |
| 3. Requires data approval | Needs public data listing/download approval. | larger Amazon route-proxy sample | plan only |
| 4. Requires company data | Needs private data owner approval and extract. | secondary_fleet/reorder economics validation | plan only |
| 5. Requires model training trigger | Needs monitored regression or company-data mismatch. | 1.5M/2M extension | template only |
| 6. Not recommended now | Adds risk without current evidence. | blind 3M/5M/10M/100M | do not execute |

## Category A: Read-Only Artifact-Health CLI / Report

Purpose:

- Make production-state assurance repeatable.
- Summarize registry active mappings, registry rows, production manifest,
  production hashes, path profiles, runtime smoke, dashboard links, and process
  scan in one read-only report.

Prerequisites:

- `docs/runbooks/20260611_hierarchical_v1_artifact_health_report_spec.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- current registry files and production manifest
- runtime loader and PolicyService tests already exist

Current blockers:

- None for a first read-only CLI/report.

Exact allowed mutations if later executed:

- add a script, recommended:
  `scripts/production_artifact_health_report.py`
- add focused tests, recommended:
  `tests/orchestration/test_production_artifact_health_report.py`
- add a fresh run report:
  `docs/runs/20260611_step2_artifact_health_cli_report.md`
- optionally add a fresh JSON output under a non-protected report path:
  `reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`
- update `docs/00_PROJECT_DASHBOARD.md` links/status only

Forbidden mutations:

- no registry, production, baseline, DB, checkpoint, or existing eval-output
  writes.

Required inputs:

- `models/registry/active_models.json`
- `models/registry/models.jsonl`
- production manifest and production artifacts
- production handoff/final audit docs

Expected outputs:

- deterministic artifact-health CLI,
- focused unit tests,
- fresh health report,
- dashboard link/status update.

Tests/verifications:

- pure parser/hash/path-profile tests,
- dry-run CLI test against temporary fixture files,
- py_compile touched Python files,
- one real read-only report generation against current production,
- protected hashes/path profiles unchanged,
- no train/eval/gate/AWS process.

Reviewer requirement:

- one gpt-5.5 read-only reviewer after implementation.

Stop conditions:

- protected hash drift,
- active registry mismatch,
- production manifest mismatch,
- runtime smoke failure,
- forbidden process found,
- any script attempts a protected write.

Final classification:

- `STEP2_ARTIFACT_HEALTH_CLI_READY`
- `STEP2_ARTIFACT_HEALTH_CLI_NEEDS_FIXES`
- `STEP2_ARTIFACT_HEALTH_CLI_BLOCKED`

Safe to execute now:

- Yes.

## Category B: Monitoring Report Generator

Purpose:

- Convert the Step 1 monitoring report schema into a future report generator
  that can summarize production telemetry, action 24/32 rates, residual
  watches, and alert decisions.

Prerequisites:

- `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
- agreed telemetry input format

Current blockers:

- No live company telemetry extract is present.
- Source field names and denominators are not yet mapped to actual production
  tables.

Exact allowed mutations if later executed:

- script and tests for a schema/template generator only,
- sample synthetic fixture data,
- fresh docs/run report.

Required inputs:

- Step 1 KPI dictionary and report spec,
- synthetic or approved local sample telemetry.

Expected outputs:

- report schema validator,
- template generator,
- sample synthetic report.

Tests/verifications:

- JSON schema validation tests,
- synthetic metrics aggregation tests,
- no protected artifact mutation proof.

Reviewer requirement:

- read-only reviewer after implementation.

Stop conditions:

- requires private data not present,
- unclear denominators,
- any write outside allowed docs/scripts/tests/fresh report paths.

Final classification:

- `STEP2_MONITORING_REPORT_GENERATOR_READY`
- `STEP2_MONITORING_REPORT_GENERATOR_NEEDS_DATA_CONTRACT`
- `STEP2_MONITORING_REPORT_GENERATOR_BLOCKED`

Safe to execute now:

- Partially. A skeleton/template is safe, but full generator value depends on a
  telemetry input contract.

## Category C: Company-Data Intake Validation Package

Purpose:

- Prepare future company-data intake without ingesting private data.
- Validate schemas, required fields, join keys, timestamp semantics, and privacy
  assumptions before data arrives.

Prerequisites:

- `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`
- data owner agreement on extract format

Current blockers:

- No company data extract exists.
- No approved data access path exists.

Exact allowed mutations if later executed:

- schema files,
- validation docs,
- synthetic fixture tests,
- no private data files.

Required inputs:

- company data request package,
- synthetic schemas or mocked CSV headers.

Expected outputs:

- schema validator skeleton,
- required-field checklist,
- privacy/access checklist.

Tests/verifications:

- synthetic positive/negative schema tests,
- no DB writes,
- no private data read.

Reviewer requirement:

- reviewer required before any real data access.

Stop conditions:

- private data appears without approval,
- join keys cannot be defined,
- privacy/access assumptions unresolved.

Final classification:

- `STEP2_COMPANY_DATA_INTAKE_PACKAGE_READY`
- `STEP2_COMPANY_DATA_INTAKE_REQUIRES_DATA_OWNER`
- `STEP2_COMPANY_DATA_INTAKE_BLOCKED`

Safe to execute now:

- Documentation and synthetic schema validator only. Real validation requires
  company data approval.

## Category D: Bounded Amazon Full-Route Proxy Expansion Plan

Purpose:

- Plan a larger public-route proxy analysis beyond the approved 13-route small
  sample.

Prerequisites:

- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`
- exact S3 listing and size estimate
- user approval for specific files and size

Current blockers:

- Full dataset is about `3.1 GiB`.
- Larger download requires explicit approval.

Exact allowed mutations if later executed:

- fresh approved data directory only,
- fresh analysis output docs,
- script fixes/tests if needed.

Required inputs:

- approved file list,
- size estimate,
- local data target.

Expected outputs:

- expanded route-side proxy statistics,
- stress-bucket concentration analysis,
- route-side limits clearly documented.

Tests/verifications:

- file size/hash verification,
- no protected artifact mutation,
- no training/eval/gate.

Reviewer requirement:

- reviewer after analysis.

Stop conditions:

- no explicit download approval,
- expected size exceeds approved budget,
- files already exist in target dir,
- public data is used to claim fleet/reorder validation.

Final classification:

- `STEP2_AMAZON_EXPANSION_PLAN_READY`
- `STEP2_AMAZON_EXPANSION_REQUIRES_DOWNLOAD_APPROVAL`
- `STEP2_AMAZON_EXPANSION_BLOCKED`

Safe to execute now:

- Plan only. Download/analysis requires approval.

## Category E: Productization Plan

Purpose:

- Plan serving health-check API, operator dashboard, and model metadata endpoint
  without touching production serving state.

Prerequisites:

- artifact-health CLI/report,
- monitoring report spec,
- existing PolicyService/runtime smoke tests.

Current blockers:

- Need product/API ownership decisions.
- Need deployment target and authentication model.

Exact allowed mutations if later executed:

- docs and non-production prototype code only,
- tests with fixtures,
- no production deployment mutation.

Required inputs:

- runtime service paths,
- artifact-health report shape,
- monitoring report shape.

Expected outputs:

- API plan,
- endpoint contracts,
- dashboard wire/data spec,
- health-check test plan.

Tests/verifications:

- contract tests,
- local-only smoke tests,
- no active registry or production mutation.

Reviewer requirement:

- reviewer before any runtime patch or deployment integration.

Stop conditions:

- requires production deployment access,
- authentication/authorization undefined,
- endpoint would expose sensitive model or customer data.

Final classification:

- `STEP2_PRODUCTIZATION_PLAN_READY`
- `STEP2_PRODUCTIZATION_NEEDS_OWNER_DECISION`
- `STEP2_PRODUCTIZATION_BLOCKED`

Safe to execute now:

- Plan only. Implementation needs product/runtime approval.

## Category F: Governance Automation

Purpose:

- Turn lifecycle governance into reusable approval templates, rollback drill
  templates, and protected-hash audit bundles.

Prerequisites:

- `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`
- current production truth sources

Current blockers:

- None for docs/templates.
- Automation that writes registry/production remains forbidden without separate
  approval.

Exact allowed mutations if later executed:

- docs/templates,
- read-only audit scripts,
- tests with fixture registries/manifests.

Required inputs:

- governance runbook,
- artifact-health report spec.

Expected outputs:

- approval request templates,
- rollback drill templates,
- protected-hash audit checklist/bundle.

Tests/verifications:

- template completeness checks,
- no protected writes,
- reviewer signoff.

Reviewer requirement:

- reviewer for broad governance package.

Stop conditions:

- template implies automatic protected mutation,
- rollback template omits explicit approval requirement.

Final classification:

- `STEP2_GOVERNANCE_AUTOMATION_READY`
- `STEP2_GOVERNANCE_AUTOMATION_NEEDS_FIXES`
- `STEP2_GOVERNANCE_AUTOMATION_BLOCKED`

Safe to execute now:

- Yes for docs/templates. Lower immediate value than artifact-health CLI.

## Category G: Future Trigger-Based 1.5M/2M Extension Plan Template

Purpose:

- Prepare a template for future bounded training only if monitoring or company
  data produces a concrete trigger.

Prerequisites:

- monitored regression or company-data mismatch,
- exact-resume parent proof,
- residual-watch gate design,
- explicit user approval.

Current blockers:

- No current trigger.
- Residual watches are accepted after equal-budget assurance.

Exact allowed mutations if later executed:

- plan/template docs only unless a future approval explicitly allows config or
  training.

Required inputs:

- monitoring evidence,
- company-data evidence,
- exact-resume checkpoint metadata.

Expected outputs:

- trigger-based ladder template,
- stop conditions,
- no-blind-training policy restatement.

Tests/verifications:

- no config creation,
- no training,
- protected no-mutation proof.

Reviewer requirement:

- reviewer before any config/training.

Stop conditions:

- no concrete trigger,
- user asks for blind scale-up,
- protected state drift.

Final classification:

- `STEP2_TRIGGERED_EXTENSION_TEMPLATE_READY`
- `STEP2_EXTENSION_REQUIRES_TRIGGER`
- `STEP2_EXTENSION_BLOCKED`

Safe to execute now:

- Template only. Not recommended as first Step 2 because no trigger exists.

## Category H: Baseline-Update Decision Memo Template

Purpose:

- Prepare a future decision memo template for possible baseline update after
  sustained production evidence.

Prerequisites:

- stable monitoring windows,
- business approval,
- explicit baseline update request.

Current blockers:

- No monitoring history yet.
- Baseline update is a separate high-risk approval domain.

Exact allowed mutations if later executed:

- memo template only,
- no baseline files.

Required inputs:

- monitoring report history,
- business acceptance criteria,
- rollback requirements.

Expected outputs:

- baseline decision memo template,
- approval checklist,
- non-goal reminders.

Tests/verifications:

- no baseline mutation,
- no registry/production mutation,
- reviewer signoff.

Reviewer requirement:

- reviewer before any real baseline update.

Stop conditions:

- no monitoring evidence,
- no explicit approval,
- proposed baseline write bundled with unrelated changes.

Final classification:

- `STEP2_BASELINE_DECISION_TEMPLATE_READY`
- `STEP2_BASELINE_DECISION_REQUIRES_EVIDENCE`
- `STEP2_BASELINE_DECISION_BLOCKED`

Safe to execute now:

- Template only. Not recommended as first Step 2.

## Prioritized Step 2 Future Tasks

### 1. Safe-Now Docs/Tooling Tasks

1. Artifact-health CLI/report implementation.
2. Governance templates and rollback dry-run docs.
3. Company-data intake schema/checklist using synthetic headers only.

### 2. Safe-Now Read-Only Scripts

1. Artifact-health CLI/report.
2. Synthetic monitoring report generator skeleton.
3. Protected-hash audit bundle.

### 3. Requires Data Approval

1. Bounded Amazon route-proxy expansion.
2. Any download beyond the already approved small sample.

### 4. Requires Company Data

1. Secondary-fleet economics validation.
2. Reorder-none validation.
3. Real dispatch failure taxonomy mapping.
4. Real monitoring report generation from production telemetry.

### 5. Requires Model Training Trigger

1. Any 1.5M/2M extension.
2. Any hierarchical_v2 implementation.
3. Any new reward/config ladder.

### 6. Not Recommended Now

1. Blind 3M/5M/10M/100M.
2. Baseline update without monitoring history.
3. Production or registry mutation bundled with tooling.
4. Public-route-only claims about secondary-fleet or reorder economics.

## Recommended First Step 2 Executable Goal

Recommended goal:

`Implement a read-only hierarchical v1 artifact-health CLI/report.`

Evidence-based rationale:

- Step 1 artifact-health spec is complete and current.
- Current production protection is the highest-value near-term task.
- The CLI can operate entirely on local metadata and protected hashes.
- It can produce a fresh report without mutating registry, production,
  baselines, DB, checkpoints, or existing eval outputs.
- Monitoring generator and company-data validator are valuable but depend on
  telemetry/data contracts that are not yet present.
- Training extension templates are lower priority because no trigger exists.

Future execution command:

```text
/goal Read docs/goals/20260611_execute_step2_selected_future_work_goal.txt and execute it exactly.
```

## Non-Goals

- No model training.
- No offline scenario eval.
- No long-run gate.
- No dataset download.
- No registry mutation.
- No `active_models.json` or `models.jsonl` mutation.
- No production mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No existing eval-output edit.
- No training config creation.
- No 1.5M/2M/3M/5M/10M/100M.
- No claim that Amazon public data validates `secondary_fleet` or reorder
  economics.

## Future Training Policy Reminder

No blind 3M/5M/10M/100M is recommended.

Future training is only appropriate after a concrete trigger:

- repeated monitored service/lateness regression,
- action 24/32 concentration paired with operational degradation,
- hard blockers reappearing,
- company-data mismatch in route/fleet/reorder economics,
- route-choice evidence contradicting the current policy under comparable
  stress.

Even then, the next training plan should start with a bounded 1.5M/2M gated
plan, exact resume, fresh output dirs, residual-watch gates, and explicit
approval. This Step 2 plan does not create configs or authorize training.

## Relation To Production Monitoring And Real-World Calibration

Artifact-health work supports production monitoring by proving that the thing
being monitored is still the approved production artifact.

Monitoring report generation should come after artifact-health because:

- monitoring conclusions are unsafe if registry/production identity has drifted,
- action 24/32 interpretations require stable model identity,
- protected hash evidence is needed before incident or warning triage.

Real-world calibration remains company-data dependent:

- Amazon small-sample evidence supports route-side plausibility only.
- `secondary_fleet` and reorder `none` require company fleet, cost, inventory,
  dispatch, and outcome data.

## Exact Command To Execute The Future Goal

```text
/goal Read docs/goals/20260611_execute_step2_selected_future_work_goal.txt and execute it exactly.
```

## Protected No-Mutation Proof

This planning task only writes:

- `docs/plans/20260611_step2_future_work_plan.md`
- `docs/goals/20260611_execute_step2_selected_future_work_goal.txt`
- `docs/00_PROJECT_DASHBOARD.md` links/status only

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

Pre-write process scan:

```text
NO_TRAIN_EVAL_GATE_OR_AWS_DOWNLOAD_PROCESS
```

Post-write verification must confirm these remain unchanged.

## Final Classification

`STEP2_FUTURE_WORK_PLAN_READY`
