# CODEX Project Future Strategy Plan

Date: 2026-06-11

Strategic classification: `FULL_PROJECT_CHRONOLOGY_AND_ROADMAP_READY`

## Scope

This plan defines the next strategic tracks for CODEX PROJE after hierarchical
v1 1M became the active and copy-promoted production model.

This plan does not authorize training, offline eval, long-run gate execution,
registry mutation, production mutation, baseline mutation, DB mutation,
checkpoint mutation, existing eval-output mutation, heavy public-data download,
or new training config creation.

Current production:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- obs/action: `73 / 48`
- status: active registry plus copy-only production promotion complete
- residual watches: accepted for monitoring

## Strategy Principles

1. Protect the current production state first.
2. Treat residual watches as monitoring signals unless they pair with real
   operational degradation.
3. Prefer company-data calibration over more simulation training.
4. Do not start blind 3M or larger runs.
5. Keep candidate registration, active registry activation, production copy,
   baseline update, DB mutation, and checkpoint mutation as separate approval
   domains.
6. Use exact resume for any future gated continuation.
7. Require equal-budget comparisons before interpreting small residual deltas.

## Recommended Near-Term Plan

Priority horizon: next 1 to 3 tasks.

1. Run periodic read-only production state audits.
2. Convert the monitoring runbook into a structured telemetry/KPI data
   dictionary and report spec.
3. Draft the company-data request package for route, fleet, inventory, dispatch
   failure, and cost validation.

Non-goals:

- no training,
- no eval/gate reruns,
- no registry/production/baseline/DB/checkpoint writes,
- no baseline update.

## Recommended Medium-Term Plan

Priority horizon: after monitoring and data contracts exist.

1. Implement a read-only artifact health CLI/report.
2. Build action-watch visualizations for action 24/32, top action
   concentration, and failure counters.
3. Expand Amazon public-route proxy analysis only with an explicit bounded data
   approval.
4. Reconcile company-data fields with simulator KPIs.
5. Produce a baseline-update decision memo if production monitoring remains
   stable, but do not update baselines without a separate approval task.

## Recommended Long-Term Plan

Priority horizon: after real-world data closes calibration gaps.

1. Calibrate simulator route, fleet, reorder, and cost parameters against
   company data.
2. Consider `hierarchical_v2` only if monitoring or calibration shows a specific
   failure of `hierarchical_v1`.
3. Consider a gated 1.5M/2M extension only if a concrete production or
   calibration issue requires model adaptation.
4. Productize serving, health checks, artifact reports, and operator dashboards.
5. Formalize governance: approvals, rollback drills, audit trails, data privacy,
   and simulation-to-real limitations.

## Track A: Production Hardening

Priority: P0

Why it matters:

- Hierarchical v1 is now production-promoted. The next risk is accidental
  registry/production drift or unobserved action-quality regression, not lack of
  training.

Prerequisites:

- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- current registry and production manifests

Concrete work:

- periodic read-only production audits,
- active registry and production manifest hash checks,
- runtime smoke automation,
- rollback dry-run documentation,
- dashboard cleanup,
- structured monitoring report spec.

Risk:

- Low technical risk, high operational value.

Complexity:

- Low to medium.

Concrete next Codex task prompt:

```text
USESuperpowers:
- using-superpowers
- executing-plans
- verification-before-completion

Create a read-only hierarchical v1 production artifact health report CLI/spec.
Do not train, eval, gate, mutate registry, production, baselines, DB,
checkpoints, or existing eval outputs. It must summarize active registry,
production manifest hashes, eval verdicts, long-run gate evidence, residual
watches, and process scan requirements.
```

Explicit non-goals:

- no new model configs,
- no baseline update,
- no registry activation,
- no production copy.

Stop conditions:

- protected hash drift,
- active registry mismatch,
- missing production artifact,
- runtime smoke fails,
- any process scan finds training/eval/gate.

## Track B: Real-World Calibration

Priority: P0

Why it matters:

- The main remaining uncertainty is whether action 24/32 route/fleet/reorder
  choices are economically realistic. Public data supports route-side
  plausibility only.

Prerequisites:

- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`
- company data owner approval and schema access

Concrete work:

- company data contract,
- KPI mapping to OTIF/service/lateness/fleet/reorder/cost,
- secondary_fleet economics validation,
- reorder-none validation,
- dispatch failure taxonomy,
- route planned/actual comparison,
- simulator parameter calibration only after evidence.

Risk:

- Medium to high, mostly due to missing or ambiguous company data.

Complexity:

- Medium for data contract, high for calibration.

Concrete next Codex task prompt:

```text
Create a company-data request and KPI mapping package for hierarchical v1
calibration. Include order, route, fleet, inventory, dispatch failure, and cost
fields; map them to service, lateness, action 24/32, secondary_fleet, and
reorder-none validation. Do not train, eval, mutate model artifacts, or download
data.
```

Explicit non-goals:

- no private data ingestion without approval,
- no model retraining,
- no DB mutation,
- no assumption that public data validates fleet/reorder economics.

Stop conditions:

- missing join keys,
- undefined SLA or lateness semantics,
- no fleet cost/availability fields,
- no inventory/reorder ground truth,
- privacy or access restrictions unresolved.

## Track C: Model Research Extension

Priority: P3 until triggered

Why it matters:

- Training should resume only if monitoring or real data proves a specific gap.
  The current production model passed the ladder and equal-budget assurance.

When 1.5M/2M/3M would be justified:

- repeated monitored regression with denominators and action/failure counters,
- company-data mismatch in fleet/reorder economics that cannot be fixed by
  simulation parameter calibration alone,
- route-choice evidence contradicts current route preference under comparable
  stress,
- hard-blocker or runtime issue is found and fixed with tests.

Why blind 3M is not recommended:

- nextgen 300k/1M showed longer training can worsen behavior,
- V2/V2.5 showed continuation can migrate policy pockets,
- residual watches are currently acceptable after equal-budget comparison,
- real-world economics remain unvalidated.

Prerequisites:

- exact-resume parent checkpoint,
- monitoring evidence or company-data evidence,
- explicit gated extension plan,
- residual watch gates,
- rollback plan,
- protected hash proof.

Concrete work:

- design 1.5M/2M gated extension only after trigger,
- use exact `--resume`,
- no `--allow-empty-replay-resume`,
- fresh output dirs,
- long-run gate at each rung,
- no 3M unless 1.5M/2M evidence supports it.

Risk:

- High.

Complexity:

- High.

Concrete next Codex task prompt:

```text
Draft a trigger-based hierarchical v1 1.5M/2M extension plan from current
production evidence. Do not create configs, train, eval, or mutate artifacts.
Require exact resume, residual-watch gates, rollback rules, and stop conditions.
```

Explicit non-goals:

- no blind training,
- no 3M/5M/10M/100M,
- no failed-branch parents,
- no registry or production mutation.

Stop conditions:

- no concrete trigger,
- missing exact-resume proof,
- no equal-budget comparator,
- any protected path drift,
- any gate failure.

## Track D: Architecture Evolution

Priority: P2

Why it matters:

- `hierarchical_v1` solved the first-order flat action-pocket problem, but it
  still concentrates action 24/32. Future architecture should be evidence-led.

Possible directions:

- `hierarchical_v2` with route/fleet/reorder calibration heads,
- action concentration regularization,
- uncertainty estimates or ensemble confidence,
- safe-policy fallback under low confidence,
- off-policy evaluation and policy confidence tooling,
- state-action interpretability reports.

Prerequisites:

- production monitoring evidence,
- company calibration data,
- current hierarchical v1 analysis,
- tests preserving v5/73/48 external API unless a new contract is explicitly
  approved.

Risk:

- High because architecture changes can break a working production model.

Complexity:

- High.

Concrete next Codex task prompt:

```text
Prepare a read-only hierarchical_v2 architecture options memo. Use production
monitoring and calibration evidence only. Do not patch code, create configs,
train, eval, or mutate artifacts. Preserve v5/73/48 unless arguing for a
separate architecture decision.
```

Explicit non-goals:

- no immediate code patch,
- no action-space change without architecture approval,
- no replacement of hierarchical v1 production.

Stop conditions:

- no production or company-data trigger,
- proposed change cannot preserve runtime compatibility,
- no test strategy for checkpoint compatibility.

## Track E: Data And Tooling

Priority: P1

Why it matters:

- The project has strong memory but it is spread across many docs, configs,
  eval dirs, and manifests. Tooling can make future work safer.

Concrete work:

- Obsidian/project memory index,
- artifact index,
- model registry report,
- eval artifact browser,
- action-watch visualization,
- read-only production audit bundle,
- typed JSON schema validation for configs,
- AGENTS.md or project memory instructions,
- dashboard refresh automation.

Prerequisites:

- current dashboard,
- registry/production manifests,
- eval summaries,
- docs/runs chronology.

Risk:

- Low to medium.

Complexity:

- Medium.

Concrete next Codex task prompt:

```text
Create a read-only artifact index for CODEX PROJE. Index docs, configs, eval
summaries, registry rows, production manifests, and current truth sources. Do
not read binary checkpoint bodies and do not mutate protected artifacts.
```

Explicit non-goals:

- no training,
- no artifact cleanup,
- no deletion,
- no registry writes.

Stop conditions:

- indexer tries to parse checkpoint bodies,
- indexer would write into model/eval/registry dirs,
- unclear truth-source precedence.

## Track F: Productization

Priority: P2

Why it matters:

- The project now has a production model and runtime path. Productization makes
  it operable by another engineer or service process.

Concrete work:

- API/service layer,
- model/version metadata endpoint,
- policy serving health checks,
- integration tests,
- CI-like local test bundle,
- operator dashboard,
- SLA/alert definitions,
- handoff documentation for another engineer.

Prerequisites:

- stable production model,
- runtime smoke behavior,
- monitoring metric definitions,
- registry artifact health report.

Risk:

- Medium.

Complexity:

- Medium to high.

Concrete next Codex task prompt:

```text
Design a production serving health-check API for the current torch_joint
hierarchical model. Include model id, registry ids, contract, obs/action, hash
summary, runtime smoke, and SB3-loader guard status. Do not mutate registry,
production, baselines, DB, checkpoints, or eval outputs.
```

Explicit non-goals:

- no live production deployment without approval,
- no model mutation,
- no baseline update,
- no DB schema change unless separately approved.

Stop conditions:

- health check requires DB writes,
- runtime path cannot be verified without unsafe mutation,
- metadata endpoint would expose sensitive data.

## Track G: Safety And Governance

Priority: P0

Why it matters:

- Most high-impact failures in this project are not coding errors alone; they
  are process errors: wrong parent, unsafe resume, wrong artifact, premature
  registry/production writes, or over-trusting a narrow metric.

Concrete work:

- no-registry/no-production mutation rules,
- explicit approval gates,
- rollback drills,
- audit trails,
- protected hash checks,
- data privacy policy for company data,
- simulation-to-real limitation statements,
- read-only vs write task templates.

Prerequisites:

- production handoff,
- monitoring runbook,
- registry lifecycle docs,
- rollback paths.

Risk:

- Low technical risk, high operational protection.

Complexity:

- Low to medium.

Concrete next Codex task prompt:

```text
Create a governance and approval-gate runbook for CODEX PROJE model lifecycle
tasks. Separate read-only audit, candidate registration, active activation,
production copy, baseline update, DB mutation, eval, and training permissions.
Do not mutate protected artifacts.
```

Explicit non-goals:

- no approval bypass,
- no automatic activation,
- no production cleanup,
- no baseline promotion.

Stop conditions:

- any ambiguity about allowed mutation scope,
- missing rollback evidence,
- protected hashes drift.

## Trigger-Based Training Policy

Training is allowed only under a new explicit task and only if one of these
triggers exists:

- monitored service/lateness regression crosses the monitoring threshold,
- action 24/32 concentration rises materially and pairs with operational
  degradation,
- no-current/no-unassigned/failed-noop appears on action 24 or 32,
- route failure or no-vehicle grows materially with business impact,
- company data proves secondary_fleet or reorder-none economics are wrong,
- a tested code fix changes behavior and requires a new gated candidate.

Required controls before any training:

- read current dashboard and production handoff,
- verify protected hashes,
- exact-resume parent proof,
- fresh output dirs,
- tests before training,
- no failed parent checkpoints,
- no `--allow-empty-replay-resume`,
- offline eval and long-run gate at each rung,
- stop on any gate FAIL,
- do not register/promote without separate approval.

## Real-World Data Strategy

Phase 1: data contract

- define order, route, fleet, inventory, dispatch failure, and cost fields,
- define join keys and timestamp semantics,
- map to service, lateness, dispatch success, route failure, no vehicle,
  secondary_fleet rate, reorder-none rate, stockout, backlog, and cost KPIs.

Phase 2: route-side validation

- extend Amazon Last Mile analysis only with bounded approval,
- compare actual route sequences against shortest/fastest/reliability proxies,
- bucket by time-window stress and travel-time asymmetry.

Phase 3: company economics

- validate secondary fleet cost/availability/acceptance,
- validate reorder-none against stockout/backlog/emergency reorder behavior,
- calibrate simulation parameters only after evidence.

Phase 4: model decision

- if current production is aligned, keep monitoring;
- if data contradicts policy behavior, design a gated calibration or extension
  plan.

## Productization Path

Recommended build order:

1. Read-only artifact health CLI.
2. Monitoring report generator.
3. Serving health-check endpoint.
4. Operator dashboard.
5. CI-like local test bundle.
6. Registry and production lifecycle runbooks.
7. Handoff documentation for another engineer.

## Governance / Rollback Path

Governance rules:

- Every write task must list allowed mutation scope.
- Registry candidate, active activation, production copy, baseline update, DB
  mutation, checkpoint mutation, and eval/training are separate approvals.
- Every model operation must record protected hashes before and after.
- Read-only tasks may write only their requested docs.

Rollback:

1. Restore active registry to the prior balanced-retention active pair only
   under explicit approval or approved incident policy.
2. Preserve registry rows for audit.
3. Quarantine production dir only under explicit approval.
4. Do not mutate baselines or DB as part of rollback.
5. Run read-only production/registry/runtime audit after rollback.

## Recommended Next 10 Tasks

1. Read-only production refresh audit.
2. Monitoring telemetry/KPI data dictionary.
3. Company-data request package.
4. Read-only monitoring report spec.
5. Artifact health CLI/spec.
6. Bounded Amazon route-proxy expansion plan.
7. Amazon stress-bucket route proxy analysis after approval.
8. Baseline-update decision memo template.
9. Rollback/audit dry-run tests.
10. Trigger-based 1.5M/2M extension plan only if evidence appears.

## Final Classification

`FULL_PROJECT_CHRONOLOGY_AND_ROADMAP_READY`
