# Next-Gen Long-Run Redesign Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement any future tasks from this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the failed next-gen continuation branch and define the gates required before any future long-run curriculum, including a possible 100M-scale run, can be considered.

**Architecture:** Keep the production-promoted balanced-retention model as the active operational baseline. Treat the failed next-gen 1M branch as evidence for redesign, not as a parent for longer runs. Future long runs must be gated by periodic offline evaluation, dispatch-quality telemetry, early stopping, and rollback before extending beyond short stability checkpoints.

**Tech Stack:** PyTorch joint policy checkpoints, `src.eval.evaluate_real_world_scenarios`, `models/registry`, `models/production`, v5 MDP contract `physical_reality_v5_route_candidate_visibility`, 73D observation contract, 48-action discrete dispatch contract.

---

## Current Decision

Recommended immediate decision:

`KEEP_PRODUCTION_MODEL_AND_STOP_NEXTGEN`

Final classification:

`NEXTGEN_LONGRUN_REDESIGN_PLAN_READY`

No registry update, production overwrite, baseline update, DB cleanup, checkpoint mutation, eval-output edit, or new training is part of this plan.

## Evidence Summary

Production model:

`joint_torch_v5_balanced_retention_ft_200k_20260608`

Current production state:

- Active and production-promoted.
- Offline evaluation: 8/8 scenarios PASS.
- Global hard blockers: zero.
- Production remains the best operational model.

Production eval output:

`models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios`

Next-gen 1M branch:

`joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609`

Next-gen 1M eval output:

`models/eval/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios`

Observed result:

- 6/8 scenarios PASS.
- Global hard blockers: zero.
- `premium_sla_pressure` failed service and dispatch-success thresholds.
- `route_disruption_congestion` failed dispatch-success threshold.
- Root-cause classification: `NEXTGEN_1M_OVERTRAINED_DISPATCH_AGGRESSION`.

Next-gen 300k intermediate eval output:

`models/eval/joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios`

Observed result:

- 2/8 scenarios PASS.
- Global hard blockers: zero.
- Severe under-serving and lateness across baseline, demand spike, route disruption, lead time, high holding cost, and mixed stress.
- Classification: `OFFLINE_SCENARIO_EVALUATION_FAIL_SCENARIO_THRESHOLD`.

## Why The Next-Gen 1M Branch Is Stopped

The 1M continuation is not a registry or production candidate because it regressed against the active production model on scenario thresholds:

- Production passes 8/8 scenarios; next-gen 1M passes 6/8.
- The failures are not hard-blocker leaks; they are policy-quality regressions.
- Premium SLA regressed from production-grade service to a threshold failure.
- Route disruption reached perfect service but failed dispatch-success quality, indicating excessive failed/no-op dispatch behavior under stress.
- The prior root-cause audit identified late-stage overtrained dispatch aggression, not a missing evaluator metric or registry issue.

Stopping the branch avoids promoting a model that is clean by hard-blocker telemetry but inferior by operational scenario behavior.

## Why The 300k Intermediate Is Not A Candidate

The 300k periodic checkpoint is not a candidate because it is worse than both production and the final 1M branch by pass count:

- Production: 8/8 PASS.
- Next-gen 1M: 6/8 PASS.
- Next-gen 300k: 2/8 PASS.

300k avoids the final 1M over-dispatch pattern, but it does so by under-serving:

- Baseline service and lateness fail.
- Route disruption service and lateness fail.
- Demand spike service and lateness fail.
- Lead time service fails.
- High holding cost service and lateness fail.
- Mixed stress lateness fails.

This means the branch does not have a known good intermediate checkpoint at 300k. It should not be used for registry, production, or baseline updates.

## Why 500k Eval Is Optional Diagnostic Only

Evaluating 500k can help confirm the shape of the failure curve, but it is not required before stopping this branch.

Reasons:

- The branch is already bracketed by two failing points: 300k under-serves, 1M over-dispatches.
- Production already passes 8/8 and remains the operational baseline.
- A 500k pass would still require a full candidate review because the branch has demonstrated instability before and after that point.
- A 500k fail would only reinforce the existing stop decision.

Optional diagnostic use:

- Run a 500k intermediate offline eval only if the user wants curve confirmation.
- Treat the result as diagnostic evidence, not as an automatic registry candidate.
- Do not block the stop decision on 500k.

## Why Longer Runs Must Not Start From This Branch

Do not start 3M, 5M, 8M, 10M, or 100M from `joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609`.

Reasons:

- The branch failed offline evaluation at 1M.
- The 300k intermediate also failed offline evaluation.
- The failure modes differ by time: early under-serving and later over-dispatch aggression.
- Longer continuation is likely to amplify policy drift without addressing the underlying schedule and objective issues.
- Starting larger runs before stability gates would spend time and compute on a branch that has already failed production comparison.

## What 100M Would Require

A 100M-scale run requires a redesign before launch. Minimum requirements:

- Gated periodic offline evaluation.
- Evaluable periodic artifacts at each gate.
- Intermediate joint checkpoints or confirmed evaluator-compatible periodic checkpoint artifacts.
- Anti-overdispatch objective and telemetry.
- Dispatch-success retention metrics.
- No-current-work and no-unassigned action-quality tracking.
- Scenario-level action quality by route, fleet, reorder, and action id.
- Premium SLA behavior telemetry.
- Route disruption action-quality telemetry.
- Mixed-stress route/fleet/reorder telemetry.
- Schedule stability checks.
- Early stopping rules.
- Rollback plan to production.
- Explicit human approval before each horizon extension.

The 100M path should not be a blind continuation from production. It should be a gated curriculum program that proves stability at shorter horizons first.

## Future Long-Run Design

Use this staged design before considering any 100M run:

1. Keep production active.
2. Create a new long-run-ready curriculum only after the schedule and dispatch-quality gates are specified.
3. Start with a short stability branch, not a 100M run.
4. Evaluate at fixed gates:
   - 250k
   - 500k
   - 1M
   - 2M
   - 3M
   - 5M
5. Stop immediately if any scenario threshold fails.
6. Stop immediately if any global hard blocker appears.
7. Stop if premium dispatch success, route disruption dispatch success, or baseline service regresses below production acceptance thresholds.
8. Stop if no-current-work or no-unassigned dispatch exposure becomes a dominant top-action failure reason.
9. Continue only when each gate passes and the human reviewer accepts residual watches.

## Required Future Gates

Every future long-run gate must record and review:

- Scenario pass count.
- Global hard-blocker totals.
- Service level by scenario.
- True lateness pressure by scenario.
- Dispatch success per attempt by scenario.
- Dispatch/hold ratio by scenario.
- Route distribution by successful dispatch, failed dispatch, and hold.
- Fleet distribution.
- Reorder distribution.
- Top DQN action distribution.
- Failed/no-op dispatch by action id.
- No-current-work counts by action id.
- No-unassigned counts by action id.
- Already-assigned exposure by action id.
- No-vehicle exposure by action id.
- Route failures by action id.
- Premium missed useful dispatch rate.
- Premium missed useful reorder rate.
- Route disruption high-resilience-selected-when-not-best count.
- Mixed-stress service, lateness, and route/fleet/reorder behavior.

## Anti-Overdispatch Requirements

Future training or fine-tuning design must explicitly guard against the 1M failure pattern:

- Track dispatch-success retention during training and offline eval.
- Track failed/no-op dispatch actions separately from successful dispatch.
- Track no-current-work and no-unassigned failure reasons by action id.
- Gate on dispatch quality, not just service level.
- Gate on scenario-specific dispatch quality for premium SLA and route disruption.
- Avoid rewarding action labels that do not correspond to useful current work.
- Retain production-grade behavior on baseline, lead time, high holding cost, and mixed stress.

## Schedule Stability Requirements

Future schedules must avoid both observed failure modes:

- Early under-serving, seen in the 300k checkpoint.
- Late over-dispatch aggression, seen in the 1M checkpoint.

The next schedule should include:

- Balanced retention stages that revisit normal, route disruption, premium SLA, demand spike, lead time, high holding cost, vehicle scarcity, and mixed stress.
- Explicit retention checks before and after stress-heavy stages.
- No horizon extension without a passing offline eval gate.
- No branch promotion based on training metrics alone.

## Rollback Requirements

Rollback remains simple because production is already valid:

1. Keep `joint_torch_v5_balanced_retention_ft_200k_20260608` active.
2. Do not change `models/registry/active_models.json` for next-gen.
3. Do not copy next-gen artifacts into `models/production`.
4. Do not overwrite baselines.
5. If a future candidate fails, keep production active and archive the failed branch as non-authoritative.

## Optional Diagnostic

Optional only:

- Evaluate the 500k intermediate if the user wants curve confirmation.
- Expected value: identify whether there was a narrow stable window between 300k under-serving and 1M over-dispatch aggression.
- Decision impact: not required to stop the current next-gen branch.
- Promotion impact: no automatic registry or production action even if 500k passes.

## Non-Goals

This plan does not authorize:

- Registry update.
- Production overwrite.
- Baseline update.
- DB cleanup.
- New training.
- New fine-tuning.
- New offline evaluation.
- Checkpoint mutation.
- Eval-output mutation.
- Starting 3M, 5M, 8M, 10M, or 100M runs.

## Future Implementation Tasks

### Task 1: Freeze Next-Gen Branch Status

**Files:**
- Read: `models/eval/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios/scenario_summary.json`
- Read: `models/eval/joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios/scenario_summary.json`
- Optional future docs update: project handoff files, if the user approves.

- [ ] Confirm next-gen 1M remains classified as `NEXTGEN_1M_OVERTRAINED_DISPATCH_AGGRESSION`.
- [ ] Confirm next-gen 300k remains classified as `OFFLINE_SCENARIO_EVALUATION_FAIL_SCENARIO_THRESHOLD`.
- [ ] Confirm production remains the only operational model.
- [ ] Record that no 3M/5M/8M/10M/100M continuation should start from this branch.

### Task 2: Optional 500k Curve Diagnostic

**Files:**
- Read: `models/checkpoints/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609/ppo_torch_joint_500000.pt`
- Future output only if explicitly approved: a new isolated eval output directory.

- [ ] Only run if the user explicitly asks for curve confirmation.
- [ ] Use the 500k periodic artifact as `--checkpoint`.
- [ ] Write to a fresh output directory only.
- [ ] Compare 500k against production, 300k, and 1M.
- [ ] Treat the result as diagnostic, not registry-ready by default.

### Task 3: Design The Next Stability Curriculum

**Files:**
- Future create or modify only after approval: a new curriculum config.
- Future modify only after approval: focused config tests.

- [ ] Define stop gates before writing the config.
- [ ] Require offline eval at 250k, 500k, 1M, 2M, 3M, and 5M.
- [ ] Include retention checks for normal, premium SLA, route disruption, mixed stress, demand spike, lead time, high holding cost, and vehicle scarcity.
- [ ] Require dispatch-success retention thresholds.
- [ ] Require no-current-work and no-unassigned action-quality reporting.
- [ ] Do not create a 100M config until the 500k and 1M stability gates pass.

### Task 4: Add Long-Run Gate Automation

**Files:**
- Future modify only after approval: evaluation or runbook tooling.
- Future test only after approval: focused eval-gate tests.

- [ ] Build a gate summary that compares candidate, production, and previous gate.
- [ ] Fail the gate on any scenario threshold failure.
- [ ] Fail the gate on any global hard blocker.
- [ ] Warn on production regression even if scenario thresholds pass.
- [ ] Require human acceptance of residual watches before the next horizon.

## Final Recommendation

Keep the active production model:

`joint_torch_v5_balanced_retention_ft_200k_20260608`

Stop the current next-gen branch:

`joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609`

Do not start longer continuations from it. Treat any 500k evaluation as optional diagnostic evidence only.

