# Long-Run Stability Design V2 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign future long-run experiments so a 500k, 1M, 3M, 5M, 10M, or 100M branch cannot proceed unless it preserves production-grade scenario behavior, dispatch quality, hard-blocker cleanliness, and action stability.

**Architecture:** Treat the failed next-gen branch as evidence of a broader long-run stability problem, not just a bad stage list. Keep the production-promoted balanced-retention model active while future work adds dispatch-quality instrumentation, reward/action-quality retention checks, periodic offline gates, and early stopping before any longer horizon is launched.

**Tech Stack:** PyTorch joint policy checkpoints, v5 MDP contract `physical_reality_v5_route_candidate_visibility`, 73D observation contract, 48-action DQN dispatch contract, `src.eval.check_long_run_gate`, existing offline scenario summaries, and future focused `unittest` coverage.

---

## Final Classification

`LONG_RUN_STABILITY_DESIGN_V2_PLAN_READY`

## Executive Answer

Curriculum design is not the sole root cause.

The next-gen branch failed because curriculum/schedule instability exposed a broader long-run stability problem:

- Early branch behavior under-served most scenarios at 300k.
- Final 1M behavior over-dispatched in premium SLA and route disruption.
- Hard blockers stayed zero, so the issue is policy quality and action stability rather than a forbidden reward-credit leak.
- The trainer had no periodic scenario gate, no production comparison gate, no dispatch-success retention gate, and no early stop when DQN action concentration drifted.
- Reward logic contains useful guards, but no-current-work/no-unassigned dispatch attempts can still receive only a small direct no-op penalty when no dispatch work occurs.

Production remains active and should remain the operational model.

## Source Evidence

Read-only evidence inspected:

- `docs/plans/20260609_nextgen_longrun_redesign_plan.md`
- `docs/plans/20260609_long_run_gate_automation_plan.md`
- `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json`
- `models/eval/joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios/scenario_summary.json`
- `models/eval/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios/scenario_summary.json`
- `models/checkpoints/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609/training_metrics.jsonl`
- `configs/training_joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609.json`
- `configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json`
- `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json`
- `src/act/env_5pl.py`
- `src/act/discrete_action_mapper.py`
- `src/eval/long_run_gate.py`
- `src/eval/check_long_run_gate.py`

## Evidence Table

| Model or checkpoint | Offline pass count | Hard blockers | Primary behavior | Key evidence |
| --- | ---: | --- | --- | --- |
| Production balanced-retention 200k | 8/8 PASS | Zero | Best operational baseline | Baseline service 0.938, premium service 0.980, route disruption service 0.929, mixed service 0.951 |
| Next-gen 300k intermediate | 2/8 PASS | Zero | Under-serving and lateness | Baseline service 0.199 with lateness 0.599; route disruption service 0.485 with lateness 0.455; mixed service 0.765 with lateness 0.428 |
| Next-gen final 1M | 6/8 PASS | Zero | Over-dispatch and failed/no-op dispatch quality regression | Premium service 0.914 and dispatch success 0.475; route disruption dispatch success 0.377; route disruption no-current/no-unassigned total 1043 |

## Detailed Scenario Comparison

| Scenario | Production | Next-gen 300k | Next-gen 1M | Interpretation |
| --- | --- | --- | --- | --- |
| `baseline_normal` | PASS, service 0.938, dispatch rate 0.487 | FAIL, service 0.199, lateness 0.599, dispatch rate 0.223 | PASS, service 0.953, dispatch success 0.869, no-op total 118 | 300k under-served; 1M recovered service but introduced failed/no-op dispatch exposure |
| `route_disruption_congestion` | PASS, service 0.929, dispatch success 0.822, no-op total 171 | FAIL, service 0.485, lateness 0.455, dispatch rate 0.278 | FAIL, service 1.000, dispatch success 0.377, no-op total 1043 | Failure flips from under-dispatch to severe dispatch-quality collapse |
| `premium_sla_pressure` | PASS, service 0.980, dispatch success 0.769, no-op total 178 | PASS, service 0.930, dispatch success 1.000 | FAIL, service 0.914, dispatch success 0.475, no-op total 730 | Premium SLA is most sensitive to over-dispatch/no-op drift |
| `mixed_stress` | PASS, service 0.951, dispatch success 0.996, top action 33 at 29.2% | FAIL, service 0.765, lateness 0.428, top action 25 at 44.6% | PASS, service 0.933, dispatch rate 0.977, top action 40 at 70.4% | 1M passes thresholds but shows action concentration watch |
| `demand_spike_volatility` | PASS, service 0.843, lateness 0.177 | FAIL, service 0.155, lateness 0.653 | PASS, service 0.942, dispatch rate 0.940 | Demand spike improves at 1M but by high dispatch intensity |
| `lead_time_volatility` | PASS, service 0.936, dispatch success 1.000 | FAIL, service 0.352 | PASS, service 0.974, dispatch success 0.548, no-op total 524 | Threshold pass masks dispatch-quality regression |
| `high_holding_cost` | PASS, service 0.943 | FAIL, service 0.332, lateness 0.158 | PASS, service 0.916, lateness 0.044 | 1M recovers threshold but not cleanly above production |
| `vehicle_scarcity_capacity_shock` | PASS, service 0.914, dispatch success 1.000 | PASS, service 0.905 | PASS, service 0.982, dispatch success 0.749, no-op total 390 | 1M service improves while dispatch quality degrades |

## Action Drift Evidence

Decoded action IDs of concern:

| Action | Decoded action |
| ---: | --- |
| 24 | dispatch + shortest + secondary_fleet + none |
| 25 | dispatch + shortest + secondary_fleet + conservative |
| 28 | dispatch + shortest + primary_fleet + none |
| 31 | dispatch + shortest + primary_fleet + emergency |
| 33 | dispatch + low_congestion + secondary_fleet + conservative |
| 34 | dispatch + low_congestion + secondary_fleet + aggressive |
| 38 | dispatch + low_congestion + primary_fleet + aggressive |
| 39 | dispatch + low_congestion + primary_fleet + emergency |
| 40 | dispatch + high_resilience + secondary_fleet + none |
| 41 | dispatch + high_resilience + secondary_fleet + conservative |
| 42 | dispatch + high_resilience + secondary_fleet + aggressive |
| 45 | dispatch + high_resilience + primary_fleet + conservative |

Confirmed 1M action-quality failures:

- `route_disruption_congestion`
  - Action 25: 253 attempts, 49 successes, 204 failed/no-op, dispatch success 0.194.
  - Action 28: 148 attempts, 12 successes, 136 failed/no-op, dispatch success 0.081.
  - Action 33: 114 attempts, 3 successes, 111 failed/no-op, dispatch success 0.026.
  - Action 45: 152 attempts, 104 successes, 48 failed/no-op, dispatch success 0.684.
- `premium_sla_pressure`
  - Action 39: 308 attempts, 60 successes, 248 failed/no-op, dispatch success 0.195.
  - Action 25: 107 attempts, 49 successes, 58 failed/no-op, dispatch success 0.458.
  - Action 40: 81 attempts, 39 successes, 42 failed/no-op, dispatch success 0.481.
- `lead_time_volatility`
  - Action 39: 153 attempts, 17 successes, 136 failed/no-op, dispatch success 0.111.
  - Action 40: 187 attempts, 87 successes, 100 failed/no-op, dispatch success 0.465.
  - Action 42: 48 attempts, 22 successes, 26 failed/no-op, dispatch success 0.458.
- `vehicle_scarcity_capacity_shock`
  - Action 40: 458 attempts, 309 successes, 149 failed/no-op, dispatch success 0.675.
  - Action 42: 148 attempts, 102 successes, 46 failed/no-op, dispatch success 0.689.
- `mixed_stress`
  - Action 40 concentration reached 608 of 864 decisions, 70.4%, while the scenario still passed thresholds.

This is a DQN action-drift and action-quality retention problem, not a route-decoding problem.

## Training Metrics Evidence

The next-gen schedule:

1. `normal_v4`: 75k steps
2. `route_disruption`: 100k steps
3. `premium_sla`: 100k steps
4. `demand_spike`: 125k steps
5. `lead_time_delay`: 100k steps
6. `vehicle_scarcity`: 75k steps
7. `high_holding_cost`: 100k steps
8. `mixed_stress`: 125k steps
9. `route_disruption`: 75k steps
10. `demand_spike`: 50k steps
11. `normal_v4`: 75k steps

Observed training metrics:

- First 100k window mean service was about 0.950.
- 200k-300k window mean service was about 0.937.
- 300k-400k window mean service dropped to about 0.856.
- 400k-500k window mean service stayed low at about 0.849.
- 500k-1M windows stayed near 0.853-0.869, rather than recovering to the production eval behavior.
- DQN action entropy declined from about 3.858 in the first 100k window to about 3.367 around 600k-700k.
- Action 25 became a recurring high-share action beginning around the 400k-700k region.
- Final 925k-1M `normal_v4` segment did not restore production-like training service; its last logged service was about 0.848.

The 300k offline failure aligns with the training-service drop after the early demand-spike transition. The 1M offline failure aligns with later deterministic action concentration and dispatch-quality collapse.

## Reward And Guard Evidence

Read-only inspection of `src/act/env_5pl.py` shows:

- `current_dispatch_work_signal` is only true when `dispatch == "dispatch"` and current dispatch work count is positive.
- Dispatch feasibility and progress credits require `dispatched_orders > 0.0`.
- If `dispatch == "dispatch"` and `dispatched_orders <= 0.0`, the direct `unnecessary_dispatch_penalty` is 0.08.
- `infeasible_dispatch_penalty_weight` is 0.04 and is repeat/exposure gated.
- Demand dispatch quality penalties exist, but they are applied inside the useful-dispatch branch where `dispatched_orders > 0.0`.
- Premium primary-fleet credit is blocked when there is no current dispatch work.
- Route candidate scores and route candidate alignment credit require dispatch, useful work, service health, no route failure, no customer revisit, and no blocked/impossible dispatch.
- Action 24/25 shortest-secondary candidate credit is guarded by severe-pressure, route-failure, no-useful-work, impossible-dispatch, no-vehicle, already-assigned, service-degraded, and concentration checks.
- DQN delivery credit is blocked when dispatch has no current work signal.

Interpretation:

- The reward implementation contains important anti-leak guards.
- The 1M failures are not explained by hard-blocker leaks or route credit leakage.
- The no-current-work/no-unassigned no-op penalty path is probably too weak for long-run DQN drift, especially when deterministic policy concentration appears after exploration decay.
- Reward-side changes alone are not sufficient unless they are paired with training/eval gates that catch regressions before the branch continues.

## Root-Cause Matrix

### Confirmed

1. Curriculum/schedule instability
   - 300k failed 6 scenarios by under-serving and lateness.
   - 1M failed a different way, by over-dispatching premium and route disruption.
   - The next-gen schedule has no embedded offline gates between stage groups.

2. Missing periodic eval gates
   - The branch trained to 1M despite 300k being an offline failure.
   - The existing long-run gate checker would reject production-regressed candidates after eval, but the training schedule did not use such gates to stop the branch.

3. Missing dispatch-success retention gates
   - Production route disruption dispatch success was 0.822; next-gen 1M route disruption fell to 0.377.
   - Production premium dispatch success was 0.769; next-gen 1M premium fell to 0.475.
   - These are scenario-threshold failures and production regressions.

4. DQN action drift
   - DQN entropy declines over training.
   - Offline eval shows high action concentration: mixed stress action 40 at 70.4%; route disruption action 25 at 29.3%; premium action 39 at 35.7%.
   - No-current/no-unassigned no-op exposure explodes in several scenarios at 1M.

5. Insufficient early stopping
   - Training metrics showed service degradation around 300k-500k.
   - No stop occurred before the final 1M checkpoint.

6. Hard blockers are not the failure mode
   - Production, next-gen 300k, and next-gen 1M all report zero hard blockers.
   - The failure is operational policy quality.

### Likely

1. No-current-work/no-unassigned penalty weakness
   - The no-dispatch-work branch applies a small 0.08 direct penalty.
   - Demand quality penalties are not clearly applied to dispatch attempts that produce no dispatched orders.
   - 1M no-op totals reached 1043 in route disruption, 730 in premium SLA, 524 in lead time, and 390 in vehicle scarcity.

2. Exploration-decay plus replay/action-value drift
   - DQN exploration final epsilon is 0.05 with exploration fraction 0.25.
   - Action concentration and entropy compression appear during later windows.
   - The eval failures are deterministic and action-family specific.

3. Final-stage retention was too weak
   - The final `normal_v4` stage did not restore production-like service in training metrics.
   - The final model passes baseline offline service but carries no-op dispatch exposure into stress scenarios.

4. Stress-heavy stage interactions caused policy oscillation
   - Early demand/lead-time sections correlate with under-serving.
   - Later route/demand/normal retention correlates with service recovery but dispatch-quality regression.

### Possible

1. Stage-end bias
   - The final checkpoint may overrepresent the late route/demand/normal sequence.
   - This is plausible but not fully proven because training metrics are window summaries and do not emit enough per-step reward attribution to isolate the final-stage mechanism.

2. Reward/action-quality instrumentation gap
   - Eval artifacts contain action failure counts by scenario, but training metrics do not expose the same no-current/no-unassigned action-quality breakdown by stage/window.
   - Per-component DQN reward averages by action family would make the root cause more precise.

3. Objective imbalance between service recovery and dispatch quality
   - 1M improves some service metrics while reducing dispatch success.
   - This suggests the objective lets failed/no-op dispatch exposure rise as long as service stays acceptable in some scenarios.

### Ruled Out

1. Contract or observation mismatch
   - Eval summaries report contract `physical_reality_v5_route_candidate_visibility` and observation dim 73.

2. Hard-blocker reward leak
   - Hard blockers are zero in production, next-gen 300k, and next-gen 1M.

3. Registry or production promotion issue
   - Production remains the active/best model; next-gen was not promoted.

4. Action decoding issue
   - Action IDs decode consistently through `DiscreteActionMapper.map`.

5. Need to run 3M/5M/10M/100M to know more
   - The current branch is already bracketed by failing 300k and failing 1M evaluations.

## Direct Answers To Required Questions

### Is curriculum design the sole root cause?

No. Curriculum design is a confirmed contributor, but the root cause is broader long-run instability: schedule instability, missing dispatch-success retention gates, insufficient early stopping, DQN action drift, and likely weak penalties for no-current-work/no-unassigned dispatch attempts.

### Did next-gen fail due to curriculum/schedule instability?

Yes. The schedule produced early under-serving at 300k and late over-dispatch at 1M.

### Did next-gen fail due to missing dispatch-success retention gates?

Yes. Premium and route dispatch success regressed below production and below scenario thresholds.

### Did next-gen fail due to reward penalty weakness for no-current-work/no-unassigned dispatch?

Likely. The direct no-work dispatch penalty is small and the 1M eval shows large no-current/no-unassigned exposure. This should be patched or at least instrumented before another long run.

### Did next-gen fail due to stage-end bias?

Possible. The final stage sequence did not restore training service to production-like levels and the final deterministic policy shows action concentration. More granular training-stage telemetry is needed before calling this confirmed.

### Did next-gen fail due to DQN action drift?

Yes. Action concentration and no-op action exposure are directly visible in eval artifacts and training entropy trends.

### Did next-gen fail due to insufficient early stopping?

Yes. 300k and 1M failures would have been stopped by periodic eval gates and production comparison.

### Did next-gen fail due to lack of periodic eval gates?

Yes. This is confirmed by the fact that the branch continued after a 300k checkpoint that later evaluated as 2/8 PASS.

### Should v2 be config-only?

No. V2 should include reward/action-quality instrumentation and gate automation usage before any new long-run config. A config-only patch would still leave the system unable to stop action drift early.

### Is optional 500k eval needed before v2?

No. A 500k eval is optional diagnostic curve confirmation only. It is not needed to justify stopping this branch or designing V2 because the branch already has failing checkpoints at 300k and 1M.

### Should production remain active?

Yes. `joint_torch_v5_balanced_retention_ft_200k_20260608` remains active and production-promoted until a future candidate passes all gates.

## Required V2 Components

### 1. Curriculum Schedule V2

Design goals:

- Avoid long uninterrupted stress exposure without retention checks.
- Alternate stress stages with production-retention anchors.
- Revisit `normal_v4`, `premium_sla`, `route_disruption`, and `mixed_stress` frequently.
- Keep `baseline_anchor_probability` above zero unless a test proves it is harmful.
- Do not use failed next-gen 1M as a parent.
- Use production as the rollback baseline, not as a blind continuation launch point.

Required gates before writing a new config:

- Define eval gate checkpoints at 250k, 500k, and 1M.
- Define additional gates at 2M, 3M, and 5M only after 1M passes.
- No 10M or 100M config until the 500k and 1M gates pass.

### 2. Anti-Overdispatch Objective And Telemetry

Future reward/instrumentation work must expose:

- Dispatch attempt count by scenario, stage, and action id.
- Dispatch success count by scenario, stage, and action id.
- Failed/no-op dispatch count by action id.
- No-current-work count by action id.
- No-unassigned count by action id.
- No-vehicle and already-assigned exposure by action id.
- Top action concentration by stage/window.
- Mean no-work dispatch penalty by action id.
- Mean DQN local reward by action id and action family.

Required anti-overdispatch gate:

- Fail if premium or route disruption dispatch success regresses from production by more than 0.005.
- Fail if no-current/no-unassigned totals exceed production-comparison thresholds.
- Fail if a no-current/no-unassigned action becomes a top action with material concentration.

### 3. Dispatch-Success Retention

V2 must retain production dispatch quality:

- `premium_sla_pressure.dispatch_success_per_attempt` must not regress materially below production.
- `route_disruption_congestion.dispatch_success_per_attempt` must not regress materially below production.
- `lead_time_volatility` and `vehicle_scarcity_capacity_shock` dispatch success must be monitored even when scenario thresholds pass.
- Action 25, 28, 33, 39, 40, 42, and 45 must have per-scenario low-success exposure checks.

### 4. No-Current-Work And No-Unassigned Guard

Future implementation should either:

- Strengthen the reward penalty when `dispatch == "dispatch"` and no current dispatch work is available, or
- Add a separate train/eval fatal gate that prevents such actions from accumulating even if scenario thresholds pass.

The preferred V2 posture is both:

- Patch reward/action-quality logic in a focused way.
- Keep the eval gate as an independent safety layer.

### 5. Periodic Eval Gates

Use `src.eval.check_long_run_gate` after each approved offline eval output exists.

Required command shape:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary <candidate_eval_dir>\scenario_summary.json `
  --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json `
  --previous-summary <previous_gate_eval_dir>\scenario_summary.json `
  --candidate-label <candidate_label>
```

Gate policy:

- Exit code 0 is required to continue.
- Exit code 2 stops the branch.
- Warnings require human review before horizon extension.

### 6. Early Stopping

Training should stop or at least pause for evaluation when:

- Rolling training service drops below a production-derived floor.
- DQN entropy compresses while top action concentration rises.
- Dispatch/hold distribution moves abruptly across stage boundaries.
- No-current-work or no-unassigned dispatch counts rise.
- Premium or route disruption windows lose dispatch-success retention.

### 7. Production Comparison

Every gate must compare against production:

- Scenario pass count.
- Service level.
- True lateness pressure.
- Dispatch success per attempt.
- Dispatch/hold ratio.
- Top action concentration.
- No-current/no-unassigned exposure.
- Route, fleet, and reorder distribution.
- Residual watches.

### 8. Gate Checker Usage

The existing read-only gate checker is required for all future long-run candidates. Before any longer branch continues:

- Production self-check must PASS.
- Candidate gate must PASS.
- Previous-gate comparison must not warn on unaccepted regressions.
- Any warning must be reviewed by a human before continuing.

## What Not To Do

Do not:

- Start a blind 100M run.
- Start 3M, 5M, 10M, or 100M from the failed next-gen 1M branch.
- Continue from `joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609`.
- Promote or register next-gen checkpoints.
- Update production.
- Update baselines.
- Mutate DB rows.
- Mutate checkpoint artifacts.
- Treat hard-blocker zero status as sufficient for production readiness.
- Treat a possible 500k pass as automatically registry-ready.

## Proposed Next Implementation Step

Recommended next implementation step:

`INSTRUMENTATION_AND_REWARD_ACTION_QUALITY_PATCH_PLAN_REQUIRED`

The next approved work should create a focused implementation plan for:

1. Training-window telemetry for dispatch success and no-current/no-unassigned action quality.
2. Eval summary additions if any gate-critical field is missing from training or summaries.
3. A focused reward/action-quality patch for no-current-work/no-unassigned dispatch attempts.
4. Unit tests proving the patch does not reintroduce hard-blocker leaks.
5. A new V2 curriculum config only after the telemetry and reward/gate checks are ready.

## Future Implementation Tasks

### Task 1: Add Training Action-Quality Telemetry

**Files:**
- Future modify: `src/learn/train_joint_torch.py`
- Future modify: metrics aggregation helpers used by joint training
- Future test: focused training metrics unit tests

- [ ] Add per-window counts for `action_no_current_work_by_id`.
- [ ] Add per-window counts for `action_no_unassigned_by_id`.
- [ ] Add per-window counts for `action_failed_noop_by_id`.
- [ ] Add per-window `dispatch_success_per_attempt` by stage and action id.
- [ ] Add per-window top-action concentration.
- [ ] Add a focused test that constructs a metrics window with no-current-work action 25 exposure and verifies it appears in the emitted metrics record.
- [ ] Run focused unit tests only.

### Task 2: Patch No-Current/No-Unassigned Dispatch Reward Guard

**Files:**
- Future modify: `src/act/env_5pl.py`
- Future test: `tests/act/test_reward_physics_contract.py`

- [ ] Add a failing test where `dispatch == "dispatch"` and no current work/no unassigned work produces a stronger negative DQN local outcome than the current weak no-op path.
- [ ] Add a failing test proving valid useful dispatch still receives bounded feasibility/progress credit.
- [ ] Implement a focused penalty or gate for no-current-work/no-unassigned dispatch attempts.
- [ ] Verify hard-blocker counters remain zero for valid useful dispatch.
- [ ] Run focused act reward tests and py_compile only.

### Task 3: Extend Gate Checker If Training/Eval Fields Require It

**Files:**
- Future modify: `src/eval/long_run_gate.py`
- Future modify: `tests/eval/test_long_run_gate.py`

- [ ] Add tests for any newly emitted action-quality fields.
- [ ] Preserve CLI stdout-only JSON.
- [ ] Preserve exit 0 PASS and exit 2 FAIL.
- [ ] Verify production self-check remains PASS.
- [ ] Verify next-gen 300k and 1M remain FAIL.

### Task 4: Create V2 Stability Curriculum Config

**Files:**
- Future create: a new config with a unique V2 environment id
- Future test: focused curriculum config tests

- [ ] Use production as parent only if the reward/action-quality patch and telemetry are in place.
- [ ] Include short alternating retention stages.
- [ ] Include nonzero baseline anchoring unless a test proves otherwise.
- [ ] Include explicit output directory that does not overwrite existing artifacts.
- [ ] Require manual run flags for registry skip and trace logging disablement.
- [ ] Add config tests proving no failed next-gen, v3, v4, invalid 700k, or old rewardfix artifacts are referenced.

### Task 5: Gate Future Runs Before Horizon Extension

**Files:**
- Future read: new eval output directories
- Future no writes unless user approves a runbook or report

- [ ] Evaluate the first approved V2 checkpoint only after training is manually launched and completes.
- [ ] Run `src.eval.check_long_run_gate` against production and previous gate.
- [ ] Stop immediately on any FAIL.
- [ ] If PASS with warnings, require human acceptance before any longer horizon.
- [ ] Keep production active until a future candidate completes registry and production review.

## Optional 500k Diagnostic

The 500k intermediate next-gen eval is optional only.

Use it only if the user wants curve confirmation between the 300k under-serving checkpoint and the 1M over-dispatch checkpoint. It is not required before V2 design, and it must not be treated as an automatic candidate even if it passes.

## Non-Goals

This plan does not authorize:

- Training.
- Evaluation.
- Registry update.
- Production update.
- Baseline update.
- DB mutation.
- Checkpoint mutation.
- Config creation.
- Code changes outside a future approved implementation task.
- Starting 500k, 1M, 3M, 5M, 10M, or 100M branches.

## Final Recommendation

Keep production active:

`joint_torch_v5_balanced_retention_ft_200k_20260608`

Stop the failed next-gen branch:

`joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609`

Do not launch another long run until V2 has reward/action-quality instrumentation, dispatch-success retention, no-current/no-unassigned guards, periodic eval gates, early stopping, and production comparison wired into the workflow.

