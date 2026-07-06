# Stability V2.1 Reward Balance and Lateness Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the Stability V2 action-quality gains while restoring useful-dispatch urgency, primary-fleet timing, and service/lateness retention before any longer run.

**Architecture:** Make a bounded reward-balance patch in the environment, add focused timing/action-quality telemetry, and keep the long-run gate strict. V2.1 should reward only useful dispatch under real pressure, never no-work labels, no-work dispatch, or hard-blocker paths.

**Tech Stack:** Python, unittest, PyTorch joint training artifacts, existing 5PL environment reward code, joint training metrics, offline scenario summaries, long-run gate checker.

---

## Current Evidence

Inputs reviewed for this plan:

- `docs/plans/20260609_action_quality_reward_instrumentation_patch_plan.md`
- `docs/plans/20260609_long_run_stability_design_v2_plan.md`
- `docs/plans/20260609_long_run_gate_automation_plan.md`
- `models/eval/joint_torch_v5_prod_stability_v2_250k_20260609_offline_scenarios/scenario_summary.json`
- `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json`
- `models/checkpoints/joint_torch_v5_prod_stability_v2_250k_20260609/training_metrics.jsonl`
- `src/act/env_5pl.py`
- `src/learn/joint_metrics.py`
- `src/eval/long_run_gate.py`

Stability V2 250k result:

- Scenario gate: 4/8 PASS.
- Hard blockers: zero.
- No-current/no-unassigned exposure improved versus next-gen 1M: 2825 to 78.
- Failed-noop exposure improved versus next-gen 1M: 1413 to 39.
- Failed scenarios:
  - `high_holding_cost`: `true_lateness_pressure=0.118`, threshold `0.080`.
  - `lead_time_volatility`: `true_lateness_pressure=0.126`, threshold `0.100`.
  - `premium_sla_pressure`: `service_level=0.912`, threshold `0.920`.
  - `route_disruption_congestion`: `true_lateness_pressure=0.234`, threshold `0.120`.

## What V2 Fixed

- The `no_current_or_unassigned_dispatch_penalty` guard worked: no-current/no-unassigned action exposure dropped sharply.
- Failed/no-op dispatch exposure dropped sharply.
- Premium no-work exploit stayed zero.
- Hard blockers remained zero.
- Dispatch success was mostly clean outside the route-disruption edge.
- Action31 collapse was not the active failure mode.

## What V2 Broke

- High holding cost became late despite clean dispatch attempts.
- Lead time volatility became late despite clean dispatch attempts.
- Premium SLA lost the service edge.
- Route disruption became too late and under-responsive.
- Several failures point to timing and response quality, not hard-blocker leakage.

## Root-Cause Matrix

| Status | Cause | Evidence | V2.1 Response |
| --- | --- | --- | --- |
| Confirmed | No-work guard effective | No-current/no-unassigned fell from 2825 to 78; failed-noop fell from 1413 to 39 | Keep guard unchanged in spirit and covered by tests |
| Confirmed | Under-response or slow timing in failed scenarios | Four failures are lateness/service threshold misses with hard blockers zero | Add bounded useful-dispatch urgency under real pressure |
| Likely | Too much hold under pressure | Route disruption and premium show high hold exposure and missed useful dispatch opportunities | Add telemetry and bounded hold-under-pressure penalty |
| Likely | Too much secondary fleet in timing-sensitive stages | High holding and lead time show very low primary-fleet share versus production | Add useful-work-gated primary-fleet timing justification |
| Likely | Weak urgency reward after V2 guard | V2 reduced bad dispatch but did not sufficiently preserve timing response | Add pressure-gated urgency credit, not global dispatch credit |
| Possible | Route/fleet/reorder timing imbalance | Route disruption and premium failures include route/fleet/reorder watch fields | Add telemetry before broad route/reorder reward changes |
| Possible | Stage-specific curriculum pressure imbalance | V2 250k short gate failed different timing-sensitive stages | Retune V2.1 curriculum after reward tests pass |
| Ruled out | Hard-blocker leak | All global hard blockers were zero | Do not weaken gates |
| Ruled out | Premium no-work exploit recurrence | Premium no-work exploit remained zero | Preserve premium no-work guard |
| Ruled out | Action31 collapse as primary cause | Route failure profile did not show action31 as the collapse driver | Do not target action31 specifically |

## Proposed V2.1 Reward-Balance Patch

Preserve existing behavior:

- Keep `no_current_or_unassigned_dispatch_penalty`.
- Keep premium no-work credit blocks.
- Keep route candidate hard gates.
- Keep hold-route label neutralization.
- Keep observation dimension 73, action count 48, and contract `physical_reality_v5_route_candidate_visibility`.

Add bounded useful-work urgency:

- Add a DQN local component named `useful_dispatch_lateness_urgency_credit`.
- It may be positive only when all are true:
  - `dispatch == "dispatch"`.
  - `current_dispatch_work_signal > 0`.
  - Useful dispatch pressure exists through true lateness, SLA pressure, route disruption pressure, urgent ratio, due-soon pressure, or pending work pressure.
  - The dispatch is not blocked by no-current/no-unassigned exposure.
- Bound the credit so it restores timing without globally encouraging dispatch.
- Suggested shape:

```python
useful_dispatch_pressure = max(
    actionable_lateness_pressure,
    premium_sla_pressure,
    route_disruption_pressure,
    urgent_ratio,
    due_soon,
    pending_work_pressure,
)
useful_dispatch_lateness_urgency_credit = (
    useful_dispatch_allowed
    * min(0.12, 0.04 + 0.08 * useful_dispatch_pressure)
)
```

Add useful-work-gated primary-fleet timing:

- Add a DQN local component named `primary_fleet_timing_justification_credit`.
- It may be positive only when all are true:
  - `dispatch == "dispatch"`.
  - `mode == "primary_fleet"`.
  - `current_dispatch_work_signal > 0`.
  - The scenario has real timing pressure: premium/SLA, true lateness, lead-time volatility, high holding cost, route disruption, urgent orders, or due-soon orders.
  - Dispatch is not a no-current/no-unassigned action.
- This must not reward no-work hold-primary labels.
- This must not penalize valid secondary fleet globally; it should only restore timing-sensitive primary use when justified.

Add secondary-fleet lateness exposure telemetry first:

- Add a telemetry field named `secondary_fleet_lateness_exposure`.
- Consider a small `secondary_fleet_lateness_timing_penalty` only if tests and artifacts show persistent secondary overuse under real lateness pressure.
- If implemented, it must be bounded and active only under real timing pressure with useful dispatch work available.

Add bounded hold-under-pressure handling:

- Add telemetry named `hold_under_lateness_pressure_by_scenario`.
- Consider a small `hold_under_lateness_pressure_penalty` when:
  - `dispatch == "hold"`.
  - Real useful dispatch work exists.
  - True lateness/SLA/route-disruption pressure is present.
- Do not penalize hold in calm/no-work conditions.
- Do not reward route or fleet labels attached to hold.

## Proposed Telemetry Additions

Training-window telemetry:

- `lateness_pressure_by_stage_action`
- `useful_dispatch_opportunity_missed_by_action`
- `primary_fleet_useful_dispatch_by_scenario`
- `hold_under_lateness_pressure_by_scenario`
- `secondary_fleet_lateness_exposure`
- `mean_useful_dispatch_lateness_urgency_credit_by_action_family`
- `mean_primary_fleet_timing_justification_credit_by_action_family`

Telemetry rules:

- Training metrics should aggregate these fields without changing replay, observation shape, or action space.
- Evaluation summaries should only be extended if the same fields can be emitted reliably from episode metrics.
- Do not over-extend `src/eval/long_run_gate.py` with training-only fields that are absent from eval outputs.

## Proposed Tests

Reward physics tests in `tests/act/test_reward_physics_contract.py`:

- No-current dispatch still receives `no_current_or_unassigned_dispatch_penalty`.
- Explicit no-unassigned dispatch still receives `no_current_or_unassigned_dispatch_penalty`.
- Valid useful dispatch under lateness pressure receives bounded `useful_dispatch_lateness_urgency_credit`.
- No-work dispatch receives zero urgency credit.
- Hold under no work receives zero urgency credit and no primary-fleet timing credit.
- Premium no-work guard remains active.
- Premium useful primary-fleet dispatch can receive bounded timing justification.
- Secondary fleet is not globally suppressed in calm or justified contexts.
- Route candidate hard gates remain active.
- Hard-blocker counters remain zero in calm normal-v4 contract tests.

Training metrics tests in `tests/learn/test_train_joint_curriculum.py`:

- `JointMetricsSnapshot.as_json_dict()` emits all new timing/action-quality fields.
- Synthetic transitions aggregate `lateness_pressure_by_stage_action`.
- Synthetic transitions aggregate `useful_dispatch_opportunity_missed_by_action`.
- Synthetic transitions aggregate `primary_fleet_useful_dispatch_by_scenario`.
- Synthetic transitions aggregate `hold_under_lateness_pressure_by_scenario`.
- Synthetic transitions aggregate `secondary_fleet_lateness_exposure`.

Gate tests in `tests/eval/test_long_run_gate.py` only if gate code changes:

- Existing production self-check still passes.
- Next-gen 1M still fails.
- Next-gen 300k still fails.
- V2 250k still fails until the future V2.1 eval proves otherwise.
- Any new warning field is warning-only unless it is backed by eval-summary evidence.

## Proposed V2.1 Curriculum Direction

- Keep a short 250k gate before any 500k or longer branch.
- Increase timing retention for:
  - `lead_time_delay`
  - `high_holding_cost`
  - `route_disruption`
  - `premium_sla`
- Keep a final `normal_v4` retention stage.
- Preserve balanced exposure across all 8 offline scenarios.
- Do not start 500k until V2.1 250k passes scenario thresholds, hard blockers, and long-run gate comparison.
- Do not continue from failed next-gen 1M.

## Implementation Tasks

- [ ] Task A: Add RED tests for reward-balance behavior.
  - Files: `tests/act/test_reward_physics_contract.py`.
  - Assert existing no-current/no-unassigned penalty remains active.
  - Assert useful dispatch under lateness pressure receives bounded urgency credit.
  - Assert no-work dispatch and hold labels receive no urgency/fleet credit.
  - Assert premium no-work guard and route candidate hard gates remain active.

- [ ] Task B: Implement bounded reward-balance components.
  - File: `src/act/env_5pl.py`.
  - Add `useful_dispatch_lateness_urgency_credit`.
  - Add `primary_fleet_timing_justification_credit`.
  - Add `secondary_fleet_lateness_exposure` telemetry.
  - Add `hold_under_lateness_pressure` telemetry, and only add a penalty if tests show the pressure is otherwise invisible.
  - Keep all new credits guarded by useful current dispatch work.

- [ ] Task C: Add training-window timing telemetry tests.
  - Files: `tests/learn/test_train_joint_curriculum.py`, `src/learn/joint_metrics.py`.
  - Add snapshot fields listed above.
  - Add synthetic transition tests for aggregation.

- [ ] Task D: Evaluate whether the long-run gate needs extension.
  - Files only if needed: `src/eval/long_run_gate.py`, `tests/eval/test_long_run_gate.py`.
  - Skip gate extension if new fields are training-only and absent from eval summaries.
  - If eval summaries include new fields, add warnings for hold-under-pressure, secondary-fleet lateness exposure, and useful-dispatch misses.

- [ ] Task E: Create V2.1 250k config readiness only after Tasks A-D pass.
  - Do not create the config during the reward patch.
  - The config must initialize from the active production checkpoint, not from failed next-gen or failed V2 artifacts.
  - The config must keep registry skipped and trace logging disabled.

## Verification Commands For Future Implementation

Run focused tests first:

```powershell
python -m unittest tests.act.test_reward_physics_contract -v
python -m unittest tests.learn.test_train_joint_curriculum -v
python -m unittest tests.eval.test_long_run_gate -v
```

Compile changed Python files:

```powershell
python -m py_compile src\act\env_5pl.py src\learn\joint_metrics.py tests\act\test_reward_physics_contract.py tests\learn\test_train_joint_curriculum.py
```

If gate files change:

```powershell
python -m py_compile src\eval\long_run_gate.py src\eval\check_long_run_gate.py tests\eval\test_long_run_gate.py
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label production_self_check
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label nextgen_1m
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label nextgen_300k
```

Expected gate behavior:

- Production self-check exits 0.
- Next-gen 1M exits 2.
- Next-gen 300k exits 2.

## Non-Goals

- No registry update.
- No production overwrite.
- No baseline update.
- No DB cleanup or mutation.
- No blind 100M.
- No continuation from failed next-gen 1M.
- No direct 500k run before a V2.1 250k gate pass.
- No observation-dimension change.
- No action-space change.
- No contract rename.
- No hard-blocker weakening.
- No premium no-work credit reintroduction.

## Readiness Decision

Recommended next implementation step:

Implement Tasks A-C with test-driven development, then decide Task D based on whether the new fields are present in eval summaries. After that, create a separate V2.1 250k config readiness plan and run the short gate manually only after code/tests pass.

Final classification:

`STABILITY_V2_1_REWARD_BALANCE_LATENESS_PLAN_READY`
