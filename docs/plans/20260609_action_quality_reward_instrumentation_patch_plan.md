# Action Quality Reward Instrumentation Patch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the focused training telemetry, reward/action-quality guard, and gate coverage required before any V2 long-run training can be prepared or launched.

**Architecture:** Extend the existing joint-training metrics path instead of adding a second logger, then patch only the DQN local reward no-work dispatch path while preserving valid useful dispatch, route, premium, and hard-blocker behavior. Keep the read-only long-run gate checker as the post-eval safety layer and defer V2 curriculum config creation until telemetry and reward tests are green.

**Tech Stack:** Python `unittest`, PyTorch joint transition buffers, `src.learn.joint_metrics.JointMetricsAggregator`, `src.learn.train_joint_torch.append_training_metrics_jsonl`, `src.act.env_5pl.FivePLDigitalTwinEnv`, `src.eval.long_run_gate`, v5 contract `physical_reality_v5_route_candidate_visibility`, 73D observation, 48-action DQN dispatch space.

---

## Final Classification

`ACTION_QUALITY_REWARD_INSTRUMENTATION_PATCH_PLAN_READY`

## Current State

Production remains active:

`joint_torch_v5_balanced_retention_ft_200k_20260608`

Failed branch remains stopped:

`joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609`

Do not use the failed next-gen branch as a parent for additional long runs. Do not start 500k, 1M, 3M, 5M, 10M, or 100M branches from it.

## Evidence Inputs

This plan is based on read-only inspection of:

- `docs/plans/20260609_long_run_stability_design_v2_plan.md`
- `docs/plans/20260609_long_run_gate_automation_plan.md`
- `src/act/env_5pl.py`
- `src/learn/train_joint_torch.py`
- `src/learn/joint_metrics.py`
- `src/eval/long_run_gate.py`
- Production eval: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json`
- Next-gen 1M eval: `models/eval/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios/scenario_summary.json`
- Next-gen 300k eval: `models/eval/joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios/scenario_summary.json`

## Root-Cause Synthesis

Confirmed:

- Curriculum and schedule instability are confirmed. Next-gen 300k was 2/8 PASS with under-serving and lateness; next-gen 1M was 6/8 PASS with a different over-dispatch failure mode.
- Dispatch-success retention is missing. Production `premium_sla_pressure.dispatch_success_per_attempt` was 0.7689 and next-gen 1M fell to 0.4752; production `route_disruption_congestion.dispatch_success_per_attempt` was 0.8219 and next-gen 1M fell to 0.3770.
- DQN action drift is confirmed. Next-gen 1M concentrated around failed/no-op dispatch actions, including action 25 in route disruption, action 39 in premium SLA, and action 40 in mixed stress.
- Early stopping is missing. Training continued to 1M even though the 300k checkpoint later evaluated as 2/8 PASS.
- Periodic eval gates are now available through `src.eval.check_long_run_gate`, but they are not yet part of any launch workflow.

Likely:

- No-current/no-unassigned dispatch penalty is likely weak. In `src/act/env_5pl.py`, dispatch with no dispatched work receives a direct `unnecessary_dispatch_penalty` of 0.08, while larger demand quality penalties are primarily applied inside the useful-dispatch branch.
- The reward path already blocks premium no-work credit, route candidate no-work credit, and DQN delivery credit without current dispatch work; the failure is not a hard-blocker reward leak.

Ruled out for this patch:

- Contract mismatch: summaries use `physical_reality_v5_route_candidate_visibility`.
- Observation/action mismatch: current contract is 73 observation dimensions and 48 discrete actions.
- Registry or production issue: production remains the active/best model.
- Need for a blind longer run: the branch is already bracketed by failing 300k and 1M evaluations.

## Relevant Current Code Surfaces

Training metrics:

- `src/learn/train_joint_torch.py`
  - `run_training_loop` records every `JointTransition` into `JointMetricsAggregator`.
  - `append_training_metrics_jsonl` writes `snapshot.as_json_dict()` into `training_metrics.jsonl`.
- `src/learn/joint_metrics.py`
  - `JointMetricsAggregator.record_transition` already decodes every DQN action through `DiscreteActionMapper`.
  - Existing snapshot fields include `top_dqn_action_distribution`, dispatch/route/mode/reorder distributions, `dqn_action_entropy`, and mean DQN local reward.
  - Missing snapshot fields include per-window `action_failed_noop_by_id`, `action_no_current_work_by_id`, `action_no_unassigned_by_id`, dispatch success by action, top-action concentration, and DQN local reward by action family.

Reward logic:

- `src/act/env_5pl.py`
  - `_dqn_local_reward_components` exposes `current_dispatch_work_count`, `current_dispatch_work_signal`, `dispatch_feasibility_credit`, `dispatch_progress_credit`, `unnecessary_dispatch_penalty`, `dqn_delivery_credit`, premium no-work block fields, route candidate fields, and action 24/25 guard telemetry.
  - Valid useful dispatch receives feasibility/progress credit only when `dispatched_orders > 0.0`.
  - No-current-work dispatch currently receives a small direct no-op penalty and no delivery credit.

Gate logic:

- `src/eval/long_run_gate.py`
  - Fails scenario threshold failures.
  - Fails hard blockers.
  - Fails premium and route dispatch-success regressions versus production.
  - Fails no-current/no-unassigned explosions versus production.
  - Warns on selected production regressions and action concentration.

## Required Implementation Order

A. Training action-quality telemetry.

B. Reward guard patch.

C. Gate checker extension if newly emitted fields are needed.

D. V2 stability curriculum config.

E. Gated short run only after A-D are implemented, tested, and explicitly approved.

This order is mandatory because a V2 config without telemetry and reward guards would recreate the same blind-long-run risk.

## Task A: Training Action-Quality Telemetry

**Files:**

- Modify: `src/learn/joint_metrics.py`
- Modify: `tests/learn/test_train_joint_curriculum.py`
- Read-only reference: `src/learn/train_joint_torch.py`

### Target Fields

Add these fields to `JointMetricsSnapshot.as_json_dict()`:

- `action_failed_noop_by_id`: `dict[str, int]`
- `action_no_current_work_by_id`: `dict[str, int]`
- `action_no_unassigned_by_id`: `dict[str, int]`
- `action_dispatch_attempts_by_id`: `dict[str, int]`
- `action_dispatch_successes_by_id`: `dict[str, int]`
- `action_dispatch_success_ratio_by_id`: `dict[str, float]`
- `dispatch_success_per_attempt_by_stage_action`: `dict[str, float]`
- `top_action_concentration`: `float`
- `mean_dqn_local_reward_by_action_family`: `dict[str, float]`

Use compact JSON-compatible keys. Action IDs must be strings in JSON output, matching eval summary style.

### Telemetry Source Mapping

For each `JointTransition`, use:

- DQN action ID: `transition.dqn_action`
- DQN local reward: `transition.reward_dqn_local`
- Current stage: `transition.info["curriculum_stage"]` when present; otherwise `"unknown"`
- Dispatch attempt: decoded DQN action dispatch equals `"dispatch"`
- Dispatch success count: `transition.info["reward_components"]["current_dispatch_work_count"] > 0` or `transition.info["dispatch_success_count"] > 0`
- Failed/no-op dispatch: dispatch attempt with no current dispatch work signal or no dispatched orders
- No-current-work: `transition.info["reward_components"]["current_dispatch_work_signal"] <= 0` for dispatch actions, or explicit `dispatch_no_current_work` field if added later
- No-unassigned: `transition.info["dispatch_no_unassigned_orders"] > 0`

If both `reward_components` and top-level `info` expose the same signal, prefer `reward_components` for reward-derived fields and top-level `info` for projection/dispatch diagnostics.

### Exact Test Plan

- [ ] **Step A1: Write failing snapshot field test**

Add a test to `tests/learn/test_train_joint_curriculum.py` that constructs a `JointMetricsAggregator`, records synthetic transitions for actions 25 and 39, and asserts the new fields exist in `snapshot.as_json_dict()`.

Use existing `JointTransition` construction style from the test file. The synthetic transition for action 25 should include:

```python
info={
    "curriculum_stage": "route_disruption",
    "dispatch_success_count": 0.0,
    "dispatch_no_unassigned_orders": 1.0,
    "reward_components": {
        "current_dispatch_work_count": 0.0,
        "current_dispatch_work_signal": 0.0,
        "dqn_local": -0.22,
    },
}
```

Expected assertions:

```python
self.assertEqual(row["action_failed_noop_by_id"]["25"], 1)
self.assertEqual(row["action_no_current_work_by_id"]["25"], 1)
self.assertEqual(row["action_no_unassigned_by_id"]["25"], 1)
self.assertEqual(row["action_dispatch_attempts_by_id"]["25"], 1)
self.assertEqual(row["action_dispatch_successes_by_id"].get("25", 0), 0)
self.assertEqual(row["action_dispatch_success_ratio_by_id"]["25"], 0.0)
self.assertGreater(row["top_action_concentration"], 0.0)
self.assertIn("dispatch:shortest:secondary_fleet:conservative", row["mean_dqn_local_reward_by_action_family"])
```

- [ ] **Step A2: Run RED verification**

Run:

```powershell
python -m unittest tests.learn.test_train_joint_curriculum -v
```

Expected:

- FAIL because the new snapshot fields are not emitted.

- [ ] **Step A3: Implement telemetry aggregation**

In `src/learn/joint_metrics.py`:

- Extend `JointMetricsSnapshot` with the target fields.
- Initialize counters in `JointMetricsAggregator.reset()`.
- In `record_transition`, decode the DQN action once and record the new action-quality counters.
- Use helper functions to read numeric values from `transition.info` and nested `reward_components`.
- Add `_ratio_distribution(numerators, denominators)`.
- Add `_mean_by_key(sum_counter, count_counter)`.
- Add an action-family key helper:

```python
def _action_family_key(decoded: dict[str, str]) -> str:
    return (
        f"{decoded['dispatch']}:{decoded['route']}:"
        f"{decoded['mode']}:{decoded['reorder']}"
    )
```

- Do not change DB writes, checkpoint writes, registry behavior, or training loop control.

- [ ] **Step A4: Run GREEN verification**

Run:

```powershell
python -m unittest tests.learn.test_train_joint_curriculum -v
python -m py_compile src\learn\joint_metrics.py tests\learn\test_train_joint_curriculum.py
```

Expected:

- Unit tests PASS.
- Py compile exits 0.

## Task B: Reward Guard Patch For No-Current/No-Unassigned Dispatch

**Files:**

- Modify: `src/act/env_5pl.py`
- Modify: `tests/act/test_reward_physics_contract.py`

### Patch Intent

Strengthen the DQN local reward penalty when a dispatch action has no current dispatch work or no unassigned work, without suppressing valid useful dispatch and without changing observation/action contracts. This is the stronger no-current-work/no-unassigned dispatch penalty required before the next long-run branch.

The patch must:

- Preserve valid useful dispatch credit.
- Preserve safe shortest dispatch credit when useful and bounded.
- Preserve high_resilience and low_congestion dispatch behavior when useful.
- Preserve premium no-work credit blocking and premium no-work exploit prevention.
- Preserve route candidate alignment hard gates.
- Preserve hard-blocker counters and do not weaken hard-blocker gates.
- Avoid global dispatch suppression.
- Do not globally suppress dispatch or shortest/high_resilience.

### Suggested Reward Shape

Inside `_dqn_local_reward_components`, after `current_dispatch_work_signal` and dispatch diagnostic exposures are known:

- Introduce `no_current_or_unassigned_dispatch_penalty`.
- Apply it only when `dispatch == "dispatch"` and `current_dispatch_work_signal <= 0.0`.
- Scale by operational pressure so the penalty is stronger in the exact failure regimes:
  - no-current-work signal
  - no-unassigned signal
  - no immediate delivery
  - action-family concentration proxy if available
  - demand dispatch pressure
- Keep the penalty bounded so useful dispatch is not globally discouraged.

Concrete initial target:

```python
no_current_dispatch_work = 1.0 if dispatch == "dispatch" and current_dispatch_work_signal <= 0.0 else 0.0
no_current_or_unassigned_dispatch_penalty = (
    no_current_dispatch_work
    * (0.14 + (0.10 * demand_dispatch_pressure) + (0.08 * action_family_concentration_proxy))
)
reward -= no_current_or_unassigned_dispatch_penalty
```

If the implementation can safely use a top-level no-unassigned diagnostic, include it in the gate:

```python
no_unassigned_dispatch_exposure = 1.0 if _safe_float(projection_info.get("dispatch_no_unassigned_orders"), 0.0) > 0.0 else 0.0
```

Then the penalty can be:

```python
no_current_or_unassigned_dispatch_penalty = (
    max(no_current_dispatch_work, no_unassigned_dispatch_exposure)
    * (0.14 + (0.10 * demand_dispatch_pressure) + (0.08 * action_family_concentration_proxy))
)
```

The exact values may be adjusted in implementation if tests show an existing valid dispatch contract would be broken. Do not exceed a level that makes all dispatch unattractive in normal valid contexts.

### Exact Test Plan

- [ ] **Step B1: Write failing no-current/no-unassigned reward test**

Extend the existing no-current-work reward test near `tests/act/test_reward_physics_contract.py` no-current-work coverage.

Construct two component calls:

1. A no-current/no-unassigned dispatch with:

```python
projection_info={
    "action": {
        "dispatch": "dispatch",
        "route": "shortest",
        "mode": "secondary_fleet",
        "reorder": "conservative",
    },
    "dqn_dispatched_orders": 0.0,
    "dispatch_success_count": 0.0,
    "dispatch_no_unassigned_orders": 1.0,
    "macro_dispatch_budget": 1.0,
}
```

2. A valid useful dispatch with:

```python
projection_info={
    "action": {
        "dispatch": "dispatch",
        "route": "shortest",
        "mode": "primary_fleet",
        "reorder": "none",
    },
    "dqn_dispatched_orders": 1.0,
    "dispatch_success_count": 1.0,
    "macro_dispatch_budget": 1.0,
}
```

Expected assertions:

```python
self.assertGreater(no_work["no_current_or_unassigned_dispatch_penalty"], 0.0)
self.assertLessEqual(no_work["dqn_local"], -0.14)
self.assertEqual(no_work["dispatch_feasibility_credit"], 0.0)
self.assertEqual(no_work["dispatch_progress_credit"], 0.0)
self.assertEqual(no_work["dqn_delivery_credit"], 0.0)
self.assertGreater(valid_dispatch["dispatch_feasibility_credit"], 0.0)
self.assertEqual(valid_dispatch["no_current_or_unassigned_dispatch_penalty"], 0.0)
```

This is the valid dispatch still positive/bounded test for a representative useful dispatch: assert positive feasibility/progress credit, no new no-work guard, and bounded local reward under the existing envelope. Do not generalize this into an assertion that every valid dispatch context must be net positive.

- [ ] **Step B2: Write failing premium no-work exploit preservation test**

Extend existing premium guard coverage to assert:

```python
self.assertEqual(zero_work["premium_sla_fleet_credit_blocked_no_current_work"], 1.0)
self.assertEqual(zero_work["premium_primary_adaptation_credit_blocked_no_current_work"], 1.0)
self.assertEqual(zero_work["premium_fleet_credit_allowed"], 0.0)
self.assertGreater(zero_work["no_current_or_unassigned_dispatch_penalty"], 0.0)
self.assertEqual(useful_dispatch["premium_sla_fleet_credit_blocked_no_current_work"], 0.0)
self.assertEqual(useful_dispatch["premium_fleet_credit_allowed"], 1.0)
self.assertEqual(useful_dispatch["no_current_or_unassigned_dispatch_penalty"], 0.0)
```

- [ ] **Step B3: Write hard-blocker preservation test**

Use existing hard-blocker style assertions to confirm valid dispatch does not emit:

```python
self.assertEqual(components["dqn_delivery_credit_blocked_no_current_dispatch_work"], 0.0)
self.assertEqual(components["candidate_alignment_blocked_impossible_dispatch"], 0.0)
self.assertEqual(components["candidate_alignment_blocked_no_useful_work"], 0.0)
```

Also confirm no-current/no-unassigned dispatch does not gain:

```python
self.assertEqual(no_work["route_candidate_alignment_credit"], 0.0)
self.assertEqual(no_work["premium_fleet_credit_allowed"], 0.0)
self.assertEqual(no_work["dqn_delivery_credit"], 0.0)
```

- [ ] **Step B4: Run RED verification**

Run:

```powershell
python -m unittest tests.act.test_reward_physics_contract -v
```

Expected:

- FAIL because `no_current_or_unassigned_dispatch_penalty` is missing or zero.

- [ ] **Step B5: Implement focused reward guard**

In `src/act/env_5pl.py`:

- Add `no_unassigned_dispatch_exposure`.
- Add `no_current_or_unassigned_dispatch_penalty`.
- Subtract it only for dispatch actions with no current dispatch work or explicit no-unassigned exposure.
- Return both fields in the reward component dictionary.
- Do not change observation dim, action count, config schema, DB, registry, checkpoints, baselines, or production.

- [ ] **Step B6: Run GREEN verification**

Run:

```powershell
python -m unittest tests.act.test_reward_physics_contract -v
python -m py_compile src\act\env_5pl.py tests\act\test_reward_physics_contract.py
```

Expected:

- Unit tests PASS.
- Py compile exits 0.

## Task C: Gate Checker Extension If Needed

**Files:**

- Modify only if needed: `src/eval/long_run_gate.py`
- Modify only if needed: `tests/eval/test_long_run_gate.py`

### Decision Rule

Do not extend the gate checker if existing eval summaries already contain all fields required to fail next-gen 1M and 300k while passing production.

Extend it only if Task A or B produces a new field that should become part of future long-run gating.

Candidate future fatal fields:

- `action_failed_noop_by_id`
- `action_dispatch_success_ratio_by_id`
- `top_action_concentration`
- `mean_dqn_local_reward_by_action_family`

### Exact Test Plan If Extension Is Needed

- [ ] **Step C1: Add failing gate test for low-success top action**

In `tests/eval/test_long_run_gate.py`, add a test where:

```python
candidate = _scenario(
    "route_disruption_congestion",
    dispatch_success=0.80,
    action_no_current_work_by_id={"25": 130},
    action_no_unassigned_by_id={"25": 130},
    top_action_ids=[{"action_id": 25, "count": 260, "percentage": 0.30}],
)
```

Expected:

```python
self.assertEqual(report["decision"], "FAIL")
self.assertTrue(any("action 25" in item for item in report["fatal_failures"]))
```

- [ ] **Step C2: Preserve production self-check**

Add or keep the existing self-check test proving production-like existing exposure does not fail when candidate equals production.

- [ ] **Step C3: Run RED verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- FAIL if the new rule is not implemented.

- [ ] **Step C4: Implement minimal gate extension**

In `src/eval/long_run_gate.py`:

- Keep CLI stdout-only JSON behavior.
- Keep exit code 0 PASS and 2 FAIL in `src/eval/check_long_run_gate.py`.
- Compare candidate against production, not absolute values only.
- Do not read checkpoints, mutate files, or call evaluator/training.

- [ ] **Step C5: Run real-artifact sanity checks**

Run:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json `
  --production-summary models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json `
  --candidate-label production_self_check
```

Expected:

- Exit code 0.
- Decision PASS.

Run:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models/eval/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios/scenario_summary.json `
  --production-summary models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json `
  --candidate-label nextgen_1m
```

Expected:

- Exit code 2.
- Decision FAIL.

Run:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models/eval/joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios/scenario_summary.json `
  --production-summary models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json `
  --candidate-label nextgen_300k
```

Expected:

- Exit code 2.
- Decision FAIL.

- [ ] **Step C6: Run focused compile**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
python -m py_compile src\eval\long_run_gate.py src\eval\check_long_run_gate.py tests\eval\test_long_run_gate.py
```

Expected:

- Unit tests PASS.
- Py compile exits 0.

## Task D: V2 Stability Curriculum Config

**Files:**

- Future create only after Tasks A-C are green: a new V2 training config with unique identity.
- Future modify: focused config validation tests.

### Preconditions

Do not start this task until:

- Task A telemetry fields exist in `training_metrics.jsonl`.
- Task B reward guard tests pass.
- Task C is either explicitly skipped because no gate extension is needed, or implemented and verified.
- Production self-check passes the gate checker.
- Next-gen 1M and 300k still fail the gate checker.

### Config Requirements

The V2 config must:

- Use a unique experiment, environment, and team id.
- Use contract `physical_reality_v5_route_candidate_visibility`.
- Keep observation/action contract 73/48.
- Use production checkpoint as parent only after A-C are complete.
- Avoid failed next-gen 1M as parent.
- Avoid v3/v4/invalid 700k/old invalid rewardfix references.
- Require `--skip-registry`.
- Require trace logging disabled.
- Write to a fresh output directory only.
- Include short alternating retention stages.
- Include explicit gate stops at 250k, 500k, and 1M before any longer horizon.

### Test Requirements

Focused config tests must prove:

- Unique identity.
- Correct contract.
- Correct observation/action dimensions.
- Correct parent checkpoint.
- No failed-branch parent.
- No registry write by default/manual command.
- No existing output directory overwrite.

## Task E: Gated Short Run Only After A-D

This task is not authorized by this plan.

After A-D, the user may manually approve a short gated run. Until then:

- Do not train.
- Do not run offline evaluation.
- Do not create 500k/1M/3M/5M/10M/100M branches.
- Do not update registry or production.

If a short run is later approved:

- Evaluate only to a fresh eval output directory.
- Run `src.eval.check_long_run_gate`.
- Stop immediately on any FAIL.
- Require human acceptance of warnings before continuing.

## Required Final Verification For Future Implementation

After Tasks A-C, run:

```powershell
python -m unittest tests.learn.test_train_joint_curriculum -v
python -m unittest tests.act.test_reward_physics_contract -v
python -m unittest tests.eval.test_long_run_gate -v
python -m py_compile src\learn\joint_metrics.py src\learn\train_joint_torch.py src\act\env_5pl.py src\eval\long_run_gate.py src\eval\check_long_run_gate.py tests\learn\test_train_joint_curriculum.py tests\act\test_reward_physics_contract.py tests\eval\test_long_run_gate.py
```

Also run the three read-only gate sanity checks described in Task C if the gate checker changed. These checks prove the gate checker still passes production and fails next-gen 1M and 300k.

Do not use green tests to justify training. Training remains a separate manual approval.

## Non-Goals

This plan does not authorize:

- No blind 100M.
- No continuation from failed next-gen 1M.
- No registry/production change.
- No baseline update.
- Any DB mutation.
- Any checkpoint mutation.
- Any config edit during this plan-only step.
- Any code edit during this plan-only step.
- Any offline evaluation during this plan-only step.
- Any training during this plan-only step.

## Implementation Posture

Recommended posture:

`REWARD_AND_TELEMETRY_PATCH_BEFORE_V2_CONFIG`

Reason:

- A config-only change would not make the next long run observable or stoppable.
- A reward-only change would not prove action-quality stability.
- A telemetry-only change would detect drift but not reduce the reward incentive for no-current/no-unassigned dispatch.
- A gate-only change catches failures after eval but does not improve training behavior.

The safe path is telemetry first, reward guard second, gate adjustment only if needed, and V2 curriculum config last.
