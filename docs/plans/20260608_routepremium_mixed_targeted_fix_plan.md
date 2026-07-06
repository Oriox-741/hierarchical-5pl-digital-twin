# Route Premium Mixed Targeted Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare a safe, evidence-first fix path for the clean v5 rewardfix perfclean model's premium SLA service failure, route-disruption action31 no-op failure, and mixed-stress route concentration watch.

**Architecture:** First expose missing evaluator telemetry that the environment already computes, then rerun offline evaluation, then create a targeted continuation fine-tune config if the telemetry confirms policy undertraining rather than a reward bug. Do not change registry, baselines, production checkpoints, canonical thresholds, or the existing eval outputs.

**Tech Stack:** Python 3.12, `unittest`, PyTorch trainer entrypoint `src.learn.train_joint_torch`, evaluator entrypoint `src.eval.evaluate_real_world_scenarios`, JSON curriculum configs, v5 contract `physical_reality_v5_route_candidate_visibility`.

---

## Current State Summary

Current valid run:
- Checkpoint: `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt`
- Evaluation output: `models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608_offline_scenarios`
- Env id: `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation/action: `73/48`
- Global step: `1000000`
- Offline eval classification: `OFFLINE_SCENARIO_EVALUATION_FAIL_SCENARIO_THRESHOLD`
- Global hard blockers: all zero.

Invalid artifacts that must remain unused:
- `joint_curriculum_v5_clean_cold_start_1m`
- `models/checkpoints/joint_torch_v5_clean_cold_start_1m`
- `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix`
- `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix`
- any v3/v4 checkpoint
- any invalid 700k artifact

Files inspected while creating this plan:
- `models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608_offline_scenarios/episode_metrics.jsonl`
- `models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608_offline_scenarios/scenario_summary.json`
- `models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608_offline_scenarios/scenario_summary.csv`
- `models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608_offline_scenarios/real_world_evaluation_report.md`
- `configs/eval_scenarios/premium_sla_pressure.json`
- `configs/eval_scenarios/route_disruption_congestion.json`
- `configs/eval_scenarios/mixed_stress.json`
- `src/eval/scenario_metrics.py`
- `src/eval/real_world_scenario_arena.py`
- `src/act/env_5pl.py`
- `src/act/observation_builder.py`
- `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/training_metrics.jsonl`

## Classifications

Route disruption action31 classification:

`ROUTE_ACTION31_CURRICULUM_FINE_TUNE_REQUIRED`

Reason: action31 is valid in rare contexts, but the clean 1M policy over-selects it in route disruption where it almost never produces dispatch work. Existing reward logic already applies unnecessary dispatch penalties and blocks no-work route credit; the immediate evidence points to policy/curriculum undertraining, not a clear reward-side exploit.

Premium SLA classification:

`PREMIUM_SLA_CURRICULUM_FINE_TUNE_REQUIRED`

Reason: premium hard blockers are clean and primary-fleet reward guard telemetry is behaving, but the policy holds too often and does not use emergency/aggressive reorder under tight SLA pressure. The configured `min_service_level` is below the scenario's `service_target=0.98`, so the failure should not be treated as a threshold-only issue.

Mixed stress classification:

`MIXED_STRESS_ROUTE_ADAPTIVE_ACCEPTABLE_WATCH`

Reason: mixed stress passes thresholds with service `0.9227`, lateness `0.0`, dispatch success `1.0`, and zero failed/no-op dispatch. High-resilience concentration is large, but the scenario combines route disruption, congestion, demand, holding-cost, and vehicle-scarcity pressure. Current artifacts do not prove avoidable overconservatism.

## Evidence

### Premium SLA Failure Evidence

Scenario config:
- `configs/eval_scenarios/premium_sla_pressure.json`
- `service_target=0.98`
- `premium_sla_ratio=0.55`
- `urgent_due_window_multiplier=0.65`
- `premium_lateness_penalty_multiplier=1.8`
- `primary_fleet_penalty=0.08`
- failing threshold: `min_service_level=0.92`

Aggregate eval evidence:
- Verdict: `FAIL`
- Scenario threshold verdict: `FAIL`
- Hard-blocker verdict: `PASS`
- Threshold failure: `service_level 0.871 < 0.920`
- Service: `0.8709914182475158`
- Lateness: `0.0`
- Dispatch success: `0.9475795971410007`
- Delivered delta / mean step delivered delta: `0.5462962962962963`
- Total delivered: `472`
- No vehicle: `0`
- Already assigned context rows: `133`
- Route failures: `1`
- Dispatch/hold: `388/476`
- Fleet distribution: `primary_fleet=587`, `secondary_fleet=277`
- Route distribution: `shortest=287`, `low_congestion=328`, `high_resilience=249`
- Reorder distribution: `conservative=319`, `none=545`, `emergency=0`, `aggressive=0`
- `no_work_positive_dqn_local=0`
- `premium_sla_fleet_credit_blocked_no_current_work=336`
- `premium_primary_adaptation_credit_blocked_no_current_work=336`
- `premium_fleet_credit_allowed=251`

Per-episode evidence:
- Seed 102: service `0.9024`, lateness `0.0`, attempts/successes `133/126`, dispatch/hold `133/155`, total delivered `164`, no vehicle `0`, route failures `0`.
- Seed 103: service `0.8841`, lateness `0.0`, attempts/successes `135/122`, dispatch/hold `135/153`, total delivered `164`, no vehicle `0`, route failures `1`.
- Seed 104: service `0.8264`, lateness `0.0`, attempts/successes `120/119`, dispatch/hold `120/168`, total delivered `144`, no vehicle `0`, route failures `0`.

Interpretation:
- Service failure is not caused by no-vehicle exposure or fake reward credit.
- Primary fleet is used heavily, so the failure is not simply lack of primary-fleet choice.
- Reorder behavior is conservative: no emergency/aggressive reorder appears under premium SLA pressure.
- The clearest policy issue is too much hold and too little urgent-useful action volume, especially seed 104.

### Route Action31 Failure Evidence

Scenario config:
- `configs/eval_scenarios/route_disruption_congestion.json`
- `route_disruption_probability=0.60`
- `congestion_multiplier=3.0`
- `shortest_route_risk_multiplier=2.0`
- `traversal_cost_multiplier=1.5`
- `disruption_risk_multiplier=1.8`
- failing threshold: `min_dispatch_success_per_attempt=0.50`

Aggregate eval evidence:
- Verdict: `FAIL`
- Scenario threshold verdict: `FAIL`
- Hard-blocker verdict: `PASS`
- Threshold failure: `dispatch_success_per_attempt 0.387 < 0.500`
- Dispatch success per attempt: `0.386647939657`
- Service: `1.0`
- Lateness: `0.0`
- Delivered delta / mean step delivered delta: `0.49421296296296297`
- Total delivered: `427`
- No vehicle: `0`
- Already assigned context rows: `104`
- Route failures: `1`
- Dispatch/hold: `729/135`
- Route distribution: `shortest=380`, `low_congestion=349`, `high_resilience=135`
- Reorder distribution: `emergency=378`, `conservative=144`, `none=342`

Route split:
- Total route exposure: shortest `380/864 = 44.0%`, low_congestion `349/864 = 40.4%`, high_resilience `135/864 = 15.6%`
- Successful dispatch: shortest `1/282 = 0.4%`, low_congestion `216/282 = 76.6%`, high_resilience `65/282 = 23.0%`
- Failed/no-op dispatch: shortest `365/447 = 81.7%`, low_congestion `14/447 = 3.1%`, high_resilience `68/447 = 15.2%`
- Hold: shortest `14/135 = 10.4%`, low_congestion `119/135 = 88.1%`, high_resilience `2/135 = 1.5%`

Action31 evidence:
- Decode: `dispatch + shortest + primary_fleet + emergency`
- Attempts: `366`
- Successes: `1`
- Failed/no-op dispatch: `365`
- Seed 92: action31 `122`, success `1`, failed/no-op `121`, already-assigned context rows `30`, no vehicle `0`, route failures `1`
- Seed 93: action31 `125`, success `0`, failed/no-op `125`, already-assigned context rows `38`, no vehicle `0`, route failures `0`
- Seed 94: action31 `119`, success `0`, failed/no-op `119`, already-assigned context rows `36`, no vehicle `0`, route failures `0`

Interpretation:
- Shortest is not useful in route disruption here; it is mostly a no-op label attached to action31.
- The problem is action quality, not a route-distribution cosmetics issue.
- The existing artifacts do not attribute action31 failures to no-unassigned, no-current-work, already-assigned, no-vehicle, or route-failure at the per-action level.

### Mixed Stress Watch Evidence

Scenario config:
- `configs/eval_scenarios/mixed_stress.json`
- `rolling_demand_probability_multiplier=1.7`
- `rolling_demand_volume_multiplier=2.0`
- `order_units_multiplier=1.45`
- `urgent_order_probability_multiplier=1.8`
- `holding_cost_multiplier=2.0`
- `vehicle_availability_multiplier=0.5`
- `fleet_capacity_multiplier=0.6`
- `capacity_shock_severity=0.4`
- `route_disruption_probability=0.55`
- `congestion_multiplier=2.5`
- `traversal_cost_multiplier=1.4`
- `stockout_penalty_multiplier=1.6`
- `inventory_shortfall_penalty_multiplier=1.5`
- `transport_cost_penalty_scale=180.0`

Aggregate eval evidence:
- Verdict: `PASS`
- Scenario threshold verdict: `PASS`
- Hard-blocker verdict: `PASS`
- Threshold failures: none
- Threshold warning: `mixed_success_route_failure_steps present: 1`
- Service: `0.9227176069195916`
- Lateness: `0.0`
- Dispatch success per attempt: `1.0`
- Delivered delta / mean step delivered delta: `1.4027777777777777`
- Total delivered: `1212`
- No vehicle: `48`
- Already assigned context rows: `369`
- Route failures: `1`
- Dispatch/hold: `802/62`
- Route distribution: `shortest=36`, `low_congestion=128`, `high_resilience=700`
- Fleet distribution: `primary_fleet=413`, `secondary_fleet=451`
- Reorder distribution: `aggressive=12`, `conservative=214`, `none=638`

Route split:
- Total route exposure: shortest `36/864 = 4.2%`, low_congestion `128/864 = 14.8%`, high_resilience `700/864 = 81.0%`
- Successful dispatch: shortest `7/802 = 0.9%`, low_congestion `98/802 = 12.2%`, high_resilience `697/802 = 86.9%`
- Failed/no-op dispatch: `0`
- Hold: shortest `29/62 = 46.8%`, low_congestion `30/62 = 48.4%`, high_resilience `3/62 = 4.8%`

Interpretation:
- Mixed high-resilience dominance is currently plausible adaptive routing under severe stress.
- It is not currently a threshold failure because service, lateness, dispatch success, hard blockers, and failed/no-op dispatch all pass.
- It becomes a true failure only if telemetry shows high_resilience was repeatedly selected when shortest or low_congestion had near-best or better candidate scores, safe useful dispatch opportunity was present, and service/lateness could be preserved with less conservative routing.

## Source Findings

Evaluator telemetry currently reports action-level success and failed/no-op counts, but not per-step route candidate scores:
- `src/eval/scenario_metrics.py` records `dispatch_success_by_action_id` and `failed_noop_dispatch_by_action_id`.
- It records hard-blocker counters including no-current-work delivery credit, hold leaks, action8 leaks, unsafe 24/25 candidate credit, no-work positive DQN local, and emergency zero-useful credit.
- It does not store or aggregate `selected_route_candidate_score`, `shortest_route_candidate_score`, `low_congestion_route_candidate_score`, `high_resilience_route_candidate_score`, `selected_route_candidate_score_gap`, `selected_route_candidate_near_best`, `route_candidate_mismatch_penalty`, `route_candidate_alignment_credit`, or `useful_dispatch_opportunity`.

The env already computes route candidate and premium guard telemetry:
- `src/act/env_5pl.py` returns reward components including `shortest_route_candidate_score`, `low_congestion_route_candidate_score`, `high_resilience_route_candidate_score`, `selected_route_candidate_score`, `best_route_candidate_score`, `selected_route_candidate_score_gap`, `selected_route_candidate_near_best`, `route_candidate_alignment_credit`, `route_candidate_mismatch_penalty`, `route_candidate_useful_work_factor`, `route_adaptation_pressure`, `route_disruption_reliability_pressure`, `route_congestion_cost_pressure`, `premium_sla_fleet_credit_blocked_no_current_work`, `premium_primary_adaptation_credit_blocked_no_current_work`, and `premium_fleet_credit_allowed`.
- `src/act/observation_builder.py` includes v5 route candidate visibility features in the 73D observation: `shortest_candidate_score`, `low_congestion_candidate_score`, `high_resilience_candidate_score`, score gaps, `shortest_candidate_near_best`, safety/brittle flags, secondary fleet feasibility, and `useful_dispatch_opportunity`.

Conclusion:
- Mixed stress should not receive a reward patch before evaluator telemetry exposes selected-vs-best route evidence.
- Route action31 and premium SLA need targeted continuation training, but adding telemetry first makes the fine-tune verifiable.

## Fix Option Matrix

| Issue | Telemetry-only patch | Reward patch | Curriculum/fine-tune | Threshold review | No-op |
|---|---|---|---|---|---|
| Route action31 no-op | Required first to add per-action no-unassigned/current-work and route candidate attribution | Not first; existing no-op penalties and no-work credit gates exist | Recommended after telemetry patch | Not sufficient; the action is objectively poor | No |
| Premium SLA service | Useful to expose backlog/pending/urgent pressure by step and premium useful dispatch context | Not first; guard telemetry is clean and no-work credit is zero | Recommended after telemetry patch | Not first; threshold is below `service_target=0.98` | No |
| Mixed stress route concentration | Required before calling it overconservative | Not justified now | Only if telemetry proves high-resilience overconservatism | Not needed now; scenario passes | Accept as watch for now |

## Recommended First Implementation Step

Implement a minimal evaluator telemetry patch that wires existing env reward components into eval outputs and adds per-action reason counters for failed/no-op dispatch.

Future files to modify:
- `src/eval/scenario_metrics.py`
- `src/eval/real_world_scenario_arena.py` only if the markdown report should surface new telemetry tables.
- `tests/eval/test_real_world_scenario_evaluator.py`

Do not modify:
- training configs
- checkpoints
- registry
- baselines
- production folders
- existing eval output directories

### Proposed Codex Prompt For First Implementation Step

```text
USESuperpowers:
- using-superpowers
- systematic-debugging
- test-driven-development
- verification-before-completion
- requesting-code-review

Implement the evaluator telemetry patch only. Do not train. Do not run full evaluation. Do not update registry/DB/checkpoints/baselines/production. Do not overwrite existing eval outputs.

Goal:
Expose existing route candidate and failed/no-op reason telemetry in offline eval outputs.

Allowed source/test files:
- src/eval/scenario_metrics.py
- src/eval/real_world_scenario_arena.py only if report table wiring is needed
- tests/eval/test_real_world_scenario_evaluator.py

Requirements:
1. Add EpisodeMetrics fields and aggregation for:
   - shortest_route_candidate_score
   - low_congestion_route_candidate_score
   - high_resilience_route_candidate_score
   - selected_route_candidate_score
   - best_route_candidate_score
   - selected_route_candidate_score_gap
   - selected_route_candidate_near_best
   - selected_route_candidate_action_gate
   - route_candidate_alignment_credit
   - route_candidate_mismatch_penalty
   - route_candidate_useful_work_factor
   - route_adaptation_pressure
   - route_disruption_reliability_pressure
   - route_congestion_cost_pressure
   - useful_dispatch_opportunity if available from observation/raw info
2. Add split counters by action id for failed/no-op reason fields where reward_components already include them:
   - dispatch_no_unassigned_orders
   - dispatch_no_vehicle_available
   - dispatch_route_failure
   - dispatch_already_assigned_count
   - current_dispatch_work_count <= 0 and current_dispatch_work_signal <= 0
3. Preserve all existing hard-blocker behavior and output field names.
4. Add focused tests proving the new fields appear in episode and scenario outputs.
5. Do not add thresholds or change verdict logic.

Validation:
- python -m unittest tests.eval.test_real_world_scenario_evaluator -v
- python -m py_compile src\eval\scenario_metrics.py src\eval\real_world_scenario_arena.py tests\eval\test_real_world_scenario_evaluator.py
```

## Implementation Tasks

### Task 1: Evaluator Route-Candidate Telemetry Patch

**Files:**
- Modify: `src/eval/scenario_metrics.py`
- Test: `tests/eval/test_real_world_scenario_evaluator.py`

- [ ] **Step 1: Write failing test for route candidate fields**

Add a focused test that constructs an episode row through the evaluator metric accumulator with reward components including:

```python
reward_components = {
    "selected_route_candidate_score": 0.42,
    "shortest_route_candidate_score": 0.20,
    "low_congestion_route_candidate_score": 0.42,
    "high_resilience_route_candidate_score": 0.31,
    "best_route_candidate_score": 0.42,
    "selected_route_candidate_score_gap": 0.0,
    "selected_route_candidate_near_best": 1.0,
    "selected_route_candidate_action_gate": 1.0,
    "route_candidate_alignment_credit": 0.004,
    "route_candidate_mismatch_penalty": 0.0,
    "route_candidate_useful_work_factor": 1.0,
    "route_adaptation_pressure": 0.70,
    "route_disruption_reliability_pressure": 0.55,
    "route_congestion_cost_pressure": 0.60,
}
```

Assert that `EpisodeMetrics.to_dict()` and `summarize_episodes()` include mean values for those fields.

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected before implementation: fail because the new fields are absent.

- [ ] **Step 3: Implement minimal metric fields**

Add float fields to `EpisodeMetrics`, collect them in `EpisodeMetricAccumulator.last_values`, pass them through `finish()`, include them in `to_dict()`, and aggregate with `mean()` in `summarize_episodes()`.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected after implementation: pass.

- [ ] **Step 5: Compile changed files**

Run:

```powershell
python -m py_compile src\eval\scenario_metrics.py tests\eval\test_real_world_scenario_evaluator.py
```

Expected: exit code `0`.

### Task 2: Failed/No-Op Reason Split Patch

**Files:**
- Modify: `src/eval/scenario_metrics.py`
- Test: `tests/eval/test_real_world_scenario_evaluator.py`

- [ ] **Step 1: Write failing test for action-level no-op reasons**

Add a test where raw action id `31` has `dispatch_success_count=0` and reward components include:

```python
reward_components = {
    "dispatch_success_count": 0.0,
    "dispatch_attempted": 1.0,
    "dispatch_no_unassigned_orders": 1.0,
    "dispatch_no_vehicle_available": 0.0,
    "dispatch_route_failure": 0.0,
    "dispatch_already_assigned_count": 4.0,
    "current_dispatch_work_count": 0.0,
    "current_dispatch_work_signal": 0.0,
}
```

Assert scenario output includes action-level counters such as:

```python
{
    "failed_no_unassigned_by_action_id": {"31": 1},
    "failed_no_current_work_by_action_id": {"31": 1},
    "failed_already_assigned_context_by_action_id": {"31": 1},
}
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected before implementation: fail because the action-level reason counters are absent.

- [ ] **Step 3: Implement the minimal counters**

When `dispatch == "dispatch"` and `dispatch_success_count <= 0`, increment per-action counters for any positive reason component. Keep existing `failed_noop_dispatch_by_action_id` unchanged.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected: pass.

- [ ] **Step 5: Compile changed files**

Run:

```powershell
python -m py_compile src\eval\scenario_metrics.py tests\eval\test_real_world_scenario_evaluator.py
```

Expected: exit code `0`.

### Task 3: Read-Only Diagnostic Offline Eval Rerun After Telemetry Patch

**Files:**
- Write new eval artifacts only under a new output dir.
- Do not overwrite existing output dirs.

- [ ] **Step 1: Run one offline eval with patched telemetry**

Run only after Tasks 1 and 2 pass:

```powershell
python -m src.eval.evaluate_real_world_scenarios `
  --checkpoint models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt `
  --scenario-dir configs/eval_scenarios `
  --output-dir models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608_offline_scenarios_routepremium_mixed_telemetry `
  --device cpu `
  --deterministic
```

Expected:
- 8 scenarios
- 24 episode rows
- Existing hard blockers remain zero
- New route candidate fields present in `episode_metrics.jsonl`, `scenario_summary.json`, and `scenario_summary.csv`
- Existing output dir remains untouched.

- [ ] **Step 2: Audit route action31 with new reason splits**

Confirm whether action31 failures are mostly:
- no unassigned work
- no-current-work
- already-assigned saturation
- no vehicle
- route failure
- poor candidate score or selected-vs-best gap

- [ ] **Step 3: Audit mixed stress with candidate scores**

Confirm whether high_resilience selections are near-best or best under stress. Mixed becomes a true issue only if high_resilience repeatedly has a poor selected-vs-best score while shortest or low_congestion is safe and near-best.

### Task 4: Targeted Fine-Tune Config

**Files:**
- Create: `configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft.json`
- Modify: `tests/learn/test_curriculum_config.py`

- [ ] **Step 1: Write failing config validation test**

Add a test proving the future config:
- uses contract `physical_reality_v5_route_candidate_visibility`
- keeps observation/action `73/48`
- uses unique identity `joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft`
- sets `total_timesteps` to `1150000`
- preserves checkpoint source by CLI resume, not in config
- avoids references to invalid old 1M, aborted rewardfix, v3/v4, and invalid 700k artifacts
- has consumed schedule prefix `normal_v4=1000000`, then `route_disruption=50000`, `premium_sla=50000`, `mixed_stress=50000`

Expected future test skeleton:

```python
def test_routepremium_mixed_ft_config_loads_and_is_resume_safe(self) -> None:
    config = load_config(Path("configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft.json"))
    shared = config["shared_global_parameters"]
    curriculum = curriculum_config_from_training_config(config, fallback_seed=99)

    self.assertEqual(config["mdp_contract_version"], "physical_reality_v5_route_candidate_visibility")
    self.assertEqual(shared["observation_dim"], 73)
    self.assertEqual(shared["total_timesteps"], 1_150_000)
    self.assertEqual(shared["experiment_name"], "joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft")
    self.assertEqual(shared["environment_id"], "joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft")
    self.assertEqual(shared["team_id"], "joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft")
    self.assertTrue(curriculum.enabled)
    self.assertEqual(curriculum.stage, "normal_v4")
    self.assertEqual(
        [(entry.name, entry.steps) for entry in curriculum.stage_schedule],
        [
            ("normal_v4", 1_000_000),
            ("route_disruption", 50_000),
            ("premium_sla", 50_000),
            ("mixed_stress", 50_000),
        ],
    )
    text = Path("configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft.json").read_text(encoding="utf-8")
    self.assertNotIn("joint_curriculum_v5_clean_cold_start_1m_after_rewardfix\\\"", text)
    self.assertNotIn("joint_curriculum_v5_clean_cold_start_1m\\\"", text)
    self.assertNotIn("v4", text)
    self.assertNotIn("700k", text.lower())
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```powershell
python -m unittest tests.learn.test_curriculum_config -v
```

Expected before config creation: fail because the config file does not exist.

- [ ] **Step 3: Create the config**

Derive from `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json` with these exact changes:
- `shared_global_parameters.experiment_name`: `joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft`
- `shared_global_parameters.environment_id`: `joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft`
- `shared_global_parameters.team_id`: `joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft`
- `shared_global_parameters.total_timesteps`: `1150000`
- `curriculum.stage`: `normal_v4`
- `curriculum.stage_schedule`:

```json
[
  {"name": "normal_v4", "steps": 1000000},
  {"name": "route_disruption", "steps": 50000},
  {"name": "premium_sla", "steps": 50000},
  {"name": "mixed_stress", "steps": 50000}
]
```

Rationale: the trainer resumes from checkpoint global step `1000000`; the consumed schedule prefix prevents the resumed run from starting at the wrong stage.

- [ ] **Step 4: Run focused validation**

Run:

```powershell
python -m unittest tests.learn.test_curriculum_config -v
python -m py_compile tests\learn\test_curriculum_config.py
```

Expected: both commands exit `0`.

### Task 5: Manual Targeted Fine-Tune

**Files:**
- New output dir only: `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft`
- Do not update registry.
- Do not write DB rows.
- Do not write production final checkpoints.

- [ ] **Step 1: User manually runs the fine-tune command**

DO NOT RUN FROM CODEX unless explicitly approved later:

```powershell
python -m src.learn.train_joint_torch `
  --config configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft.json `
  --output-dir models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft `
  --resume models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt `
  --device cpu `
  --torch-num-threads 2 `
  --torch-num-interop-threads 1 `
  --skip-registry `
  --disable-trace-logging `
  --final-ppo-path models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft/ppo_torch_joint_final_routepremium_mixed_ft.pt `
  --final-dqn-path models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft/dqn_torch_joint_final_routepremium_mixed_ft.pt
```

Expected:
- resumes from global step `1000000`
- trains to global step `1150000`
- trace logging disabled
- registry skipped
- no DB trace rows
- final PPO/DQN paths remain inside the new output dir
- no production checkpoint folder writes

### Task 6: Offline Evaluation Rerun After Fine-Tune

**Files:**
- New eval output dir only: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_offline_scenarios`
- Do not overwrite existing eval dirs.

- [ ] **Step 1: Run one offline eval after fine-tune**

DO NOT RUN until the fine-tune finishes and checkpoint metadata is verified:

```powershell
python -m src.eval.evaluate_real_world_scenarios `
  --checkpoint models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft/joint_torch_latest.pt `
  --scenario-dir configs/eval_scenarios `
  --output-dir models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_offline_scenarios `
  --device cpu `
  --deterministic
```

Expected pass criteria:
- 8 scenarios
- 24 episode rows
- hard blockers all zero
- `premium_sla_pressure` service >= `0.920`
- `route_disruption_congestion` dispatch_success_per_attempt >= `0.500`
- action31 no-op exposure materially reduced
- mixed stress service/lateness preserved
- mixed stress high_resilience remains justified by candidate score telemetry or concentration relaxes without harming service

## Risk List

- Fine-tune overfits route disruption and damages premium or mixed stress.
- Reducing action31 no-ops could accidentally suppress rare valid shortest emergency contexts.
- Pushing premium dispatch volume could create no-work premium credit if reward gates are weakened; do not weaken them.
- Mixed stress high-resilience concentration could be wrongly penalized despite being adaptive under severe stress.
- Resumed trainer stage scheduling can start in the wrong stage if the consumed `normal_v4=1000000` schedule prefix is omitted.
- Default final PPO/DQN paths can write outside the output dir unless overridden in the manual command.
- Registry update would be premature until offline eval passes after fine-tune.

## Explicit Non-Goals

- No registry update.
- No production checkpoint modification.
- No baseline overwrite.
- No use of invalid old v5, aborted rewardfix, v3/v4, or invalid 700k artifacts.
- No blind shortest boost.
- No global high_resilience penalty.
- No service/lateness degradation for route diversity.
- No reintroduction of no-work premium fleet credit.
- No weakening of global hard blockers.
- No canonical scenario threshold changes without a separate threshold-review task.

## Final Recommendation

Recommended path:
1. Implement evaluator telemetry patch.
2. Rerun offline eval once into a new telemetry output dir.
3. If telemetry confirms no reward bug, create the targeted continuation fine-tune config.
4. User manually runs the targeted fine-tune.
5. Rerun offline scenario evaluation into a new output dir.
6. Only consider registry candidate review if all thresholds and hard blockers pass.

Code/config changes required next:
- Yes: evaluator telemetry source/test changes first.
- Yes later: targeted fine-tune config/test changes if telemetry confirms the current classifications.

Targeted fine-tune recommended:
- Yes, after telemetry patch and telemetry rerun.

Current classification:
- `ROUTEPREMIUM_MIXED_TARGETED_FIX_PLAN_READY`

## Balanced Retention Follow-up After 200k Regression

The first targeted 200k route/premium/mixed fine-tune is not registry-ready. It fixed the intended narrow objectives, but the offline scenario evaluation showed retention regressions:
- Route disruption dispatch success improved from `0.387` to `0.927`, and action31 failed/no-op dispatch dropped from `365` to `0`.
- Premium SLA service improved from `0.871` to `0.953`.
- Baseline normal service regressed from `0.962` to `0.887`.
- Lead-time volatility service regressed from `0.964` to `0.797`.
- High holding cost service regressed from `0.970` to `0.877`.

The regression root cause is targeted fine-tune forgetting: the narrow schedule lacked lead-time and high-holding retention, had only early normal exposure, and ended on premium SLA. The next fine-tune should initialize from the clean 1M perfclean parent again, not from the failed 200k final checkpoint, so the new run starts from the known clean rewardfix policy and uses fresh optimizers.

Balanced-retention config:
- `configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json`
- Identity: `joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k`
- Parent checkpoint: `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt`
- Future output dir: `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608`

Balanced 200k schedule:
- `normal_v4`: 20k
- `route_disruption`: 25k
- `premium_sla`: 25k
- `lead_time_delay`: 25k
- `high_holding_cost`: 25k
- `mixed_stress`: 20k
- `demand_spike`: 10k
- `vehicle_scarcity`: 10k
- `route_disruption`: 15k
- `premium_sla`: 15k
- `normal_v4`: 10k

Manual command, DO NOT RUN UNTIL USER APPROVES:

```powershell
python -m src.learn.train_joint_torch `
  --config configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json `
  --output-dir models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608 `
  --init-from-joint-checkpoint models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt `
  --final-ppo-path models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/ppo_torch_joint_final_balanced_retention_ft_200k.pt `
  --final-dqn-path models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/dqn_torch_joint_final_balanced_retention_ft_200k.pt `
  --device cpu `
  --torch-num-threads 2 `
  --torch-num-interop-threads 1 `
  --skip-registry `
  --disable-trace-logging
```

Future offline eval command, DO NOT RUN UNTIL FINE-TUNE COMPLETES:

```powershell
python -m src.eval.evaluate_real_world_scenarios `
  --checkpoint models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/joint_torch_latest.pt `
  --scenario-dir configs/eval_scenarios `
  --output-dir models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios `
  --device cpu `
  --deterministic
```

Risks:
- Route/premium improvements may regress when retention stages dilute narrow repair pressure.
- Retention stages may not fully recover baseline, lead-time, or high-holding behavior.
- The high-holding reorder/holding tradeoff may remain fragile.
- Another full offline eval rerun is mandatory before registry candidate review.
