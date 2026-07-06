# Level 2 Phase 3 review evidence manifest

Status: review evidence only. This file does not authorize V2.4 config creation,
training, offline eval, registry updates, production updates, baseline updates,
DB mutation, or checkpoint mutation.

Delegation boundary: the current thread has not received explicit authorization
for a Phase 3 independent reviewer. Do not spawn a reviewer until the user
explicitly authorizes it.

## Active Goal Constraints

Source: `docs/goals/level2_research_autopilot_goal.txt`

- Produce a clean experimental 1M candidate that passes offline scenarios and
  long-run gate, or stop only for a human-level architecture decision.
- Parent from production unless a later PASS gate explicitly approves another
  parent.
- Never update registry, `active_models.json`, `models.jsonl`, production,
  baselines, DB, or the source production checkpoint.
- Never use failed nextgen 1M, V2, V2.1, V2.2, routepremium failed 200k,
  aborted after_rewardfix, invalid 700k, v3/v4, or probe checkpoints as parent.
- Keep contract `physical_reality_v5_route_candidate_visibility`, observation
  dimension `73`, and action count `48`.
- Use fresh dirs only, `--skip-registry`, `--disable-trace-logging`, torch
  threads `2/1`.
- No 500k until a 250k candidate has `8/8 PASS` and long-run gate exit `0`.

## Phase 3 Review Package

- Review request:
  `docs/runs/20260610_level2_phase3_review_request.md`
- Verdict template:
  `docs/runs/20260610_level2_phase3_review_verdict_template.md`
- Postmortem and root-cause log:
  `docs/runs/20260610_level2_postmortem_v2_cycle_design.md`
- V2.4 pre-review plan:
  `docs/plans/20260610_level2_v2_4_pre_review_plan.md`

## V2.3 Artifacts

- Config:
  `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`
- Checkpoint:
  `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/joint_torch_latest.pt`
- Eval summary:
  `models/eval/joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios/scenario_summary.json`
- Production comparison summary:
  `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json`

## V2.3 Scenario Threshold Result

All V2.3 offline scenario threshold verdicts were `PASS`.

| Scenario | Verdict | Service | True lateness |
| --- | --- | ---: | ---: |
| `baseline_normal` | PASS | 0.964557 | 0.000000 |
| `demand_spike_volatility` | PASS | 0.955122 | 0.000063 |
| `high_holding_cost` | PASS | 0.973370 | 0.053119 |
| `lead_time_volatility` | PASS | 0.918215 | 0.090666 |
| `mixed_stress` | PASS | 0.819971 | 0.141767 |
| `premium_sla_pressure` | PASS | 0.981781 | 0.000000 |
| `route_disruption_congestion` | PASS | 0.989799 | 0.000000 |
| `vehicle_scarcity_capacity_shock` | PASS | 0.949415 | 0.024118 |

## V2.3 Long-Run Gate Failure Evidence

V2.3 failed the long-run gate on `route_disruption_congestion` action quality.

Route comparison:

- Production `dispatch_success_per_attempt`: `0.821930`
- V2.3 `dispatch_success_per_attempt`: `0.712948`
- V2.3 `dispatch_rate`: `0.514410`
- V2.3 route service/lateness: service `0.989799`, true lateness `0.000000`

V2.3 top route actions:

| Action | Count | Percentage |
| ---: | ---: | ---: |
| 32 | 2462 | 0.427431 |
| 16 | 1223 | 0.212326 |
| 20 | 852 | 0.147917 |
| 2 | 230 | 0.039931 |
| 45 | 148 | 0.025694 |

Action 32 failure pocket:

- Mapping: `dispatch`, `low_congestion`, `secondary_fleet`, `none`.
- Attempts: `2462`
- Successes: `1614`
- Failed no-ops: `848`
- Dispatch success ratio: `0.655565`
- `action_no_current_work_by_id["32"]`: `848`
- `action_no_unassigned_by_id["32"]`: `848`
- Failed dispatches by route: `low_congestion 848`, `high_resilience 1`
- Route success ratio by route: `low_congestion 0.667581`,
  `high_resilience 0.996094`, `shortest 1.000000`

Action 32 pre-patch reward-component means:

| Component | Mean |
| --- | ---: |
| `dqn_local` | 0.005758 |
| `no_current_or_unassigned_dispatch_penalty` | 0.083918 |
| `dispatch_feasibility_credit` | 0.046310 |
| `useful_dispatch_lateness_urgency_credit` | 0.059237 |
| `route_resilience_adaptation_credit` | 0.016566 |
| `route_candidate_alignment_credit` | 0.002081 |
| `route_candidate_useful_work_factor` | 0.655565 |
| `candidate_alignment_blocked_no_useful_work` | 0.344435 |

Interpretation for review:

- V2.3 restored the threshold failures from V2.2 but found a new route
  action-quality failure pocket.
- The failure is not the earlier high-resilience action 41/45 mechanism; V2.3
  route failed dispatches are overwhelmingly low-congestion action 32.
- No-work action 32 rows still retained residual low-congestion adaptation credit
  before the Phase 3 patch.
- The Phase 3 patch targets this specific leakage without penalizing useful
  action 32 low-congestion dispatch.

## Phase 3 Code Changes Under Review

Source files:

- `src/act/env_5pl.py`
  - Adds `low_congestion_no_useful_work_credit_blocked`.
  - For low-congestion dispatches with route adaptation pressure and
    `route_candidate_useful_work_factor < 0.50`, sets the blocker to `1.0` and
    zeroes low-congestion adaptation credit.
  - The blocker is independent of whether residual low-congestion adaptation
    credit was still positive after route-failure suppression.
- `src/eval/scenario_metrics.py`
  - Adds `low_congestion_no_useful_work_credit_blocked` to
    `DIAGNOSTIC_REWARD_COMPONENTS`.

Test files:

- `tests/act/test_reward_physics_contract.py`
  - `test_no_work_action32_low_congestion_dispatch_blocks_route_adaptation_credit`
  - `test_no_work_low_congestion_blocker_reports_when_route_failure_zeroes_credit`
  - `test_useful_action32_low_congestion_dispatch_keeps_route_adaptation_credit`
- `tests/eval/test_real_world_scenario_evaluator.py`
  - `test_reward_component_diagnostics_are_reported_by_action_and_family`

## Verification Evidence

Full affected verification after the Phase 3 patch:

- `python -m unittest tests.act.test_reward_physics_contract -v`: PASS, 130 tests.
- `python -m unittest tests.eval.test_real_world_scenario_evaluator -v`: PASS, 36 tests.
- `python -m unittest tests.learn.test_train_joint_curriculum tests.eval.test_long_run_gate -v`: PASS, 44 tests.
- `python -m py_compile src\act\env_5pl.py src\eval\scenario_metrics.py tests\act\test_reward_physics_contract.py tests\eval\test_real_world_scenario_evaluator.py`: PASS.

Focused review smoke test:

```powershell
python -m unittest `
  tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_no_work_low_congestion_blocker_reports_when_route_failure_zeroes_credit `
  tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_no_work_action32_low_congestion_dispatch_blocks_route_adaptation_credit `
  tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_useful_action32_low_congestion_dispatch_keeps_route_adaptation_credit `
  tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_reward_component_diagnostics_are_reported_by_action_and_family `
  -v
```

Result: PASS, 4 tests.

## Protected-Path Evidence

Current protected hashes:

- `models/registry/active_models.json`:
  `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl`:
  `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Production checkpoint:
  `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

V2.4 artifact freshness:

- `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`:
  absent
- `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610`:
  absent
- `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios`:
  absent

Process check:

- No running `python.exe` process was present during the latest guardrail sweep.

## Required Review Verdict

The independent reviewer must return exactly one verdict:

- `PHASE3_PATCH_APPROVED_FOR_V2_4_CONFIG`
- `PHASE3_PATCH_NEEDS_FIXES_BEFORE_CONFIG`
- `AUTOPILOT_NEEDS_ARCHITECTURE_DECISION`

Do not create V2.4 config, train, or run offline eval before an approved Phase 3
review verdict.
