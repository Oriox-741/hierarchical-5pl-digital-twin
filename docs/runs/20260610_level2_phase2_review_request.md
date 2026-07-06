# Level 2 Phase 2 independent review request

Date: 2026-06-10

Status: review package only. No V2.3 config, training run, registry update, production update, baseline update, DB mutation, or checkpoint mutation is authorized by this file.

Current gate status:

- Independent review completed with verdict `PHASE2_PATCH_APPROVED_FOR_V2_3_CONFIG`.
- The verdict is recorded at `docs/runs/20260610_level2_phase2_review_verdict.md`.
- V2.3 config creation may proceed only through the TDD path in `docs/plans/20260610_level2_v2_3_pre_review_plan.md`.
- A structured verdict template is available at `docs/runs/20260610_level2_phase2_review_verdict_template.md`.

## Review objective

Review the Phase 1 causal hypothesis and Phase 2 local TDD reward patch before any V2.3 250k config is created or any training run is launched.

This review is required by `docs/goals/level2_research_autopilot_goal.txt`, step 4:

> If reward/config patch is needed, implement with TDD and independent review.

## Context

Production parent:

- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`

Relevant report:

- `docs/runs/20260610_level2_postmortem_v2_cycle_design.md`

Failed V-cycle evidence:

- V2: 4/8 PASS, failed premium, high_holding, lead_time, route.
- V2.1: 4/8 PASS, failed baseline, premium, high_holding, lead_time.
- V2.2: 5/8 PASS, failed high_holding, lead_time, vehicle; also failed route action-quality gate.

Diagnostic conclusion:

- Timing failures are driven by action 9, `hold:low_congestion:secondary_fleet:conservative`, remaining over-selected under useful-dispatch pressure.
- Route action-quality failure is driven by high-resilience dispatch pockets, especially action 41 and action 45. Action 41 was net-positive in V2.2 despite no-current/no-unassigned exposure; action 45 was already negative but still selected in a pure-failure pocket.

## Patch under review

Files:

- `src/act/env_5pl.py`
- `src/eval/scenario_metrics.py`
- `tests/act/test_reward_physics_contract.py`
- `tests/eval/test_real_world_scenario_evaluator.py`

Key code references:

- `src/act/env_5pl.py:1989` initializes `hold_timing_degradation_pressure`.
- `src/act/env_5pl.py:2176` computes hold degradation pressure only in the pressured hold branch.
- `src/act/env_5pl.py:2187` applies a bounded hold-under-lateness penalty cap of `0.16`.
- `src/act/env_5pl.py:2386` initializes `high_resilience_no_useful_work_credit_blocked`.
- `src/act/env_5pl.py:2739` blocks high-resilience route credits when `route_candidate_useful_work_factor < 0.50`.
- `src/act/env_5pl.py:2996` emits `hold_timing_degradation_pressure`.
- `src/act/env_5pl.py:3035` emits `high_resilience_no_useful_work_credit_blocked`.
- `src/eval/scenario_metrics.py:24` adds `hold_timing_degradation_pressure` to diagnostic aggregation.
- `src/eval/scenario_metrics.py:33` adds `high_resilience_no_useful_work_credit_blocked` to diagnostic aggregation.

## Requirements to check

1. The patch must target the diagnosed mechanisms, not blindly alter schedule or globally suppress dispatch.
2. The hold penalty change must affect pressured holds only, preserving calm/no-work hold behavior.
3. The route credit change must suppress no-useful-work high-resilience credits without penalizing useful high-resilience dispatch under route disruption.
4. Observation dimension, action count, and contract must remain unchanged: `physical_reality_v5_route_candidate_visibility`, obs 73, action 48.
5. Hard-blocker protections must not be weakened.
6. The diagnostic additions must surface the new mechanisms by action id and action family.
7. No protected paths may be mutated:
   - `models/registry/active_models.json`
   - `models/registry/models.jsonl`
   - `models/production`
   - `models/baselines`
   - `db`
   - source production checkpoint

## TDD evidence already collected

RED before reward patch:

```powershell
python -m unittest tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_no_work_high_resilience_dispatch_gets_no_route_resilience_credit tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_severe_timing_pressure_hold_penalty_exceeds_legacy_cap_without_affecting_calm_hold -v
```

Observed failures:

- `route_resilience_credit` was `0.08133333333333333` for no-work high-resilience dispatch.
- Severe timing-pressure hold penalty was exactly the legacy `0.1` cap.

GREEN after reward patch:

- Same focused command passed, 2 tests.
- `python -m unittest tests.act.test_reward_physics_contract -v` initially passed, 125 tests.

RED before eval diagnostic allowlist update:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_reward_component_diagnostics_are_reported_by_action_and_family -v
```

Observed failure:

- Missing `high_resilience_no_useful_work_credit_blocked` in diagnostic aggregation.

GREEN after eval diagnostic allowlist update:

- Same focused command passed.
- `python -m unittest tests.eval.test_real_world_scenario_evaluator -v` passed, 36 tests.

Additional verification:

- Added direct action-ID regression coverage after the initial review package:
  - `tests/act/test_reward_physics_contract.py:2878` verifies action 41 and action 45 block high-resilience route credits when no useful dispatch work is available.
  - `tests/act/test_reward_physics_contract.py:2926` verifies useful action 41 and action 45 dispatch still retains high-resilience route credit under strong route disruption.
- `python -m unittest tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_route_failure_actions_41_and_45_block_no_work_resilience_credit tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_useful_route_actions_41_and_45_keep_high_resilience_credit -v` passed, 2 tests.
- `python -m unittest tests.act.test_reward_physics_contract -v` passed after the added action-ID coverage, 127 tests.
- `python -m unittest tests.act.test_discrete_action_mapper_contract -v` passed, 1 test.
- `python -m unittest tests.eval.test_real_world_scenario_evaluator -v` passed, 36 tests.
- `python -m unittest tests.learn.test_train_joint_curriculum -v` passed, 31 tests.
- `python -m unittest tests.learn.test_curriculum_config -v` passed, 64 tests.
- `python -m unittest tests.eval.test_long_run_gate -v` passed, 13 tests.
- `python -m py_compile src\act\env_5pl.py src\eval\scenario_metrics.py src\learn\joint_metrics.py tests\act\test_reward_physics_contract.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py` passed.
- Production self-check gate passed with exit code 0.

## Post-patch diagnostic eval evidence

Fresh diagnostic evals were run after the Phase 2 reward patch. These are read-only policy evaluations on existing checkpoints, not training runs.

Eval directories:

- Production: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_level2_phase2_patch_diag_20260610_offline_scenarios`
- V2.2: `models/eval/joint_torch_v5_prod_stability_v2_2_250k_20260610_level2_phase2_patch_diag_20260610_offline_scenarios`

Fresh gate decision:

- Candidate: V2.2 phase2 patch diagnostic eval.
- Production comparison: production phase2 patch diagnostic eval.
- Decision: FAIL.
- Fatal failures: high_holding threshold fail, lead_time threshold fail, vehicle threshold fail, route dispatch-success regression from `0.862` to `0.829`.

Key attribution deltas visible after patch:

| Scenario/action | Post-patch diagnostic finding |
| --- | --- |
| V2.2 high_holding action 9 | `hold_under_lateness_pressure_penalty` mean `0.086780`, `hold_timing_degradation_pressure` mean `0.860051`, `dqn_local` mean `-0.148722`. |
| V2.2 lead_time action 9 | `hold_under_lateness_pressure_penalty` mean `0.083217`, `hold_timing_degradation_pressure` mean `0.804835`, `dqn_local` mean `-0.135591`. |
| V2.2 vehicle action 9 | `hold_under_lateness_pressure_penalty` mean `0.094934`, `hold_timing_degradation_pressure` mean `0.914763`, `dqn_local` mean `-0.142536`. |
| V2.2 route action 41 | `high_resilience_no_useful_work_credit_blocked` sum `192`, route resilience/adaptation credit sum `54.596880`, no-current/no-unassigned penalty sum `47.648150`, `dqn_local` mean `0.100197`. |
| V2.2 route action 45 | `high_resilience_no_useful_work_credit_blocked` sum `209`, route resilience/adaptation credit sum `0.000000`, no-current/no-unassigned penalty sum `45.766187`, `dqn_local` mean `-0.438934`. |

Review implication:

- The hold patch is active in the intended failed timing scenarios, but it also raises action 9 penalty in baseline/premium contexts. The reviewer should assess whether this is acceptable or likely to damage retention during V2.3 training.
- The route patch fully blocks no-work credit for action 45 and blocks no-work cases for action 41, but successful action 41 uses still retain enough credit for positive mean `dqn_local`. The reviewer should decide whether that is acceptable or whether action 41 needs a more specific rank/quality penalty.

## Reviewer questions

1. Is the hold penalty cap increase from `0.10` to `0.16` sufficiently targeted, or does it risk recreating V2.1-style under-dispatch/hold collapse in baseline or premium?
2. Should `hold_timing_degradation_pressure` include additional or fewer factors than demand degradation, holding cost, macro capacity pressure, capacity shock, and vehicle availability pressure?
3. Is suppressing all high-resilience route resilience/adaptation credit when `route_candidate_useful_work_factor < 0.50` correct, or should route-failure/no-unassigned/no-current-work cases be separated more finely?
4. Does the patch preserve useful high-resilience dispatch under strong route disruption, as covered by `test_strong_route_disruption_still_justifies_high_resilience`?
5. Are the added action 41/action 45 tests sufficient, or does action 41 still need a more specific rank/quality penalty test?
6. Is the patch adequate to justify a V2.3 250k config after review, or does it reveal a broader architecture decision requirement?

## Requested assessment

Return one of:

- `PHASE2_PATCH_APPROVED_FOR_V2_3_CONFIG`
- `PHASE2_PATCH_NEEDS_FIXES_BEFORE_CONFIG`
- `AUTOPILOT_NEEDS_ARCHITECTURE_DECISION`

No training should start until this independent review is complete and any required fixes are verified.
