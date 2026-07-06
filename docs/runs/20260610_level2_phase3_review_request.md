# Level 2 Phase 3 independent review request

Status: review package only. No V2.4 config, training run, offline scenario eval,
registry update, production update, baseline update, DB mutation, or checkpoint
mutation is authorized by this file.

Delegation status: the current thread already used the one GPT-5.5 independent
reviewer explicitly authorized for Phase 2. A new Phase 3 reviewer must not be
spawned until the user explicitly authorizes that reviewer.

Current gate status:

- V2.3 250k training and eval completed from the production parent only.
- V2.3 offline scenario thresholds were `8/8 PASS`.
- V2.3 long-run gate failed route action-quality.
- Independent Phase 3 review completed with verdict
  `PHASE3_PATCH_APPROVED_FOR_V2_4_CONFIG`.
- No 500k or 1M horizon is authorized.

Reviewer scope:

- Read this request.
- Read `docs/runs/20260610_level2_phase3_review_evidence_manifest.md`.
- Read `docs/runs/20260610_level2_phase3_review_verdict_template.md`.
- Read `docs/runs/20260610_level2_postmortem_v2_cycle_design.md`, especially the
  V2.3 execution, route failure root cause, and Phase 3 patch sections.
- Inspect the changed files and tests needed to judge the Phase 3 patch.
- Inspect existing production, V2, V2.1, V2.2, and V2.3 evidence as needed.

Reviewer restrictions:

- Do not train.
- Do not run offline eval.
- Do not update registry.
- Do not mutate production.
- Do not mutate baselines.
- Do not mutate DB.
- Do not mutate checkpoints.
- Do not create a V2.4 config. If the verdict is approved, control returns to the
  main worker to create the config through the TDD path.

## V2.3 result summary

V2.3 artifacts:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`
- Checkpoint: `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/joint_torch_latest.pt`
- Eval: `models/eval/joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios/scenario_summary.json`

V2.3 passed all scenario thresholds but failed the long-run gate:

- Route dispatch success regressed from production `0.822` to V2.3 `0.713`.
- No-current-work/no-unassigned gate total increased from production `171` to V2.3 `1697`.
- Top no-work/no-unassigned action shifted to action 32 with `42.7%` route scenario action share.

Action 32 mapping:

- `dispatch`
- `low_congestion`
- `secondary_fleet`
- `none`

V2.3 route action 32 evidence:

- Attempts: `2462`
- Successes: `1614`
- Failed no-ops: `848`
- Success ratio: `0.6556`
- `action_no_current_work_by_id["32"]`: `848`
- `action_no_unassigned_by_id["32"]`: `848`
- Mean `dqn_local`: `0.005758`
- Mean `no_current_or_unassigned_dispatch_penalty`: `0.083918`
- Mean `route_resilience_adaptation_credit`: `0.016566`
- Mean `route_candidate_useful_work_factor`: `0.655565`
- Mean `candidate_alignment_blocked_no_useful_work`: `0.344435`

Interpretation:

- V2.3 solved the earlier threshold failures and reduced the action 41/45
  high-resilience route failure pocket.
- The route action-quality failure moved to a low-congestion action 32 pocket.
- No-work action 32 rows still retained residual low-congestion adaptation credit.
- This is a targeted reward leak, not sufficient evidence for blind schedule changes.

## Phase 3 patch under review

Changed files:

- `src/act/env_5pl.py`
- `src/eval/scenario_metrics.py`
- `tests/act/test_reward_physics_contract.py`
- `tests/eval/test_real_world_scenario_evaluator.py`

Evidence manifest:

- `docs/runs/20260610_level2_phase3_review_evidence_manifest.md`

Patch behavior:

- Add `low_congestion_no_useful_work_credit_blocked`.
- For low-congestion dispatches under route adaptation pressure, when
  `route_candidate_useful_work_factor < 0.50`, set the blocker diagnostic to `1.0`
  and zero `low_congestion_adaptation_credit`.
- The blocker diagnostic is set even when another route gate, such as route-failure
  suppression, has already reduced low-congestion adaptation credit to zero. This
  keeps the diagnostic aligned with the causal no-useful-work condition rather than
  with the residual credit amount.
- Preserve useful action 32 low-congestion dispatch route adaptation credit.
- Aggregate the new diagnostic by action id and action family in offline summaries.

TDD evidence:

- RED before reward patch:

```powershell
python -m unittest tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_no_work_action32_low_congestion_dispatch_blocks_route_adaptation_credit tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_useful_action32_low_congestion_dispatch_keeps_route_adaptation_credit -v
```

Failure: missing `low_congestion_no_useful_work_credit_blocked`.

- RED before eval diagnostic allowlist patch:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_reward_component_diagnostics_are_reported_by_action_and_family -v
```

Failure: missing aggregated `low_congestion_no_useful_work_credit_blocked`.

- GREEN after patch:
  - The two focused reward tests passed.
  - The focused eval diagnostic test passed.
- Supplemental RED/GREEN after review-readiness audit:
  - `python -m unittest tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_no_work_low_congestion_blocker_reports_when_route_failure_zeroes_credit -v`
  - Failed before the supplemental patch because the blocker stayed `0.0` when
    `route_failure_credit_suppression` already zeroed low-congestion adaptation credit.
  - Passed after removing the accidental residual-credit-positive precondition from
    the blocker diagnostic.

Verification:

| Command | Result |
| --- | --- |
| `python --version` | `Python 3.12.9` |
| `python -m unittest tests.act.test_reward_physics_contract -v` | PASS, 130 tests |
| `python -m unittest tests.eval.test_real_world_scenario_evaluator -v` | PASS, 36 tests |
| `python -m unittest tests.learn.test_train_joint_curriculum tests.eval.test_long_run_gate -v` | PASS, 44 tests |
| `python -m py_compile src\act\env_5pl.py src\eval\scenario_metrics.py tests\act\test_reward_physics_contract.py tests\eval\test_real_world_scenario_evaluator.py` | PASS |

## Proposed next V-cycle if approved

Use `docs/plans/20260610_level2_v2_4_pre_review_plan.md`.

The proposed V2.4 cycle should isolate the reward patch:

- Parent from production only.
- Create a fresh V2.4 250k config only after approval.
- Keep the V2.3 schedule and baseline-anchor policy unchanged unless the reviewer
  rejects that isolation strategy.
- Run all pre-training tests and protected-path checks.
- Train only the approved 250k candidate with `--skip-registry`,
  `--disable-trace-logging`, torch threads `2/1`, and fresh output dirs.
- Evaluate, run the long-run gate, and stop if any threshold or action-quality gate fails.

## Review questions

1. Does the V2.3 evidence support action 32 low-congestion no-work credit leakage as
   a causal route action-quality mechanism?
2. Is `route_candidate_useful_work_factor < 0.50` the right gate for suppressing
   low-congestion route adaptation credit?
3. Does the patch preserve useful action 32 low-congestion route credit?
4. Is a controlled V2.4 250k run with the same schedule as V2.3 justified, or does
   the failure point to route-rank/candidate-quality penalties, algorithm redesign,
   evaluator threshold policy, vector env changes, or a new model class?

## Required verdict

Reviewer must return exactly one verdict:

- `PHASE3_PATCH_APPROVED_FOR_V2_4_CONFIG`
- `PHASE3_PATCH_NEEDS_FIXES_BEFORE_CONFIG`
- `AUTOPILOT_NEEDS_ARCHITECTURE_DECISION`

Verdict meanings:

- `PHASE3_PATCH_APPROVED_FOR_V2_4_CONFIG`: the main worker may create the V2.4
  config through TDD and execute the 250k gate path.
- `PHASE3_PATCH_NEEDS_FIXES_BEFORE_CONFIG`: the main worker must apply only the
  requested fixes with tests, then request review again.
- `AUTOPILOT_NEEDS_ARCHITECTURE_DECISION`: stop and produce the architecture
  decision report.
