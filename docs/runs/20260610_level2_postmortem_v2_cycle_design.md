# Level 2 postmortem: production vs V2/V2.1/V2.2/V2.3

Date: 2026-06-10

Status: latest-first research log. V2.3 250k completed, passed offline
scenario thresholds, and failed the long-run route action-quality gate. The
Phase 3 code-only patch is prepared for review; no V2.4 config, training run, or
offline eval has been started.

## V2.4 post-training audit

V2.4 was created only after the Phase 3 independent review returned
`PHASE3_PATCH_APPROVED_FOR_V2_4_CONFIG`.

Artifacts:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`
- Checkpoint dir: `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610`
- Standard eval dir: `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios`
- Diagnostic eval dir: `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios_stuckdiag`
- Audit report: `docs/runs/20260610_level2_v2_4_post_training_audit.md`

Outcome:

- Training exited `0`.
- Offline eval exited `0`.
- Standard long-run gate decision: `FAIL`.
- Diagnostic long-run gate decision: `FAIL`.
- V2.4 is not authorized for 500k.

Gate fatal failure:

- `baseline_normal`: scenario threshold failure, `stuck assigned/in-transit orders present: 1`.

Phase 3 target result:

- V2.3 route dispatch success was `0.712948`.
- V2.4 route dispatch success was `1.000000`.
- The prior route action 32 no-current/no-unassigned failed-noop pocket is gone in V2.4 route eval.
- Premium dispatch success remained recovered at `0.999124`.

Diagnostic finding:

- The failing baseline order was `in_transit` at final snapshot time `86400.0`.
- Its pickup time was `86100.0`; it had been in transit for five minutes.
- Its due time was `100178.20358614686`, leaving `13778.203586146861` seconds to due.
- `is_late` was `false`.

Interpretation:

- The remaining failure is not a route reward/action-quality failure.
- It is an evaluator semantics issue: current `stuck_assigned_in_transit_orders`
  counts any active assigned or in-transit order at the finite horizon, even if
  the order is healthy, recently picked up, and not due until after the horizon.
- Changing this would be an evaluator threshold policy decision, so the autopilot
  must stop rather than silently making a V2.5 config or reclassifying V2.4.

Classification: AUTOPILOT_BLOCKED_NEEDS_ARCHITECTURE_DECISION

Ordering note: the latest V2.3/Phase 3 update appears first. Historical
production/V2/V2.1/V2.2 audit evidence follows afterward.

## V2.3 250k execution and gate result

Completed on 2026-06-10 after the independent Phase 2 review returned
`PHASE2_PATCH_APPROVED_FOR_V2_3_CONFIG`.

Artifacts:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`
- Checkpoint dir: `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610`
- Final checkpoint: `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/joint_torch_latest.pt`
- Eval dir: `models/eval/joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios`

V2.3 training exited `0` from the production parent only:

```powershell
python -m src.learn.train_joint_torch `
  --config configs\training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json `
  --output-dir models\checkpoints\joint_torch_v5_prod_stability_v2_3_250k_20260610 `
  --init-from-joint-checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt `
  --final-ppo-path models\checkpoints\joint_torch_v5_prod_stability_v2_3_250k_20260610\ppo_torch_joint_final_stability_v2_3_250k_20260610.pt `
  --final-dqn-path models\checkpoints\joint_torch_v5_prod_stability_v2_3_250k_20260610\dqn_torch_joint_final_stability_v2_3_250k_20260610.pt `
  --device cpu `
  --torch-num-threads 2 `
  --torch-num-interop-threads 1 `
  --skip-registry `
  --disable-trace-logging
```

V2.3 offline eval exited `0` and all eight scenario threshold verdicts were `PASS`.
The long-run gate failed with route action-quality fatal failures, so V2.3 is not
a clean 250k candidate and no 500k or 1M horizon is authorized.

Gate command:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios\scenario_summary.json `
  --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json `
  --candidate-label stability_v2_3_250k
```

Gate decision: FAIL.

Fatal route failures:

- `route_disruption_congestion` dispatch success regressed from production `0.822` to V2.3 `0.713`.
- No-current-work/no-unassigned exposure exploded from production gate total `171` to V2.3 gate total `1697`.
- Top-action no-current/no-unassigned explosion moved to action `32`, with `1696` combined no-current/no-unassigned counts and `42.7%` scenario action share.

V2.3 did restore the earlier failed timing/vehicle thresholds:

| Scenario | Verdict | Service | True lateness |
| --- | ---: | ---: | ---: |
| baseline_normal | PASS | 0.9646 | 0.0000 |
| demand_spike_volatility | PASS | 0.9551 | 0.0001 |
| high_holding_cost | PASS | 0.9734 | 0.0531 |
| lead_time_volatility | PASS | 0.9182 | 0.0907 |
| mixed_stress | PASS | 0.8200 | 0.1418 |
| premium_sla_pressure | PASS | 0.9818 | 0.0000 |
| route_disruption_congestion | PASS | 0.9898 | 0.0000 |
| vehicle_scarcity_capacity_shock | PASS | 0.9494 | 0.0241 |

## V2.3 route failure root cause

The V2.3 route failure is not a repeat of the V2.2 action 41/45 high-resilience
mechanism. The Phase 2 patch worked on that pocket: V2.3 route action 41 had only
`30` attempts with `1` failed no-op, and action 45 was no longer the dominant
route failure.

The new failure shifted to action 32:

- Action 32 mapping: `dispatch`, `low_congestion`, `secondary_fleet`, `none`.
- V2.3 route top actions: action 32 `2462` uses (`42.7%`), action 16 `1223`
  uses (`21.2%`), action 20 `852` uses (`14.8%`).
- Action 32 route attempts: `2462`; successes: `1614`; failed no-ops: `848`;
  success ratio: `0.6556`.
- V2.3 route failed dispatches by route: `low_congestion 848`,
  `high_resilience 1`.
- V2.3 route success ratio by route: `low_congestion 0.6676`,
  `high_resilience 0.9961`, `shortest 1.0000`.

Action 32 reward attribution before the Phase 3 patch:

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

Causal interpretation:

- The V2.3 config did not primarily fail because baseline/premium/timing/vehicle
  pressure was underweighted; those scenario thresholds passed.
- The route policy found a new mixed low-congestion pocket. Successful action 32
  uses earned legitimate dispatch and timing credits, but no-useful-work action 32
  rows still retained low-congestion adaptation credit after dampening.
- Because the aggregate action 32 local reward was only slightly positive, the
  residual low-congestion no-work adaptation credit is a plausible causal leak.
- The next fix should therefore be a low-congestion no-useful-work route-credit
  block, not a blind schedule tweak or a broad penalty against useful low-congestion
  dispatch.

## Phase 3 local TDD reward patch status

Completed locally on 2026-06-10 after the V2.3 gate failure. No V2.4 config,
training run, offline scenario eval, registry write, production mutation, baseline
mutation, DB mutation, or checkpoint mutation was started.

Patch hypothesis:

- Low-congestion route adaptation credit should require useful route/dispatch work,
  just as the Phase 2 high-resilience patch required useful work for route credit.
- Useful action 32 low-congestion dispatch should retain route adaptation credit.
- No-work action 32 low-congestion dispatch should emit a blocker diagnostic and
  receive zero low-congestion adaptation, zero route-resilience-adaptation, and zero
  route-adaptation reward/credit.

Files changed:

- `src/act/env_5pl.py`
  - Added `low_congestion_no_useful_work_credit_blocked`.
  - Zeroed `low_congestion_adaptation_credit` for low-congestion dispatches when
    route adaptation pressure exists and `route_candidate_useful_work_factor < 0.50`.
  - Preserved useful low-congestion route adaptation credit.
- `src/eval/scenario_metrics.py`
  - Added `low_congestion_no_useful_work_credit_blocked` to reward-component
    diagnostic aggregation.
- `tests/act/test_reward_physics_contract.py`
  - Added no-work action 32 coverage for low-congestion credit blocking.
  - Added no-work action 32 route-failure coverage proving the blocker diagnostic
    remains active even when route-failure suppression already zeroes residual
    adaptation credit.
  - Added useful action 32 coverage to preserve low-congestion route credit.
- `tests/eval/test_real_world_scenario_evaluator.py`
  - Added action/family diagnostic aggregation coverage for
    `low_congestion_no_useful_work_credit_blocked`.

TDD evidence:

- RED before reward patch:
  - `python -m unittest tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_no_work_action32_low_congestion_dispatch_blocks_route_adaptation_credit tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_useful_action32_low_congestion_dispatch_keeps_route_adaptation_credit -v`
  - Failed as intended with missing `low_congestion_no_useful_work_credit_blocked`.
- RED before eval diagnostic allowlist update:
  - `python -m unittest tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_reward_component_diagnostics_are_reported_by_action_and_family -v`
  - Failed as intended with missing aggregated `low_congestion_no_useful_work_credit_blocked`.
- GREEN after patch:
  - Both focused commands passed.
- Supplemental RED/GREEN after review-readiness audit:
  - `python -m unittest tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_no_work_low_congestion_blocker_reports_when_route_failure_zeroes_credit -v`
  - Failed before the supplemental patch because
    `low_congestion_no_useful_work_credit_blocked` remained `0.0` when
    `route_failure_credit_suppression` had already reduced low-congestion
    adaptation credit to zero.
  - Passed after removing the residual-credit-positive condition from the blocker
    diagnostic.

Verification after Phase 3 patch:

| Command | Result |
| --- | --- |
| `python --version` | `Python 3.12.9` |
| `python -m unittest tests.act.test_reward_physics_contract -v` | PASS, 130 tests |
| `python -m unittest tests.eval.test_real_world_scenario_evaluator -v` | PASS, 36 tests |
| `python -m unittest tests.learn.test_train_joint_curriculum tests.eval.test_long_run_gate -v` | PASS, 44 tests |
| `python -m py_compile src\act\env_5pl.py src\eval\scenario_metrics.py tests\act\test_reward_physics_contract.py tests\eval\test_real_world_scenario_evaluator.py` | PASS |

Protected-path proof after Phase 3:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Source production checkpoint SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
- Protected directory profile:
  - `models/production`: 4 files, 15040339 bytes, max write ticks `639165695318235482`
  - `models/baselines`: 2692 files, 7392576274 bytes, max write ticks `639152637214676655`
  - `db`: 7 files, 28330 bytes, max write ticks `639147968118015913`
- V2.4 artifact freshness check:
  - `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`: absent
  - `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610`: absent
  - `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios`: absent
- `python.exe` process check: no running Python process.

## Next V-cycle: V2.4 review gate before config

The next cycle should be controlled rather than exploratory:

- Require independent review of the V2.3 failure analysis and Phase 3 patch before
  creating any V2.4 config.
- If approved, create a fresh V2.4 250k config from the production parent only.
- Keep the V2.3 schedule and baseline-anchor policy unchanged for V2.4 unless the
  reviewer explicitly rejects that isolation strategy. The experiment should test
  the low-congestion no-work reward fix, not confound it with schedule tuning.
- Run the same pre-training tests, protected-path checks, training command pattern,
  offline eval, and long-run gate.
- Stop again without 500k if V2.4 fails any scenario threshold, hard blocker, or
  long-run gate action-quality comparison.

Prepared next-review artifacts:

- `docs/runs/20260610_level2_phase3_review_request.md`
- `docs/runs/20260610_level2_phase3_review_evidence_manifest.md`
- `docs/runs/20260610_level2_phase3_review_verdict_template.md`
- `docs/runs/20260610_level2_phase3_review_verdict.md`
- `docs/plans/20260610_level2_v2_4_pre_review_plan.md`

Required next gate:

- Independent Phase 3 review completed with verdict
  `PHASE3_PATCH_APPROVED_FOR_V2_4_CONFIG`.
- A fresh V2.4 250k config may now be created only through the TDD path in
  `docs/plans/20260610_level2_v2_4_pre_review_plan.md`.

Classification: LEVEL2_AUTOPILOT_RESEARCH_CONTINUES

## V2.4 500k Continuation Stop

Updated on 2026-06-10 after the evaluator-semantics resolution and 500k continuation audit.

The V2.4 250k candidate passed after refining `stuck_assigned_in_transit_orders` semantics:

- Candidate eval: `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios_semantics_v2_20260610`
- Production comparator eval: `models/eval/joint_torch_v5_balanced_retention_ft_200k_20260608_offline_scenarios_semantics_v2_20260610`
- Gate result: `docs/runs/20260610_level2_v2_4_semantics_gate_result.json`
- Gate decision: PASS, exit 0

The approved 250k checkpoint was then used only for the allowed 500k ladder continuation:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_4_500k_20260610.json`
- Resume parent: `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610/joint_torch_latest.pt`
- Output dir: `models/checkpoints/joint_torch_v5_prod_stability_v2_4_500k_20260610`
- Candidate eval: `models/eval/joint_torch_v5_prod_stability_v2_4_500k_20260610_offline_scenarios_20260610`
- Gate result: `docs/runs/20260610_level2_v2_4_500k_gate_result.json`
- Gate decision: FAIL, exit 2

Fatal 500k failures:

- Premium SLA service fell to `0.811 < 0.920`.
- Route disruption dispatch success regressed from production `0.862` to `0.622`.
- High-holding no-current/no-unassigned total rose from production `11` to `125`.
- Route disruption no-current/no-unassigned total rose from production `868` to `2930`.
- Route disruption top-action explosion: action 45, count `2350`, production count `92`, top percentage `0.236`.

Periodic checkpoint diagnostics show the instability starts before the terminal checkpoint:

| Step | Premium service | Premium missed useful dispatch | Route dispatch success | Route no-work total | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 250k | 0.988 | 0.128 | 1.000 | 0 | PASS |
| 300k | 0.913 | 0.361 | 0.983 | 92 | FAIL |
| 400k | 0.356 | 0.985 | 0.858 | 743 | FAIL |
| 500k | 0.811 | 0.528 | 0.622 | 2930 | FAIL |

Causal conclusion:

- The 250k stuck-order failure was evaluator semantics and is resolved.
- The 500k failure is continuation instability, not stuck-order semantics.
- The trainer resumes model/optimizer state but starts with `replay_buffer=empty`; the DQN has no replay retention or teacher retention for the passing 250k behavior.
- The next V-cycle requires an architecture decision before more training.

Architecture decision report:

- `docs/runs/20260610_level2_v2_4_500k_architecture_decision_report.md`

Post-training audit:

- `docs/runs/20260610_level2_v2_4_500k_post_training_audit.md`

Current classification: AUTOPILOT_BLOCKED_NEEDS_ARCHITECTURE_DECISION

## Inputs audited

- Production eval: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json`
- V2 eval: `models/eval/joint_torch_v5_prod_stability_v2_250k_20260609_offline_scenarios/scenario_summary.json`
- V2.1 eval: `models/eval/joint_torch_v5_prod_stability_v2_1_250k_20260609_offline_scenarios/scenario_summary.json`
- V2.2 eval: `models/eval/joint_torch_v5_prod_stability_v2_2_250k_20260610_offline_scenarios/scenario_summary.json`
- Train metrics for production, V2, V2.1, and V2.2 checkpoint directories.
- Reward/action code inspected in `src/act/env_5pl.py` and action mapping in `src/act/discrete_action_mapper.py`.

All three V-cycles used the production parent and the same v5 route-candidate-visibility contract. The major V2.2 change was curriculum shaping plus baseline anchoring, not a new architecture or new parent.

## Pass matrix

| Run | Pass count | Failed scenarios |
| --- | ---: | --- |
| Production | 8/8 | none |
| V2 | 4/8 | premium, high_holding, lead_time, route |
| V2.1 | 4/8 | baseline, premium, high_holding, lead_time |
| V2.2 | 5/8 | high_holding, lead_time, vehicle |

V2.2 also failed the long-run/action-quality gate even though route_disruption was scenario-threshold PASS, because route no-current/no-unassigned exposure and action concentration were too high.

## High-level movement across V-cycles

V2 preserved baseline but introduced timing and route weakness. It under-dispatched in the hard timing scenarios and route disruption: high_holding dispatch rate 0.340, lead_time 0.403, route 0.347. Route failed with true_lateness_pressure 0.234.

V2.1 fixed route scenario status but overcorrected toward holding. Baseline service collapsed to 0.676 with lateness 0.193, premium collapsed to service 0.053 with lateness 0.501, and lead_time collapsed to service 0.051 with lateness 0.511. Premium was dominated by action 16, `hold:high_resilience:secondary:none`, at 64.6%.

V2.2 restored baseline and premium by reintroducing baseline anchoring and interleaving normal/premium stages. It shifted away from V2.1's action-16 collapse and toward action 9, `hold:low_congestion:secondary:conservative`, plus action 29, `dispatch:shortest:primary:conservative`. This recovered normal and premium service, but left a conservative hold basin under lateness/holding/capacity stress.

## V2.2 scenario failures

| Scenario | Production | V2.2 | Dominant V2.2 action issue |
| --- | --- | --- | --- |
| high_holding | service 0.943, lateness 0.000, dispatch 0.387 | service 0.826, lateness 0.183, dispatch 0.406 | action 9 at 28.0%, action 16 at 9.3%; dispatch success was 0.999, so the failure is not no-work dispatch. It is too much hold under useful dispatch pressure. |
| lead_time | service 0.936, lateness 0.000, dispatch 0.455 | service 0.908, lateness 0.252, dispatch 0.347 | action 9 at 37.2%; dispatch success was 1.000. This is the clearest hold-under-lateness failure. |
| vehicle_scarcity | service 0.914, lateness 0.000, dispatch 0.731 | service 0.826, lateness 0.011, dispatch 0.664 | action 29 at 32.6% is useful, but action 9 still consumes 25.4% of decisions. Capacity shock needs production-like dispatch breadth and fewer conservative holds. |
| route action quality | route scenario PASS, failed_noop 86, no_unassigned 85, high_resilience_not_best 411 | route scenario PASS, failed_noop 459, no_unassigned 456, high_resilience_not_best 1663 | action 41 at 16.2% with 933 attempts, 192 failed/no-current and 189 no-unassigned; action 45 had 209 attempts and 0 successes. |

## Root-cause matrix

| Area | Confirmed evidence | Current interpretation |
| --- | --- | --- |
| Baseline recovery | V2.1 baseline failed at service 0.676/lateness 0.193. V2.2 baseline passed at service 0.944/lateness 0.000. | Baseline anchoring plus normal-stage interleaving successfully prevented V2.1's hold collapse. |
| Premium recovery | V2.1 premium failed at service 0.053/lateness 0.501. V2.2 premium passed at service 0.932/lateness 0.002. | The V2.2 mix restored enough useful primary dispatch and avoided the catastrophic action-16 premium hold mode. It still held 62.9%, with action 9 at 30.8%, so premium is recovered but fragile. |
| High_holding failure | V2.2 high_holding had service 0.826/lateness 0.183 with dispatch success 0.999. Action 9 was 28.0%. | Not a dispatch feasibility issue. The policy waits too often when holding cost/lateness pressure should force useful dispatch. |
| Lead_time failure | V2.2 lead_time had service 0.908/lateness 0.252 with dispatch success 1.000. Action 9 was 37.2%. | Same root as high_holding, stronger: action 9 is treated as a safe conservative response when lateness pressure requires dispatch. |
| Vehicle failure | V2.2 vehicle_scarcity dispatch was 0.664 vs production 0.731, with action 9 at 25.4%. | The model partially uses action 29 correctly but still sacrifices too many dispatch opportunities to the conservative hold basin. |
| Route action-quality failure | V2.2 route had high_resilience selected 2671 times vs production 657, high_resilience_not_best 1663 vs production 411, and failed/no-current 459 vs production 86. | The route scenario is service-healthy but action-quality-bad. High-resilience dispatch pockets, especially action 41 and action 45, are not being constrained enough by no-current/no-unassigned and candidate-rank quality. |
| Reward balance | `hold_under_lateness_pressure_penalty` is capped at 0.10. Useful dispatch gets feasibility/progress/timing credits; primary fleet can also receive timing credit. | The timing reward likely has a weak hold penalty relative to the learned conservative hold prior. However, current eval summaries do not expose per-action reward-component sums, so this remains a hypothesis pending diagnostics. |
| Route reward attribution | Candidate alignment and mismatch logic exists, with mismatch penalty capped at 0.06 and route credits gated on useful work/service/route failure. | Need per-action reward-component attribution to tell whether action 41/45 are receiving residual route credit, insufficient penalties, or simply being learned from a distribution where train success looks better than eval success. |
| Training telemetry gap | V2.2 training `action_no_unassigned_by_id` was empty in the final snapshot while eval route had no_unassigned 456. Training route-stage action 41 success appeared 0.994, while eval route action 41 success was 0.794. | Training instrumentation/distribution is not exposing the eval action-quality failure clearly enough. Do not train again before adding diagnostics. |

## Why V2.2 restored baseline and premium but failed timing/vehicle/route quality

V2.2 restored baseline and premium because it added a 0.1 baseline anchor and reinserted normal stages around the stress stages. That made the final policy retain normal-service behavior and prevented the V2.1 premium collapse. In premium, action 29 and other useful dispatches were enough to pass, and the no-work guards kept premium no-current exposure low.

The same anchoring also preserved a conservative hold family. Action 9 is acceptable in production high_holding and baseline when no lateness materializes, but V2.2 generalized it too broadly into lead_time, high_holding, vehicle_scarcity, and premium. In lead_time and high_holding, dispatch attempts were almost always successful when made, so the failures are not caused by impossible dispatches. They are caused by missed useful dispatch opportunities.

Route disruption is different. It did not fail service/lateness, but it failed action quality. V2.2 greatly increased high-resilience route selection, including second-best high-resilience selections and high-resilience dispatches with no current/unassigned work. Action 41 is a mixed pocket: many successes, but also 192 failed/no-current attempts. Action 45 is worse: 209 attempts, 0 successes. Existing route gates reduce credit on failures, but the current output does not prove whether penalties are too weak, credits leak through successes, or training never sees the same no-unassigned distribution.

## Next V-cycle design, pending approval

The next cycle should be diagnostics-first. Do not create V2.3 training config or launch a learner until Phase 0 and Phase 1 are complete.

### Phase 0: read-only diagnostic patch

Add TDD-covered offline diagnostic aggregation. The diagnostic summary should aggregate these reward components by scenario, action id, and action family:

- `dqn_local`
- `hold_under_lateness_pressure`
- `hold_under_lateness_pressure_penalty`
- `useful_dispatch_lateness_urgency_credit`
- `primary_fleet_timing_justification_credit`
- `secondary_fleet_lateness_exposure`
- `no_current_or_unassigned_dispatch_penalty`
- `dispatch_feasibility_credit`
- `dispatch_progress_credit`
- `route_resilience_credit`
- `route_resilience_adaptation_credit`
- `route_candidate_alignment_credit`
- `route_candidate_mismatch_penalty`
- `selected_route_candidate_score_gap`
- `route_candidate_useful_work_factor`
- candidate-alignment blocked flags

Also fix/extend training diagnostics so no-unassigned dispatch counts are not empty when reward/eval data shows no-unassigned exposure. This is diagnostic only and must not change production registries or source checkpoints.

### Phase 1: diagnostic eval rerun, no training

Run fresh offline diagnostic eval directories for production and V2.2. Compare:

- action 9 in high_holding, lead_time, vehicle, premium, and baseline
- action 29 in vehicle/high_holding/lead_time
- action 41 and action 45 in route_disruption
- route candidate-rank and mismatch penalties for all high-resilience dispatches
- hold penalty vs useful dispatch credit in timing scenarios

Decision gates:

- If action 9 has little or no hold-under-lateness penalty in failing scenarios, patch activation/pressure calculation.
- If action 9 has the penalty but is still selected heavily, increase the penalty or make it service/lateness-threshold-sensitive while preserving baseline/premium guardrails.
- If action 41/45 receive positive route credit while no-current/no-unassigned or candidate-rank mismatch is present, tighten route credit gates and/or mismatch/no-work penalties.
- If reward attribution is already strongly negative but policy still selects the actions, stop and escalate to architecture discussion: action space, observation credit assignment, or DQN/off-policy learning may be the limiting factor.

### Phase 2: minimal reward/config patch

Only after diagnostics identify the failing mechanism:

- Timing patch should target hold under useful dispatch pressure, not generic holds. It must preserve production-like baseline and premium behavior.
- Route patch should target high-resilience dispatch action quality under route disruption, especially action 41/45 no-current/no-unassigned and second-best route selection. It should not penalize successful production-like high-resilience dispatch globally.
- Config changes should be limited to preserving V2.2's baseline/premium anchors and modestly increasing failed-scenario exposure. Do not use schedule changes as the primary fix.

### Phase 3: V2.3 250k gate only

If Phase 2 passes tests, create a fresh V2.3 250k config from the production parent. Required gates:

- unit tests for reward diagnostics and any reward patch
- offline scenarios 8/8 PASS
- long-run gate exit 0
- route no-current/no-unassigned and top-action concentration no worse than production thresholds

No 500k/1M run is allowed until the 250k candidate passes both offline and long-run gates.

## Recommendation

Proceed with Phase 0 diagnostics and Phase 1 diagnostic eval before any new training. The evidence does not support blind schedule tweaking. V2.2 is close enough to show the failure modes, but not clean enough to scale. The next useful work is to expose reward-component attribution by action and scenario, then make one targeted V2.3 hypothesis.

## Phase 0 diagnostic patch status

Completed in the current worktree on 2026-06-10:

- Added eval summary reward-component diagnostics by action id and action family:
  - `reward_component_sum_by_action_id`
  - `reward_component_count_by_action_id`
  - `reward_component_mean_by_action_id`
  - `reward_component_sum_by_action_family`
  - `reward_component_count_by_action_family`
  - `reward_component_mean_by_action_family`
- Tracked the components needed for the next causal split, including hold-under-lateness penalties, useful-dispatch credits, primary/secondary timing attribution, no-current/no-unassigned penalty, route candidate alignment credit, route candidate mismatch penalty, candidate score gap, useful-work factor, and candidate-alignment blocked flags.
- Fixed training diagnostics so `JointMetricsAggregator` reads `dispatch_no_unassigned_orders` from `reward_components` when it is not present at top level.
- No reward shaping, model behavior, production registry, production checkpoint, baseline, or DB path was modified.

Verification completed:

| Command | Result |
| --- | --- |
| `python -m unittest tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_reward_component_diagnostics_are_reported_by_action_and_family -v` | PASS |
| `python -m unittest tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_joint_metrics_read_no_unassigned_exposure_from_reward_components -v` | PASS |
| `python -m unittest tests.eval.test_real_world_scenario_evaluator -v` | PASS, 36 tests |
| `python -m unittest tests.learn.test_train_joint_curriculum -v` | PASS, 31 tests |
| `python -m unittest tests.eval.test_long_run_gate -v` | PASS, 13 tests |
| `python -m py_compile src\eval\scenario_metrics.py src\learn\joint_metrics.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py` | PASS |

Next no-training step:

- Run fresh diagnostic offline eval directories for production and V2.2 with the patched evaluator.
- Compare action 9 reward attribution in high_holding/lead_time/vehicle/premium/baseline.
- Compare action 41 and action 45 reward attribution in route_disruption.
- Use the diagnostic comparison to choose either a narrow reward patch, a narrow config patch, or an architecture-decision stop.

## Phase 1 diagnostic eval results

Completed fresh read-only diagnostic evals on 2026-06-10. No training run was started.

Fresh diagnostic eval directories:

- Production diagnostic eval: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_level2_diag_20260610_offline_scenarios`
- V2.2 diagnostic eval: `models/eval/joint_torch_v5_prod_stability_v2_2_250k_20260610_level2_diag_20260610_offline_scenarios`

Commands:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; python -m src.eval.evaluate_real_world_scenarios --checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt --scenario-dir configs\eval_scenarios --output-dir models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_level2_diag_20260610_offline_scenarios --episodes 20 --seed 42 --device cpu --deterministic
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; python -m src.eval.evaluate_real_world_scenarios --checkpoint models\checkpoints\joint_torch_v5_prod_stability_v2_2_250k_20260610\joint_torch_latest.pt --scenario-dir configs\eval_scenarios --output-dir models\eval\joint_torch_v5_prod_stability_v2_2_250k_20260610_level2_diag_20260610_offline_scenarios --episodes 20 --seed 42 --device cpu --deterministic
```

Fresh gate command:

```powershell
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_2_250k_20260610_level2_diag_20260610_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_level2_diag_20260610_offline_scenarios\scenario_summary.json --candidate-label stability_v2_2_level2_diag
```

Fresh gate decision: FAIL.

Fatal failures:

- `high_holding_cost` threshold FAIL: service 0.826 < 0.900 and true_lateness_pressure 0.183 > 0.080.
- `lead_time_volatility` threshold FAIL: true_lateness_pressure 0.252 > 0.100.
- `vehicle_scarcity_capacity_shock` threshold FAIL: service 0.826 < 0.850.
- `route_disruption_congestion` dispatch success regressed from production diagnostic 0.862 to V2.2 diagnostic 0.829.

### Diagnostic attribution summary

| Scenario/action | Production diagnostic | V2.2 diagnostic | Finding |
| --- | --- | --- | --- |
| baseline action 9 | 346 uses, 6.0%, dqn_local -0.033, hold pressure 0.044, hold penalty 0.022 | 1623 uses, 28.2%, dqn_local -0.027, hold pressure 0.037, hold penalty 0.020 | V2.2 massively increased action 9 in baseline but still passed. This explains why baseline anchoring preserved the action-9 basin. |
| premium action 9 | 245 uses, 4.3%, dqn_local -0.027, hold penalty 0.013 | 1774 uses, 30.8%, dqn_local -0.074, hold penalty 0.035 | Premium passes but is fragile: action 9 is already the dominant hold family. |
| high_holding action 9 | 1211 uses, 21.0%, dqn_local -0.024, hold penalty 0.018 | 1612 uses, 28.0%, dqn_local -0.104, hold pressure 0.148, hold penalty 0.042 | V2.2 penalizes action 9 more than production, but the penalty is still not enough to reduce its selection under high holding/lateness pressure. |
| lead_time action 9 | 302 uses, 5.2%, dqn_local -0.012, hold penalty 0.005 | 2144 uses, 37.2%, dqn_local -0.093, hold pressure 0.139, hold penalty 0.041 | Clearest timing failure: useful dispatches are feasible and rewarded, but action 9 remains over-selected. |
| vehicle action 9 | 677 uses, 11.8%, dqn_local -0.038, hold penalty 0.030 | 1464 uses, 25.4%, dqn_local -0.091, hold pressure 0.119, hold penalty 0.044 | Capacity shock still has too much conservative hold, reducing dispatch breadth vs production. |
| route action 41 | production: 654 uses, 11.4%, 343 no-work/no-unassigned fails, dqn_local -0.061, route credit sum 37.830, no-work penalty sum 83.166 | V2.2: 933 uses, 16.2%, 192 fails, dqn_local +0.109, route credit sum 64.553, no-work penalty sum 47.648 | Action 41 is net-positive in V2.2 despite route no-work exposure. Route credit on successful uses outweighs penalties enough to make the pocket attractive. |
| route action 45 | production: 629 uses, 10.9%, 46 fails, dqn_local +0.237, route credit sum 45.748 | V2.2: 209 uses, 3.6%, 209 fails, dqn_local -0.397, route credit sum 8.698, no-work penalty sum 45.766 | Action 45 is strongly negative but still selected in a pure-failure pocket. This is likely learned policy/generalization or exploration residue, not positive reward leakage. |
| route action 32 | production: 39 uses, 0.7%, dqn_local +0.230 | V2.2: 865 uses, 15.0%, dqn_local +0.205, 33 fails | Action 32 is a strong low-congestion secondary alternative and may be absorbing much of the route policy. |

### Phase 1 conclusion

The next V-cycle should not be a schedule tweak.

Confirmed mechanism 1: timing scenarios fail because action 9 remains over-selected under useful-dispatch pressure. The reward attribution shows the hold-under-lateness penalty activates, but averages only about 0.041 to 0.044 in the failed timing scenarios. Dispatch action 29 receives positive local reward and useful timing/primary credits, but action 9 still consumes 25-37% of decisions in failed scenarios.

Confirmed mechanism 2: route action quality fails because action 41 is net-positive in V2.2 despite no-current/no-unassigned exposure. Its route credit sum increased to 64.553 while no-work penalty sum was 47.648; mean dqn_local was +0.109. Action 45 is already negative, so the route patch should target high-resilience secondary/conservative credit balance and no-useful-work route credit, not simply punish every high-resilience action.

Next hypothesis for V2.3:

- Timing reward patch: make hold-under-useful-dispatch pressure stronger only when service/lateness/holding/capacity pressure is active. Preserve low/no-pressure baseline holds.
- Route reward patch: suppress high-resilience route resilience/adaptation credit when candidate-alignment blocked flags or no-useful-work/no-unassigned exposure are present, and strengthen mismatch/no-work penalties only for high-resilience dispatches under route disruption.
- Config patch, if any, should be secondary and small: keep V2.2 baseline/premium anchoring, but do not rely on schedule changes as the main fix.

Required next gate before V2.3 config creation or training: independent review of this hypothesis and the local TDD-covered reward patch. The available sub-agent tool requires an explicit user request for delegation, so no independent sub-agent review has been run yet.

Protected-path proof after Phase 1:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Source production checkpoint SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
- Protected directory profile:
  - `models/production`: 4 files, 15040339 bytes, max_mtime_ns 1780972731823548200
  - `models/baselines`: 2692 files, 7392576274 bytes, max_mtime_ns 1779666921467665500
- `db`: 7 files, 28330 bytes, max_mtime_ns 1779200011801591300

## Phase 2 local TDD reward patch status

Completed locally on 2026-06-10. No training run, offline scenario eval, registry write, production mutation, baseline mutation, DB mutation, or checkpoint mutation was started.

Patch hypothesis:

- Timing: action 9 remained over-selected because the hold-under-lateness penalty was capped at 0.10 even under severe service/lateness/holding/capacity pressure.
- Route: high-resilience dispatches with no useful work could still retain resilience/adaptation credit, leaving action 41/45 pockets insufficiently separated from useful high-resilience dispatch.

Files changed:

- `src/act/env_5pl.py`
  - Raised the severe-pressure hold penalty cap from 0.10 to 0.16 only when useful-dispatch pressure exists and operational degradation is present.
  - Added `hold_timing_degradation_pressure` reward telemetry.
  - Suppressed `route_resilience_credit`, `high_resilience_adaptation_credit`, `route_resilience_adaptation_credit`, and `route_adaptation_reward_or_credit` for high-resilience dispatches with no useful route/dispatch work.
  - Added `high_resilience_no_useful_work_credit_blocked` reward telemetry.
- `src/eval/scenario_metrics.py`
  - Added `hold_timing_degradation_pressure` and `high_resilience_no_useful_work_credit_blocked` to reward-component diagnostic aggregation.
- `tests/act/test_reward_physics_contract.py`
  - Added RED/GREEN coverage for no-work high-resilience dispatch credit suppression.
  - Added RED/GREEN coverage for severe timing-pressure hold penalty exceeding the old 0.10 cap while calm holds remain unpenalized.
  - Updated the existing hold-under-lateness bound to the new 0.16 cap.
- `tests/eval/test_real_world_scenario_evaluator.py`
  - Added diagnostic aggregation coverage for the two new reward components by action id and action family.

TDD evidence:

- RED before reward patch:
  - `python -m unittest tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_no_work_high_resilience_dispatch_gets_no_route_resilience_credit tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_severe_timing_pressure_hold_penalty_exceeds_legacy_cap_without_affecting_calm_hold -v`
  - Failed as intended: `route_resilience_credit` was `0.08133333333333333` for no-work high-resilience dispatch, and severe hold penalty was exactly `0.1`.
- GREEN after reward patch:
  - Same focused command passed, 2 tests.
- RED before eval diagnostic allowlist update:
  - `python -m unittest tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_reward_component_diagnostics_are_reported_by_action_and_family -v`
  - Failed as intended with missing `high_resilience_no_useful_work_credit_blocked`.
- GREEN after eval diagnostic allowlist update:
  - Same focused command passed.

Verification after patch:

| Command | Result |
| --- | --- |
| `python -m unittest tests.act.test_reward_physics_contract -v` | PASS, 127 tests |
| `python -m unittest tests.eval.test_real_world_scenario_evaluator -v` | PASS, 36 tests |
| `python -m unittest tests.learn.test_train_joint_curriculum -v` | PASS, 31 tests |
| `python -m unittest tests.learn.test_curriculum_config -v` | PASS, 64 tests |
| `python -m unittest tests.eval.test_long_run_gate -v` | PASS, 13 tests |
| `python -m py_compile src\act\env_5pl.py src\eval\scenario_metrics.py src\learn\joint_metrics.py tests\act\test_reward_physics_contract.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py` | PASS |
| Production long-run gate self-check | PASS, exit 0 |
| Training/eval process check for `train_joint_torch` or `evaluate_real_world_scenarios` | No matching process |

Additional pre-review coverage added after the initial Phase 2 package:

- `tests/act/test_reward_physics_contract.py:2878` verifies the exact route failure pockets, action 41 and action 45, block high-resilience route credits when no useful dispatch work is available.
- `tests/act/test_reward_physics_contract.py:2926` verifies useful action 41 and action 45 dispatch still retains high-resilience route credit under strong route disruption.
- Focused command: `python -m unittest tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_route_failure_actions_41_and_45_block_no_work_resilience_credit tests.act.test_reward_physics_contract.RewardPhysicsContractTest.test_useful_route_actions_41_and_45_keep_high_resilience_credit -v` passed, 2 tests.
- Mapper command: `python -m unittest tests.act.test_discrete_action_mapper_contract -v` passed, 1 test.

Protected-path proof after Phase 2:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Source production checkpoint SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

## Phase 2 post-patch diagnostic evals

Completed after the local reward patch on 2026-06-10. These were read-only diagnostic evals of existing checkpoints, not training runs.

Fresh eval directories:

- Production: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_level2_phase2_patch_diag_20260610_offline_scenarios`
- V2.2: `models/eval/joint_torch_v5_prod_stability_v2_2_250k_20260610_level2_phase2_patch_diag_20260610_offline_scenarios`

Fresh V2.2 diagnostic gate decision: FAIL.

Fatal failures:

- `high_holding_cost`: service `0.826 < 0.900`; true lateness `0.183 > 0.080`.
- `lead_time_volatility`: true lateness `0.252 > 0.100`.
- `vehicle_scarcity_capacity_shock`: service `0.826 < 0.850`.
- `route_disruption_congestion`: dispatch success regressed from production diagnostic `0.862` to V2.2 diagnostic `0.829`.

Post-patch attribution:

| Scenario/action | Finding |
| --- | --- |
| V2.2 high_holding action 9 | Mean hold penalty `0.086780`, degradation pressure `0.860051`, mean `dqn_local -0.148722`. |
| V2.2 lead_time action 9 | Mean hold penalty `0.083217`, degradation pressure `0.804835`, mean `dqn_local -0.135591`. |
| V2.2 vehicle action 9 | Mean hold penalty `0.094934`, degradation pressure `0.914763`, mean `dqn_local -0.142536`. |
| V2.2 route action 41 | No-useful-work credit block sum `192`; route resilience/adaptation credit sum `54.596880`; no-work penalty sum `47.648150`; mean `dqn_local 0.100197`. |
| V2.2 route action 45 | No-useful-work credit block sum `209`; route resilience/adaptation credit sum `0.000000`; no-work penalty sum `45.766187`; mean `dqn_local -0.438934`. |

Interpretation:

- The hold patch is active where intended: action 9 is now much more negative in the failed timing scenarios.
- The route patch fully blocks no-work credit for action 45 and blocks no-work cases for action 41. Action 41 remains net-positive because successful uses still receive route credit.
- V2.2 remains a failed diagnostic candidate and remains prohibited as a parent.

Review package:

- `docs/runs/20260610_level2_phase2_review_request.md`
- `docs/runs/20260610_level2_phase2_review_verdict.md`
- `docs/runs/20260610_level2_phase2_review_verdict_template.md`

Pre-review implementation runbook:

- `docs/plans/20260610_level2_v2_3_pre_review_plan.md`
- This runbook records the proposed V2.3 identity, schedule, TDD config-test path, training command, eval command, and gate rules. It does not authorize config creation or training before independent review.

Next required gate:

- Independent review of the Phase 1 causal hypothesis and Phase 2 reward patch completed with verdict `PHASE2_PATCH_APPROVED_FOR_V2_3_CONFIG`.
- Only after independent review, create a fresh V2.3 250k config that parents from production and keeps V2.2 baseline/premium anchoring.
- Do not train until the independent review and config tests pass.

Classification: LEVEL2_AUTOPILOT_RESEARCH_CONTINUES
