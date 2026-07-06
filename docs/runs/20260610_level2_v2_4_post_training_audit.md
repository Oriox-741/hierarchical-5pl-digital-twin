# Level 2 V2.4 Post-Training Audit

Status: V2.4 250k trained and evaluated. Candidate is not approved for 500k.

## Artifacts

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`
- Checkpoint dir: `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610`
- Standard eval dir: `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios`
- Stuck-order diagnostic eval dir: `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios_stuckdiag`
- Production comparison summary: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json`

## Training

Command exited `0`.

V2.4 parent was the protected production checkpoint:

`models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`

Final artifacts present:

- `joint_torch_latest.pt`
- `ppo_torch_joint_final_stability_v2_4_250k_20260610.pt`
- `dqn_torch_joint_final_stability_v2_4_250k_20260610.pt`
- `training_metrics.jsonl`

## Verification Before Training

| Check | Result |
| --- | --- |
| V2.4 checkpoint dir absent before launch | PASS |
| V2.4 eval dir absent before launch | PASS |
| No Python training/eval process before launch | PASS |
| `python -m py_compile src\act\env_5pl.py src\eval\scenario_metrics.py src\learn\joint_metrics.py tests\act\test_reward_physics_contract.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py tests\eval\test_long_run_gate.py` | PASS |
| `python -m unittest tests.act.test_reward_physics_contract tests.eval.test_real_world_scenario_evaluator tests.learn.test_train_joint_curriculum tests.learn.test_curriculum_config tests.eval.test_long_run_gate -v` | PASS, 276 tests |

Protected-path hashes before training and after training matched:

- `models/registry/active_models.json`: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl`: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Production checkpoint: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

## Standard Offline Eval And Gate

Offline eval exited `0`.

Long-run gate decision: `FAIL`.

Fatal failures:

- `baseline_normal: scenario threshold verdict is 'FAIL'.`
- `baseline_normal: scenario threshold failure: stuck assigned/in-transit orders present: 1`

Scenario verdicts:

| Scenario | Verdict | Service | Lateness | Dispatch success | Stuck assigned/in-transit |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline_normal` | FAIL | 0.960676 | 0.000000 | 0.999569 | 1 |
| `demand_spike_volatility` | PASS | 0.948146 | 0.002085 | 1.000000 | 0 |
| `high_holding_cost` | PASS | 0.989684 | 0.000148 | 1.000000 | 0 |
| `lead_time_volatility` | PASS | 0.961797 | 0.006131 | 0.998239 | 0 |
| `mixed_stress` | PASS | 0.935666 | 0.024644 | 1.000000 | 0 |
| `premium_sla_pressure` | PASS | 0.988228 | 0.000000 | 0.999124 | 0 |
| `route_disruption_congestion` | PASS | 0.946331 | 0.001483 | 1.000000 | 0 |
| `vehicle_scarcity_capacity_shock` | PASS | 0.941837 | 0.000000 | 1.000000 | 1 |

Important comparison against V2.3:

- V2.3 route dispatch success: `0.712948`.
- V2.4 route dispatch success: `1.000000`.
- V2.3 route action 32 failed/no-current/no-unassigned pocket: action 32 count `2462`, failed no-op `848`.
- V2.4 route failed-noop/no-current/no-unassigned route pocket: zero in route scenario.

The Phase 3 patch achieved its target. The candidate failed because of a baseline horizon stuck-order threshold, not because the route action-quality regression persisted.

## Diagnostic Patch

Because standard eval only reported the stuck count, a read-only evaluator diagnostic was added with TDD:

- `src/eval/scenario_metrics.py`
- `tests/eval/test_real_world_scenario_evaluator.py`

New field:

- `stuck_assigned_in_transit_order_examples`

Focused RED:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_stuck_assigned_order_diagnostic_preserves_final_order_context -v
```

Expected failure:

- `KeyError: 'stuck_assigned_in_transit_order_examples'`

GREEN:

- Focused test passed.
- Full evaluator test file passed: `python -m unittest tests.eval.test_real_world_scenario_evaluator -v`, 37 tests.
- `python -m py_compile src\eval\scenario_metrics.py tests\eval\test_real_world_scenario_evaluator.py` passed.

The diagnostic patch changes reporting only. It does not alter reward, policy, gate logic, registry, production, baselines, DB, or checkpoints.

## Diagnostic Eval Finding

The enriched deterministic eval was written to:

`models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios_stuckdiag`

Gate decision on the diagnostic summary remained `FAIL` for the same baseline fatal failure.

Baseline stuck-order example:

| Field | Value |
| --- | --- |
| Scenario | `baseline_normal` |
| Episode / seed | `8` / `50` |
| Order status | `in_transit` |
| Final snapshot time | `86400.0` |
| Pickup time | `86100.0` |
| Delivery time | `null` |
| Due time | `100178.20358614686` |
| Seconds to due at horizon | `13778.203586146861` |
| `is_late` | `false` |
| Vehicle tier | `secondary` |
| Vehicle utilization | `0.14450211741271216` |

This order was picked up five minutes before the finite eval horizon and was still more than 3.8 hours from due time. The current evaluator labels it as a fatal stuck assigned/in-transit order solely because it is active at the final snapshot.

Vehicle scarcity also had one active in-transit order at horizon, but that scenario does not set `fail_on_stuck_assigned_in_transit_orders`, so it passed.

## Causal Conclusion

V2.4 is not a 500k candidate under the current gate.

The remaining blocker is not supported as a reward route-action failure:

- No route failed-noop/no-current/no-unassigned explosion remains.
- Premium dispatch success recovered.
- Baseline service and lateness are healthy.
- The fatal baseline order is not late and was recently picked up near the horizon.

The evidence instead points to an evaluator semantics question:

- Current metric name: `stuck_assigned_in_transit_orders`.
- Current implementation: any final order with status `assigned` or `in_transit` and an assigned vehicle.
- Observed failure: healthy in-transit order at horizon, not overdue, not stale, not late.

## Required Decision

Do not create V2.5 or 500k config before this is resolved.

The next step requires a human-level architecture/policy decision:

1. Keep the current strict horizon-zero interpretation. Then V2.4 remains failed, and the next V-cycle must train or patch the system to avoid any active assigned/in-transit orders at the finite eval boundary.
2. Redefine "stuck" to mean stale or late assigned/in-transit work, not any healthy active delivery at the horizon. Then add TDD for the threshold semantics, rerun eval/gate, and review whether V2.4 can be considered a valid 250k pass.
3. Change the eval scenario design to include a terminal drain/no-new-demand window, keeping strict horizon-zero only after the environment stops injecting near-horizon work.

Recommended decision: option 2 or option 3. Option 1 optimizes against an artificial terminal artifact unless the production requirement truly demands zero active deliveries exactly at the eval horizon.

Classification: AUTOPILOT_BLOCKED_NEEDS_ARCHITECTURE_DECISION
