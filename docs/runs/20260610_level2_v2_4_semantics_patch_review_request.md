# Level 2 V2.4 Semantics Patch Review Request

Date: 2026-06-10

## Review Scope

Read-only independent review of the V2.4 evaluator-semantics patch.

The reviewer must inspect:

- `src/eval/scenario_metrics.py`
- `tests/eval/test_real_world_scenario_evaluator.py`
- `src/eval/long_run_gate.py`
- `tests/eval/test_long_run_gate.py`
- `configs/eval_scenarios/baseline_normal.json`
- `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios_stuckdiag/scenario_summary.json`
- `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_level2_diag_20260610_offline_scenarios/scenario_summary.json`
- `docs/runs/20260610_level2_v2_4_post_training_audit.md`

## Patch Intent

V2.4 250k restored or improved baseline, premium, route-premium, and vehicle-scarcity performance, but the gate failed because `baseline_normal` had one final in-transit order at the finite 24h horizon.

Diagnostic evidence showed that order was not overdue:

- final snapshot time: `86400.0`
- status: `in_transit`
- pickup time: `86100.0`
- due time: `100178.20358614686`
- seconds to due: `13778.203586146861`
- `is_late`: `false`

The patch changes evaluator semantics so:

- `active_assigned_in_transit_orders` records all active assigned/in-transit orders at the final horizon.
- `active_assigned_in_transit_order_examples` preserves raw context examples for audit.
- `stuck_assigned_in_transit_orders` is fatal only for active assigned/in-transit orders that are overdue, explicitly late, or stale when due time is unavailable.
- Existing `fail_on_stuck_assigned_in_transit_orders` gates continue to fail true stuck work.

## Required Checks

1. Healthy future-due assigned/in-transit work at the finite horizon is no longer treated as a fatal stuck order.
2. Overdue active assigned/in-transit work is still counted as stuck and still trips the configured threshold.
3. Stale active assigned/in-transit work without due time is still counted as stuck.
4. Raw active-order diagnostics are aggregated into episode JSON and scenario summaries.
5. The long-run gate still fails any real scenario threshold failure and does not bypass hard blockers.
6. It is safe to rerun fair V2.4-vs-production offline evals after this patch.

## Verification Already Run Locally

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_healthy_in_transit_order_is_active_but_not_stuck_at_finite_horizon tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_overdue_active_order_counts_as_stuck_failure tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_stale_active_order_without_due_time_counts_as_stuck_failure -v
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
python -m py_compile src\eval\scenario_metrics.py tests\eval\test_real_world_scenario_evaluator.py
python -m unittest tests.eval.test_long_run_gate -v
```

All commands exited 0 after the patch.

## Constraints

Do not train.
Do not run offline eval.
Do not create the next config.
Do not update registry.
Do not mutate production.
Do not mutate baselines.
Do not mutate DB.
Do not mutate checkpoints.

Return exactly one verdict line:

```text
PATCH_APPROVED_FOR_NEXT_GATE
PATCH_NEEDS_FIXES_BEFORE_NEXT_GATE
AUTOPILOT_NEEDS_ARCHITECTURE_DECISION
```
