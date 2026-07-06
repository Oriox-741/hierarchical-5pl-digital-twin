# Level 2 V2.4 500k Post-Training Audit

Date: 2026-06-10

The 500k continuation trained successfully, but the long-run gate failed. The ladder is stopped before 1M.

Primary report:

- `docs/runs/20260610_level2_v2_4_500k_architecture_decision_report.md`

Key results:

- 250k semantics gate: PASS, exit 0.
- 500k training: exit 0, final `global_step=500000`.
- 500k eval: exit 0.
- 500k long-run gate: FAIL, exit 2.
- 300k periodic diagnostic gate: FAIL.
- 400k periodic diagnostic gate: FAIL.
- 1M config/training: not created, not started.

Changed files in this phase:

- `src/eval/scenario_metrics.py`
- `tests/eval/test_real_world_scenario_evaluator.py`
- `tests/learn/test_curriculum_config.py`
- `configs/training_joint_curriculum_v5_prod_stability_v2_4_500k_20260610.json`
- `docs/runs/20260610_level2_v2_4_semantics_patch_review_request.md`
- `docs/runs/20260610_level2_v2_4_semantics_patch_review_verdict_template.md`
- `docs/runs/20260610_level2_v2_4_semantics_patch_review_verdict.md`
- `docs/runs/20260610_level2_v2_4_semantics_gate_result.json`
- `docs/runs/20260610_level2_v2_4_500k_config_review_request.md`
- `docs/runs/20260610_level2_v2_4_500k_config_review_verdict_template.md`
- `docs/runs/20260610_level2_v2_4_500k_config_review_verdict.md`
- `docs/runs/20260610_level2_v2_4_300k_diag_gate_result.json`
- `docs/runs/20260610_level2_v2_4_400k_diag_gate_result.json`
- `docs/runs/20260610_level2_v2_4_500k_gate_result.json`
- `docs/runs/20260610_level2_v2_4_500k_architecture_decision_report.md`
- `docs/runs/20260610_level2_v2_4_500k_post_training_audit.md`

Verification commands:

| Command | Result |
| --- | --- |
| `python -m unittest tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_healthy_in_transit_order_is_active_but_not_stuck_at_finite_horizon tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_overdue_active_order_counts_as_stuck_failure tests.eval.test_real_world_scenario_evaluator.ScenarioMetricAlignmentTests.test_stale_active_order_without_due_time_counts_as_stuck_failure -v` | PASS |
| `python -m unittest tests.eval.test_real_world_scenario_evaluator -v` | PASS, 39 tests |
| `python -m py_compile src\eval\scenario_metrics.py tests\eval\test_real_world_scenario_evaluator.py` | PASS |
| `python -m unittest tests.eval.test_long_run_gate -v` | PASS, 13 tests |
| `python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_stability_v2_4_500k_config_loads_and_is_manual_run_safe -v` | PASS |
| `python -m json.tool configs\training_joint_curriculum_v5_prod_stability_v2_4_500k_20260610.json > $null` | PASS |

Independent reviews:

- Semantics patch reviewer: GPT-5.5 subagent `019eb252-5bce-7ba3-8571-96ff71c36460`, verdict `PATCH_APPROVED_FOR_NEXT_GATE`.
- 500k config reviewer: GPT-5.5 subagent `019eb26b-a080-7e12-bccc-52f60dc46a27`, verdict `PATCH_APPROVED_FOR_NEXT_GATE`.

Protected-path proof is recorded in the architecture decision report.

Classification: `AUTOPILOT_BLOCKED_NEEDS_ARCHITECTURE_DECISION`
