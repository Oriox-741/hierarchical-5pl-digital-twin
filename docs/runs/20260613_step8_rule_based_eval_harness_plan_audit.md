# Step 8 Rule-Based Eval Harness Plan Audit

Date: 2026-06-13

Status: planning and documentation only. No training, offline eval, long-run gate, benchmark, dependency installation, dataset download, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output edit, training config creation, private company-data ingestion, firm-facing email work, or longer training run was performed.

## Objective

Prepare the no-run planning package for connecting the existing deterministic rule-based baseline policies to the existing 5PL scenario/evaluation stack later, without duplicating `env_5pl` and without executing a benchmark or eval.

## Files Read

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/runs/20260613_rule_based_baseline_skeleton_report.md`
- `src/eval/rule_based_baselines.py`
- `tests/eval/test_rule_based_baselines.py`
- `docs/plans/20260612_rule_based_baseline_benchmark_spec.md`
- `docs/plans/20260612_rule_based_baseline_implementation_plan_no_run.md`
- `docs/reports/20260612_tez_danismani_rule_based_baseline_aciklamasi.md`
- `docs/plans/20260612_benchmark_protocol_v1_no_run.md`
- `docs/plans/20260612_first_benchmark_recommendation_plan.md`
- `docs/plans/20260612_project_development_master_plan_no_us_application.md`
- `docs/plans/20260612_autonomous_simulation_capability_audit_and_demo_plan.md`
- `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`
- `src/act/env_5pl.py`
- `src/act/action_projector.py`
- `src/act/discrete_action_mapper.py`
- `src/eval/real_world_scenario_arena.py`
- `src/eval/evaluate_real_world_scenarios.py`
- `src/eval/scenario_metrics.py`
- `src/eval/long_run_gate.py`
- `configs/eval_scenarios/*.json`

## Files Written

- `docs/plans/20260613_rule_based_baseline_eval_harness_plan_no_run.md`
- `docs/goals/20260613_execute_rule_based_baseline_eval_harness_skeleton_goal.txt`
- `docs/runs/20260613_step8_rule_based_eval_harness_plan_audit.md`
- `docs/00_PROJECT_DASHBOARD.md` only for Step 8 links/status

## Architecture Finding

The future harness should be a library adapter at `src/eval/rule_based_baseline_arena.py`, not a new simulator and not a script-first implementation.

Reasoning:

- `env_5pl` already owns the 5PL physics and joint action application.
- `real_world_scenario_arena.py` already provides scenario environment construction and the learned-policy rollout pattern.
- `scenario_metrics.py` already provides the metrics accumulator and scenario summary logic.
- The rule baseline skeleton already emits legal external DQN action ids.
- A separate library adapter keeps checkpoint evaluation and rule-baseline evaluation responsibilities separate while remaining testable without running eval.

## Continuous Action Strategy

The future skeleton should use a fixed neutral normalized continuous action vector:

```text
[0.0, 0.0, 0.0, 0.0, 0.0]
```

This is deterministic, length 5, finite, bounded in normalized PPO action space, and maps through `ActionProjector` to midpoint physical controls. It avoids checkpoint loading and avoids scenario-specific hand tuning. The caveat is explicit: this compares deterministic discrete rule logic under neutral continuous controls, not a full learned PPO+DQN policy.

## Planning Questions Answered

1. Rule policies plug into the existing stack by converting env snapshots into `RuleBasedDecisionContext`, selecting an action id, and calling `env.step` with neutral continuous controls plus that action id.
2. The chosen future harness location is `src/eval/rule_based_baseline_arena.py`, because it is testable and avoids script-first output writing.
3. Rule policies return `RuleBasedDecision.action_id`; the adapter must validate the id is in 0..47 and tests should decode with `DiscreteActionMapper`.
4. Continuous PPO-like controls should use a fixed neutral normalized vector, not scenario-specific defaults and not production PPO outputs.
5. Later results should prefer fresh `reports/rule_based_baselines/...`; fresh `models/eval/...` requires explicit eval approval.
6. Metrics compatibility comes through `EpisodeMetricAccumulator` consuming the existing environment step `info`.
7. Fair comparisons include service, lateness, dispatch success/rate, hard blockers, no-work counters, route failures, no-vehicle, already-assigned, action concentration, action distribution, secondary_fleet rate, and reorder_none rate under the same env/scenario/seed/episode budget.
8. The rules cannot fairly compare learned continuous control quality, Q values/logits, training efficiency, real-world economics, or company-data replay outcomes.
9. Tests are required for neutral continuous controls, legal action ids, adapter output shape, checkpoint independence, output-write safety, action 24/32 legality, and accumulator integration through fake/mocked envs.
10. Actual eval/benchmark requires separate explicit approval with exact scenario dir, seeds, episodes, output dir, protected hashes, and stop conditions.

## No-Run Proof

Commands inspected files, hashes, directory profiles, and process state only. The following were not run:

- training commands;
- `src.eval.evaluate_real_world_scenarios`;
- long-run gate commands;
- benchmark commands;
- AWS or dataset download commands;
- dependency installation commands;
- private company-data ingestion.

No output directory was created under `models/eval` or `reports/rule_based_baselines`.

## Protected Pre-Write Snapshot

Protected file hashes before documentation writes:

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `models/registry/active_models.json` | 314 | `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A` |
| `models/registry/models.jsonl` | 14142 | `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt` | 303413903 | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json` | 4084 | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` |

Protected directory profiles before documentation writes:

| Path | File count | Bytes |
| --- | ---: | ---: |
| `models/production` | 8 | 328265276 |
| `models/baselines` | 2692 | 7392576274 |
| `db` | 7 | 28330 |
| `models/checkpoints` | 787 | 4920830666 |
| `models/eval` | 128 | 388267659 |

Process scan found no matching train/eval/gate/AWS/private-data process patterns before writing.

## Reviewer Verdict

Independent read-only reviewer verdict:

`RULE_BASED_EVAL_HARNESS_PLAN_APPROVED`

## Final Classification

`RULE_BASED_EVAL_HARNESS_PLAN_READY`
