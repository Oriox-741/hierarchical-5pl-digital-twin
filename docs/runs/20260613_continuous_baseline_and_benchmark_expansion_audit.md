# Continuous Baseline And Benchmark Expansion Audit

Date: 2026-06-13

Final classification: `CONTINUOUS_BASELINE_AND_BENCHMARK_EXPANSION_PLAN_READY`

## Scope

Objective: analyze and prepare implementation for continuous-aware rule baselines, PPO-assisted rule tactical baselines, and non-route benchmark expansion beyond OR-Tools. This was a planning/documentation task only.

Allowed writes:

- `docs/plans/20260613_continuous_aware_baseline_plan.md`
- `docs/plans/20260613_non_route_benchmark_expansion_plan.md`
- `docs/reports/20260613_sota_pathway_for_5pl_decision_control_tr.md`
- `docs/goals/20260613_execute_continuous_aware_baseline_skeleton_goal.txt`
- `docs/runs/20260613_continuous_baseline_and_benchmark_expansion_audit.md`
- `docs/00_PROJECT_DASHBOARD.md` links/status only

## Local Evidence Read

- `docs/reports/20260613_benchmark_scorecard_for_advisor_tr.md`
- `docs/reports/20260613_benchmark_sonuclari_basit_yorum_tr.md`
- `docs/runs/20260613_full_benchmark_completion_sprint_audit.md`
- `docs/plans/20260612_rule_based_baseline_benchmark_spec.md`
- `src/eval/rule_based_baselines.py`
- `src/eval/rule_based_baseline_arena.py`
- `src/act/env_5pl.py`
- `src/act/action_projector.py`
- `src/act/observation_builder.py`
- `src/act/discrete_action_mapper.py`
- `tests/eval/test_rule_based_baselines.py`
- `tests/eval/test_rule_based_baseline_arena.py`
- `docs/reports/20260612_tez_danismani_operasyon_walkthrough_detayli.md`
- `docs/plans/20260613_rule_based_baseline_eval_harness_plan_no_run.md`
- `docs/goals/20260613_execute_rule_based_baseline_eval_harness_skeleton_goal.txt`

## Web / Public Source Evidence

Public benchmark families were checked using primary or official pages:

- OR-Gym: https://github.com/hubbs5/or-gym
- MABIM / ReplenishmentEnv: https://github.com/VictorYXL/ReplenishmentEnv
- FleetPy: https://github.com/TUM-VT/FleetPy
- SVRPBench OpenReview: https://openreview.net/forum?id=yADrJyaBJl
- PyVRP: https://github.com/PyVRP/PyVRP
- CityFlow: https://github.com/cityflow-project/CityFlow/
- LibSignal: https://darl-libsignal.github.io/
- OpenMines: https://github.com/370025263/openmines

No datasets were downloaded and no dependencies were installed.

## Key Source Findings

1. Current rule-based baseline arena uses fixed continuous control:
   - `neutral_continuous_action()` returns `np.zeros((CONTINUOUS_ACTION_DIM,), dtype=np.float32)`.
   - `build_joint_action_from_rule_decision()` combines this vector with the rule discrete action id.

2. Source-truth continuous physical fields are:
   - `reorder_fraction`
   - `dispatch_intensity`
   - `speed_multiplier`
   - `safety_stock_multiplier`
   - `capacity_buffer_fraction`

3. Normalized zero is not physical no-op:
   - with default `ActionProjector`, normalized zero maps to `reorder_fraction=0.5`, `dispatch_intensity=0.5`, `speed_multiplier=1.0`, `safety_stock_multiplier=1.5`, `capacity_buffer_fraction=0.25`.

4. Joint mode applies PPO continuous controls first with `allow_dispatch=False`.
   - continuous controls still affect replenishment, capacity buffer, speed, and macro dispatch budget;
   - DQN discrete action then handles dispatch/hold, route, fleet mode, and reorder override.

5. Existing rule tests already cover deterministic discrete rule behavior and legal action ids, including action 24 and action 32.

## Documents Written

- `docs/plans/20260613_continuous_aware_baseline_plan.md`
- `docs/plans/20260613_non_route_benchmark_expansion_plan.md`
- `docs/reports/20260613_sota_pathway_for_5pl_decision_control_tr.md`
- `docs/goals/20260613_execute_continuous_aware_baseline_skeleton_goal.txt`
- `docs/runs/20260613_continuous_baseline_and_benchmark_expansion_audit.md`
- `docs/00_PROJECT_DASHBOARD.md` dashboard link/status update

## Requirements Checklist

- Inspect neutral continuous use in `rule_based_baseline_arena`: complete.
- Identify exact PPO continuous semantics from source: complete.
- Design continuous-aware modes: complete.
  - `neutral_continuous`
  - `heuristic_continuous`
  - `ppo_assisted_continuous`
  - optional `oracle_diagnostic_continuous`, marked non-deployable.
- Define heuristic rules from non-leaky current-state signals: complete.
- Define fairness and interpretation by mode: complete.
- Create skeleton implementation goal file: complete.
- Research and document non-route benchmark families: complete.
- Write SOTA pathway report with do-not-overclaim warnings: complete.
- Independent reviewer: pending.

## Protected No-Mutation Snapshot

Pre-write read-only hash spot check:

- `models/registry/active_models.json`: `b7c6e79749044b92639b8a27ef38483022fc21d9f0ee08743af88725e42aa60a`
- `models/registry/models.jsonl`: `935d8bf34adc8dd956a68fb57581ec860b02d74ec26434dd6c084e63055c38e1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: `fdaf8de495d90bdfd69bcc2ee3772eb6a09661e104a5127ea3cb01c6d02cf99a`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: `c1e756bdf7cd6b824ca134dbe6fb021612367ab18b12f2a24e13bf2167e51cde`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7a6e4eaac7cb9cdbf5926a544cf1cab7812940586c26e4c01cc757f3fd81b52e`

Post-write read-only hash spot check matched the pre-write hashes:

- `models/registry/active_models.json`: `b7c6e79749044b92639b8a27ef38483022fc21d9f0ee08743af88725e42aa60a`
- `models/registry/models.jsonl`: `935d8bf34adc8dd956a68fb57581ec860b02d74ec26434dd6c084e63055c38e1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: `fdaf8de495d90bdfd69bcc2ee3772eb6a09661e104a5127ea3cb01c6d02cf99a`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: `c1e756bdf7cd6b824ca134dbe6fb021612367ab18b12f2a24e13bf2167e51cde`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7a6e4eaac7cb9cdbf5926a544cf1cab7812940586c26e4c01cc757f3fd81b52e`

## No-Run / No-Mutation Statement

This task:

- did not train;
- did not run offline eval;
- did not run long-run gate;
- did not run benchmarks;
- did not install dependencies;
- did not download datasets;
- did not ingest private company data;
- did not mutate registry, production, baselines, DB, checkpoints, or existing eval outputs;
- did not overwrite benchmark outputs.

## Verification

Completed:

- Confirmed all five requested new docs exist.
- Confirmed dashboard link/status update exists.
- Confirmed required continuous modes are present in the continuous-aware baseline plan.
- Confirmed source-truth continuous fields are present in the continuous-aware baseline plan.
- Confirmed non-route benchmark families are present in the non-route expansion plan.
- Confirmed SOTA pathway report includes no-SOTA and overclaim warnings.
- Confirmed future skeleton goal includes hard constraints, allowed writes, TDD requirements, verification commands, reviewer requirement, and final classifications.
- Confirmed protected hash spot check unchanged.
- Initial process scan briefly showed transient Python processes from local checks; command-line lookup returned no active process, and a follow-up process scan found no train/eval/gate/AWS process running.
- Independent read-only reviewer: pending.

## Independent Reviewer

Reviewer verdict: `CONTINUOUS_BASELINE_AND_BENCHMARK_EXPANSION_APPROVED`.

Allowed verdicts:

- `CONTINUOUS_BASELINE_AND_BENCHMARK_EXPANSION_APPROVED`
- `CONTINUOUS_BASELINE_AND_BENCHMARK_EXPANSION_NEEDS_FIXES`
- `CONTINUOUS_BASELINE_AND_BENCHMARK_EXPANSION_BLOCKED`

## Final Classification

`CONTINUOUS_BASELINE_AND_BENCHMARK_EXPANSION_PLAN_READY`
