# Rule-Based Baseline Eval Harness Plan - No Run

Date: 2026-06-13

Status: planning package only. This plan does not authorize training, offline eval, long-run gate, benchmark execution, dependency installation, dataset download, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output edits, training config creation, private company-data ingestion, firm-facing email work, or longer training runs.

Recommended classification: `RULE_BASED_EVAL_HARNESS_PLAN_READY`

## 1. Executive Summary

Step 8 should connect the existing deterministic rule-based baselines to the existing 5PL scenario and metric stack through a thin adapter layer. It should not create a second simulator and should not modify the learned-policy evaluation path as the first move.

Recommended future architecture:

- Add a testable library module: `src/eval/rule_based_baseline_arena.py`.
- Reuse `FivePLDigitalTwinEnv`, `configs/eval_scenarios/*.json`, `build_scenario_environment`, and `EpisodeMetricAccumulator`.
- Keep `src/eval/real_world_scenario_arena.py` as the checkpoint-policy arena.
- Add a CLI wrapper later only under explicit eval or benchmark approval.
- Use a fixed neutral normalized continuous action vector of length 5 for rule baselines.
- Emit only in-memory episode rows in the skeleton phase; no scenario eval run, no benchmark, and no `models/eval` writes.

This is the safest route because the rule policies in `src/eval/rule_based_baselines.py` already emit valid external DQN action ids 0..47. The missing piece is an adapter that builds a rule decision context from the simulator state and feeds the existing environment step with:

```text
{"continuous": neutral_ppo_vector, "discrete": rule_action_id}
```

## 2. Why Follows Step 7

Step 7 created the pure deterministic rule-based policy skeleton and tests. Those policies are intentionally small, transparent, and independent of checkpoints, registry state, or eval output directories.

Step 8 is the next natural planning step because a benchmark needs two extra pieces before any run is safe:

- an adapter between deterministic rules and the autonomous 5PL scenario loop;
- a result shape that is compatible with existing scenario metrics without writing benchmark/eval artifacts prematurely.

This sequencing keeps the benchmark path disciplined: policy definitions first, harness plan second, skeleton/tests third, then explicit run approval later.

## 3. Existing Skeleton Summary

`src/eval/rule_based_baselines.py` currently defines deterministic rule policies and a compact decision context:

- `OrderView`
- `RuleBasedDecisionContext`
- `RuleBasedDecision`
- `RuleBasedBaseline`
- `build_rule_based_baselines()`

The skeleton action ids are existing external actions:

- `0`: hold
- `24`: dispatch, shortest route, secondary fleet, no reorder
- `28`: dispatch, shortest route, primary fleet, no reorder
- `29`: dispatch, shortest route, primary fleet, conservative reorder
- `31`: dispatch, shortest route, primary fleet, emergency reorder
- `32`: dispatch, low-congestion route, secondary fleet, no reorder
- `36`: dispatch, low-congestion route, primary fleet, no reorder

`tests/eval/test_rule_based_baselines.py` validates deterministic behavior and legal action-id output. The skeleton does not instantiate `env_5pl`, load checkpoints, run eval, run gates, write outputs, or mutate protected artifacts.

## 4. Existing Autonomous Scenario Loop Summary

The autonomous simulator/eval stack already exists:

- `src/act/env_5pl.py` owns the 5PL simulation physics, observation contract, joint action application, reward components, and step `info`.
- `src/act/discrete_action_mapper.py` owns the external 48-action DQN contract.
- `src/eval/real_world_scenario_arena.py` loads scenario configs, builds scenario environments, runs closed-loop checkpoint policy episodes, accumulates metrics, and writes scenario outputs.
- `src/eval/scenario_metrics.py` owns episode accumulation, scenario summarization, hard-blocker checks, action distributions, and warning/threshold logic.
- `src/eval/evaluate_real_world_scenarios.py` is the offline eval CLI and must not be called by this Step 8 planning task.
- `src/eval/long_run_gate.py` is the gate comparator and must not be called by this Step 8 planning task.

The current production model remains:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- runtime: `torch_joint`
- architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: `73`
- PPO continuous action dimension: `5`
- DQN external discrete action count: `48`
- status: active registry plus copy-only production promoted
- ladder: 250k PASS, 500k PASS, 1M PASS
- equal-budget residual-watch gate: PASS
- ops bundle: `OPS_BUNDLE_CLEAN`
- accepted watches: action 24 and action 32 concentration

## 5. Proposed Eval Harness Architecture

Choose `src/eval/rule_based_baseline_arena.py` as the future skeleton location.

Rationale:

- It is a library module, so adapter behavior can be unit-tested without running a scenario eval.
- It can reuse `load_scenario_configs`, `build_scenario_environment`, and `EpisodeMetricAccumulator` without changing simulator physics.
- It keeps learned checkpoint evaluation and rule baseline evaluation separate.
- It avoids making `scripts/evaluate_rule_based_baselines.py` the first artifact, which would be more likely to drift into output writing or benchmark execution.
- It avoids adding polymorphic hooks to `real_world_scenario_arena.py`, whose current responsibilities are checkpoint metadata validation, checkpoint policy loading, and learned-policy rollout.

Future CLI stance:

- A script such as `scripts/evaluate_rule_based_baselines.py` can be planned later, but only after skeleton tests pass and the user explicitly approves a benchmark or eval run with exact fresh output directories.

## 6. Rule Policy Adapter Design

The future adapter should translate each simulator step into a `RuleBasedDecisionContext` and call a named `RuleBasedBaseline`.

Proposed small units:

- `build_rule_context(snapshot, scenario_id, step_index) -> RuleBasedDecisionContext`
- `select_rule_joint_action(policy, context) -> dict[str, object]`
- `neutral_continuous_action() -> numpy.ndarray`
- `run_rule_baseline_episode(..., dry_run_only=False) -> EpisodeMetrics`

The context builder should use only information already available from the existing environment snapshot and scenario id. It must not read checkpoints, registry state, production files, baselines, DB files, or private company data.

Legal action guarantee:

- Rule policies must return `RuleBasedDecision.action_id`.
- The adapter must validate `0 <= action_id < 48`.
- The adapter should optionally decode with `DiscreteActionMapper` in tests to prove the rule action remains inside the existing contract.

No new simulator rule:

- The adapter must call `env.step` on the existing `FivePLDigitalTwinEnv`.
- It must not implement route physics, dispatch physics, inventory physics, reward physics, or scenario overrides.

## 7. Continuous Action Handling Strategy

Use a fixed neutral normalized continuous vector:

```text
[0.0, 0.0, 0.0, 0.0, 0.0]
```

This is a normalized PPO-space vector of length 5. Through `ActionProjector`, it maps to midpoint physical controls:

- reorder fraction: 0.5
- dispatch intensity: 0.5
- speed multiplier: 1.0
- safety stock multiplier: midpoint of the configured range
- capacity buffer fraction: midpoint of the configured range

Why this is the safest default:

- It is deterministic.
- It does not load or imitate the PPO policy.
- It avoids scenario-specific hand tuning.
- It isolates the benchmark interpretation around discrete tactical rule quality.
- It is simple to test for length, finiteness, and bounded values.

Fairness caveat:

- A neutral PPO-like vector is not a learned continuous control policy. Rule baselines can fairly compare discrete tactical choices and downstream simulator outcomes under the same scenario loop, but they cannot claim parity with the hierarchical model's learned PPO continuous controls.

Rejected alternatives:

- Scenario-specific continuous defaults: too close to hand-tuned policy design before evidence.
- Environment default projector values hidden in the harness: less explicit than a fixed vector.
- Production PPO continuous controls paired with rule DQN actions: would mix learned and rule policies and make interpretation messy.

## 8. Scenario Loop Design

The future skeleton should not run the full suite. It should only expose testable functions.

Future approved episode loop:

1. Load scenario configs using the existing scenario loader.
2. Build a scenario environment using the existing environment builder.
3. Reset with an approved seed.
4. Build a rule context from the current snapshot.
5. Select a rule action id.
6. Step the environment with neutral continuous controls plus the rule action id.
7. Feed `reward` and `info` to `EpisodeMetricAccumulator.observe`.
8. Repeat until termination/truncation or an explicitly approved max step limit.
9. Return `EpisodeMetrics` in memory.

Skeleton-only phase:

- Unit tests may use synthetic snapshots or small fake contexts.
- Unit tests may instantiate adapter helpers if they do not run full scenario eval or write output.
- No benchmark report and no scenario output directory should be created.

## 9. Output Schema

The future benchmark output, only after explicit run approval, should be a fresh report under `reports/rule_based_baselines/...` unless the user explicitly approves `models/eval/...`.

Proposed benchmark report schema:

- `report_metadata`
  - `created_at`
  - `contract`
  - `observation_dim`
  - `external_discrete_action_count`
  - `continuous_action_strategy`
  - `scenario_dir`
  - `episodes`
  - `seed`
- `baseline_runs`
  - `baseline_name`
  - `scenario_id`
  - `episodes`
  - metrics copied from scenario summaries
- `action_metrics`
  - action distribution
  - top actions
  - action 24 rate
  - action 32 rate
  - no-current/no-unassigned/failed-noop by action
- `fairness_caveats`
- `decision`

Skeleton output:

- No persistent report output.
- The future skeleton report should document implementation/tests only.

## 10. Metric Compatibility

Use existing scenario metrics wherever possible:

- service level
- true lateness pressure
- dispatch rate
- dispatch success per attempt
- no-current/no-unassigned by action
- failed-noop dispatch by action
- route failure by action
- no vehicle by action
- already assigned by action
- route distribution
- fleet distribution
- reorder mode distribution
- top action concentration
- full action distribution
- action 24 and action 32 rates
- secondary_fleet rate
- reorder none rate

The key compatibility seam is `EpisodeMetricAccumulator`, which consumes environment `info` dictionaries. If the rule harness calls the same `env.step` path, those metrics remain comparable by construction.

## 11. Fairness Caveats

Fairly comparable:

- same scenario configs;
- same seeds;
- same episode budgets;
- same environment physics;
- same action id contract;
- same metrics accumulator;
- scenario-level service and lateness;
- dispatch success and dispatch rate;
- no-current/no-unassigned/failed-noop behavior;
- route failure and no-vehicle counters;
- action concentration and route/fleet/reorder distributions.

Not fairly comparable:

- learned PPO continuous control quality;
- training sample efficiency;
- Q values, logits, or action probabilities;
- exact production long-run gate semantics unless a separate rule-baseline comparator gate is approved;
- real-world secondary-fleet economics;
- real-world reorder economics;
- company-data replay outcomes.

The report should explicitly say that rule baselines are transparent heuristics inside the simulator contract, not learned joint PPO/DQN substitutes.

## 12. Fresh-Output-Dir Policy

Skeleton phase:

- no `models/eval` writes;
- no `reports/rule_based_baselines` benchmark outputs;
- no existing eval output edits.

Future benchmark phase, only after explicit approval:

- prefer a fresh `reports/rule_based_baselines/<run_id>/` directory;
- write no protected artifacts;
- if the user wants eval-equivalent outputs, use a fresh `models/eval/<approved_rule_baseline_run_id>/` only with explicit approval;
- stop if the target output directory already exists.

Existing `models/eval/**` is immutable audit evidence.

## 13. Tests Required

Required future skeleton tests:

- neutral continuous action returns length 5, finite values, and all values in normalized `[-1, 1]`;
- rule action validation accepts 0..47 and rejects out-of-range actions;
- adapter encodes `RuleBasedDecision.action_id` as `{"continuous": vector, "discrete": action_id}`;
- adapter does not load checkpoints;
- adapter does not write output directories;
- context builder handles an empty/no-work snapshot by choosing hold through the relevant baseline;
- action 24 and action 32 rule decisions remain legal external actions;
- a tiny fake-step or mocked-env loop calls `EpisodeMetricAccumulator.observe` without writing files;
- future CLI, if added later, refuses existing output dirs and refuses unapproved `models/eval` paths.

Do not run offline eval as a test substitute. Unit tests must stay focused on adapter behavior.

## 14. Future Implementation File List

Future skeleton goal should allow only:

- `src/eval/rule_based_baseline_arena.py`
- `tests/eval/test_rule_based_baseline_arena.py`
- `docs/runs/20260613_rule_based_baseline_eval_harness_skeleton_report.md`
- `docs/00_PROJECT_DASHBOARD.md` only for links/status if useful

Optional later files, not part of the skeleton unless separately approved:

- `scripts/evaluate_rule_based_baselines.py`
- `reports/rule_based_baselines/<fresh_run_id>/...`

## 15. Future Benchmark Run Approval Checklist

Before any actual rule-baseline benchmark or eval run:

- explicit user approval for benchmark/eval execution;
- exact baseline names;
- exact scenario directory;
- exact episode count;
- exact seed list or seed base;
- exact output directory;
- proof output directory is absent;
- protected registry, production, baseline, DB, checkpoint, and eval-output hashes/profiles captured;
- no train/eval/gate/AWS/private-data process already running;
- confirmation that `evaluate_real_world_scenarios.py` and `long_run_gate.py` are not being called unless specifically approved;
- independent review if the run will be used for thesis or production claims.

## 16. Non-Goals

- No training.
- No offline eval.
- No long-run gate.
- No benchmark execution.
- No dependency installation.
- No dataset download.
- No registry update.
- No `active_models.json` or `models.jsonl` mutation.
- No production mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No existing eval-output edits.
- No training config creation.
- No private company-data ingestion.
- No firm-facing data request email.
- No 1.5M/2M/3M/5M/10M/100M run.
- No new simulator.
- No duplicated env physics.
- No claim that rule policies validate real-world route, fleet, or reorder economics.

## 17. Stop Conditions

Stop the future skeleton task if:

- implementation requires changing `env_5pl` physics;
- implementation requires changing scenario JSON files;
- implementation requires loading production checkpoints;
- implementation tries to run `evaluate_real_world_scenarios.py`;
- implementation tries to run `long_run_gate.py` or a benchmark command;
- implementation tries to write under `models/eval`;
- implementation tries to write benchmark results;
- implementation tries to mutate registry, production, baselines, DB, checkpoints, or existing eval outputs;
- any protected hash/profile drifts unexpectedly;
- tests reveal the rule action adapter can emit an invalid action id;
- the harness cannot produce metrics through `EpisodeMetricAccumulator`.

## 18. Recommended Future /goal Command

Use this exact future command when implementation is approved:

```text
/goal Read docs/goals/20260613_execute_rule_based_baseline_eval_harness_skeleton_goal.txt and execute it exactly.
```

That goal is skeleton-only. It does not authorize scenario eval runs, long-run gates, benchmark result generation, `models/eval` writes, or protected mutation.
