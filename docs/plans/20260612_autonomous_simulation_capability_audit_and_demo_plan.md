# Autonomous Simulation Capability Audit and Demo Plan

Date: 2026-06-12

Status: audit and plan only. No training, offline evaluation, gate run, dataset download, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output mutation, private-data ingest, or training-config creation was performed.

Recommended classification: `AUTONOMOUS_SIMULATION_ALREADY_COVERED_BY_EXISTING_RUNTIME`

## Executive Summary

The current hierarchical v1 1M model was already trained and evaluated as an autonomous decision policy inside the existing 5PL digital-twin stack. The core autonomous loop is already present:

1. `src/act/env_5pl.py` simulates the 5PL operations state and applies PPO/DQN joint actions.
2. `configs/eval_scenarios/*.json` define stress regimes, thresholds, and scenario expectations.
3. `src/eval/real_world_scenario_arena.py` runs closed-loop policy inference against the simulator until episode termination.
4. `src/orchestration/torch_joint_runtime.py` and `src/orchestration/policy_service.py` provide runtime-compatible torch joint inference for obs 73 and actions 0..47.
5. `scripts/monitoring_report_generator.py` and `scripts/run_production_ops_bundle.py` provide reporting, health, and assurance aggregation around the production model, but they are not simulators.

A new full synthetic operations sandbox would duplicate `env_5pl` and the scenario arena. The safest Step 6 is no new sandbox. If a stakeholder-facing demonstration is needed later, it should be a docs-only demo guide first, using existing evidence and artifacts. A thin read-only demo runner over the existing environment/runtime may be considered later only if there is a concrete presentation or integration need that existing reports do not satisfy.

## Current Autonomous Capability Summary

The production hierarchical v1 1M model is already an autonomous decisioning policy under the existing external contract:

- Logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Production checkpoint: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- Architecture: `hierarchical_v1`
- Initialization: `flat_teacher_distillation_v1`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: 73
- External action count: 48
- Production handoff: `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- Ops bundle status: `OPS_BUNDLE_CLEAN` in `docs/runs/20260611_step5_production_ops_bundle_report.md`

The ladder evidence says the model passed 250k, 500k, and 1M gates. The release handoff and monitoring runbook document that the 1M candidate passed 8/8 offline scenarios, 160 episode rows, hard blockers zero, and long-run gate PASS. The later equal-budget residual-watch assurance accepted the remaining route/mixed warnings.

## Existing Components

### `env_5pl`

`src/act/env_5pl.py` is the primary synthetic operations simulator. It wraps the SimPy 5PL digital twin in a Gymnasium-style environment with `continuous`, `discrete`, and `joint` action modes.

It already simulates the operational elements needed for autonomous decisioning:

- Hub inventory, replenishment, capacity, stockout and holding-cost pressure.
- Pending and delivered orders, lateness, premium/urgent demand pressure, and rolling demand generation.
- Primary and secondary fleet capacity, speed, dispatch availability, vehicle assignment, and route execution.
- Route candidates and route pressure signals for shortest, low-congestion, and high-resilience choices.
- Scenario stressors such as demand spikes, lead-time volatility, holding-cost pressure, premium SLA pressure, vehicle scarcity, route disruption, congestion, and mixed stress.
- Joint PPO/DQN actions, where PPO controls physical continuous levers and DQN chooses dispatch/route/fleet/reorder macro actions.
- Step-level telemetry through `info`, including projected actions, projection reasons, reward components, rolling-demand counts, and simulation snapshots.

This is already the synthetic operations sandbox. Rebuilding it under another name would create duplicated physics and a high risk of semantic drift.

### Observation Builder

`src/act/observation_builder.py` defines the 73-dimensional observation contract. It converts the digital-twin snapshot and scenario stress state into normalized policy features, including:

- legacy service/cost/inventory/route/fleet signals,
- real-world stress signals,
- dispatch feasibility,
- route-candidate visibility,
- route pressure and secondary fleet safety/risk indicators.

This is the feature bridge between the simulator and policy runtime.

### Discrete Action Mapper

`src/act/discrete_action_mapper.py` defines the 48-action external contract as:

- 2 dispatch choices,
- 3 route choices,
- 2 fleet choices,
- 4 reorder choices.

This preserves the action 0..47 runtime contract. In the residual-watch context, action 24 is the dispatch + shortest-route + secondary-fleet + no-reorder combination, and action 32 is the dispatch + low-congestion-route + secondary-fleet + no-reorder combination.

### Eval Scenarios

`configs/eval_scenarios/*.json` already define the scenario suite used for autonomous stress testing:

- `baseline_normal`
- `demand_spike_volatility`
- `high_holding_cost`
- `lead_time_volatility`
- `mixed_stress`
- `premium_sla_pressure`
- `route_disruption_congestion`
- `vehicle_scarcity_capacity_shock`

Each scenario carries environment overrides, expected behavior, thresholds, and default seeds/episode counts. These files are the scenario-control layer for the existing synthetic operations loop.

### Real-World Scenario Arena

`src/eval/real_world_scenario_arena.py` provides the autonomous rollout loop over the simulator:

1. Load the checkpoint and validate metadata.
2. Build the scenario-specific `FivePLDigitalTwinEnv`.
3. Reset the environment.
4. Select PPO and DQN actions from the policy at each step.
5. Apply the joint action to the environment.
6. Accumulate service, lateness, dispatch, route, blocker, and action-distribution metrics until the episode ends.

`src/eval/evaluate_real_world_scenarios.py` is the CLI wrapper that writes scenario summaries, episode metrics, CSV, and markdown reports. Running it normally is an offline eval, so it is out of scope for this planning task, but its implementation proves that the autonomous simulation loop already exists.

### Torch Joint Runtime and PolicyService

`src/orchestration/torch_joint_runtime.py` provides checkpoint loading and deterministic runtime inference for a single observation. It validates obs 73, continuous action length 5, finite bounded continuous outputs, and discrete action 0..47.

`src/orchestration/policy_service.py` resolves active registry models and serves `predict_joint(...)` with `algorithm=torch_joint`. This is the production-facing decision API path. It is not itself an environment loop; it expects observations from a caller such as a simulator, service, or future telemetry adapter.

### Monitoring Report Generator

`scripts/monitoring_report_generator.py` consumes artifact-health status and telemetry-like JSON/CSV rows or synthetic built-in fixtures. It computes:

- protected-state status,
- scenario metrics,
- action metrics,
- residual watches,
- alerts,
- monitoring decision classification.

It is a reporting layer, not a simulator. It should remain downstream of either production telemetry, synthetic fixtures, or future demo outputs.

### Production Ops Bundle

`scripts/run_production_ops_bundle.py` is a convenience aggregator over safe read-only operational tools. It combines artifact health, synthetic monitoring report output, company-data validator status, protected-state checks, process scan, and a final ops decision. It is not an autonomous simulation loop and should not acquire simulation responsibilities.

## Audit Questions

1. Was the current model already trained for autonomous decisioning in the 5PL simulator?

   Yes. The hierarchical v1 1M model was trained and gated through the existing joint PPO/DQN 5PL digital-twin stack. The release handoff and ladder report document autonomous scenario/gate success under obs 73/action 48.

2. What exactly does `env_5pl` already simulate?

   It simulates inventory, demand, orders, lateness, fleet availability, primary/secondary dispatch, route choice, congestion/disruption, capacity shock, replenishment, premium SLA pressure, rolling demand, physical action projection, discrete action projection, reward components, and episode termination/truncation.

3. What does `real_world_scenario_arena` already provide as an autonomous loop?

   It provides the closed-loop evaluator: checkpoint load, scenario environment construction, repeated policy action selection, environment stepping, metric accumulation, and pass/fail evidence generation.

4. What does `PolicyService` provide as runtime decisioning?

   It provides active-registry resolution and single-observation torch joint inference through `predict_joint(...)`, returning PPO continuous controls plus DQN discrete action with `algorithm=torch_joint`. It is the runtime decision API, not a simulator.

5. What does `monitoring_report_generator` already provide for telemetry/reporting?

   It provides telemetry/report aggregation, residual-watch computation, action 24/32 monitoring fields, protected-state integration, and report classifications. It does not generate environment transitions.

6. Is a new synthetic operations sandbox redundant?

   Yes for autonomous simulation capability. `env_5pl` plus `real_world_scenario_arena` already cover the synthetic operations loop. A new sandbox would duplicate physics, scenario controls, and action semantics.

7. If not redundant, what exact gap does it fill?

   A new full sandbox is not justified by current evidence. The only gap is presentation and ergonomics: a short, clearly named, non-eval demo surface could help humans see one bounded autonomous rollout without interpreting the evaluation pipeline. That gap does not require a new simulator.

8. Could a thin demo runner over existing eval/runtime provide the desired autonomous simulation without duplicating `env_5pl`?

   Yes, if later approved. A thin runner should import existing `FivePLDigitalTwinEnv`, scenario configs, `PolicyService` or `load_torch_joint_policy`, and write a small demo report under a fresh `reports/demo/...` path. It should not implement physics, new rewards, new scenarios, training configs, registry writes, production writes, checkpoint writes, baseline writes, DB writes, or eval-output edits.

9. What would be the safest next step?

   The safest next step is no new code for simulation. Create or maintain a docs-only demo guide if a human-facing demo is needed. A thin read-only demo runner is second choice and should be opened only after a specific need is documented.

10. What should remain out of scope until company data exists?

   Real secondary-fleet economics, reorder/no-reorder economic optimality, carrier availability, dispatch failure reason distributions, cost calibration, actual demand/lead-time calibration, and production-telemetry validation remain out of scope until company data is available under a separate approved intake path.

## Gap Analysis

### Covered

- Autonomous simulation environment: covered by `env_5pl`.
- Scenario stress suite: covered by `configs/eval_scenarios/*.json`.
- Closed-loop policy rollout: covered by `real_world_scenario_arena`.
- Runtime checkpoint loading and inference: covered by `torch_joint_runtime`.
- Active-registry runtime decision path: covered by `PolicyService`.
- Residual-watch telemetry reporting: covered by `monitoring_report_generator`.
- Operational bundle aggregation: covered by `run_production_ops_bundle`.

### Partial Coverage

- Human-facing demo ergonomics are partial. Existing reports are rigorous but built for audit/evaluation, not quick demonstration.
- `real_world_scenario_arena` uses direct checkpoint policy loading. `PolicyService` is validated separately for active-registry runtime inference, but a demo path that explicitly shows `PolicyService -> env_5pl -> metrics` does not currently exist.
- Monitoring reports can consume telemetry-like rows, but there is no live production telemetry ingestion and no company-data ingestion by design.

### Not Covered, Deliberately

- Private company-data validation of fleet, route, cost, and reorder economics.
- Live telemetry ingestion.
- Baseline updates.
- DB cleanup or DB mutation.
- New reward/config/training experiments.
- Longer 1.5M/2M/3M/5M/10M/100M training ladders.

## Recommendation

Do not build a new synthetic operations sandbox.

The current simulator, scenario arena, runtime loader, PolicyService, monitoring report generator, and ops bundle already cover the autonomous simulation and operational assurance stack. The cleanest architecture is to preserve those responsibilities:

- `env_5pl` owns synthetic operations physics.
- `real_world_scenario_arena` owns closed-loop scenario rollouts.
- `PolicyService` owns runtime decisioning.
- `monitoring_report_generator` owns telemetry/report classification.
- `run_production_ops_bundle` owns local safe operational aggregation.

If a future demo is required, start with a docs-only demo guide that points to the existing production handoff, ladder report, scenario outputs, monitoring report, and ops bundle. Only build a thin read-only demo runner if the demo guide is insufficient.

Future goal file is not justified now. If later requested, the exact proposed goal file name is:

`docs/goals/20260612_execute_autonomous_simulation_demo_guide_goal.txt`

If a runnable demo is later approved instead of a docs-only guide, it should be planned under a separate explicit goal and should reuse existing components rather than introducing new simulation physics.

## Demo Plan

### Docs-Only Demo Guide

A docs-only guide would demonstrate the autonomous loop using existing artifacts:

- Production model identity and checkpoint from the release handoff.
- Scenario suite from `configs/eval_scenarios/*.json`.
- Closed-loop flow from `real_world_scenario_arena`.
- Runtime inference path from `PolicyService` and `torch_joint_runtime`.
- Existing 1M eval evidence and equal-budget residual-watch evidence.
- Monitoring and ops bundle reports for production assurance.

This guide would not run eval, train, mutate protected artifacts, download data, or ingest private data.

### Optional Thin Demo Runner, Only If Later Approved

If a runnable demo is needed later, it should be a thin wrapper with the following constraints:

- Use existing `FivePLDigitalTwinEnv`, scenario configs, and torch joint runtime or `PolicyService`.
- Run a bounded, explicitly demo-labeled rollout.
- Write only to a fresh `reports/demo/...` directory.
- Produce a small JSON/markdown summary with step count, action 24/32 rates, service, lateness, dispatch success, and hard-blocker counters.
- Avoid writing to `models/eval`, registry, production, baselines, DB, checkpoints, or training configs.
- Avoid calling it an offline eval or long-run gate.
- Avoid changing physics, reward, action mapping, observation mapping, or scenario definitions.

This runner would be useful only for presentation ergonomics. It is not needed to prove autonomous capability.

## Non-Goals

- No training.
- No offline evaluation.
- No long-run gate.
- No dataset download.
- No registry update.
- No `active_models.json` or `models.jsonl` mutation.
- No production mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No edits to existing eval outputs.
- No private company-data ingestion.
- No training-config creation.
- No new simulation physics.
- No replacement for `env_5pl`.
- No baseline update or production rollback.

## Protected No-Mutation Proof

Pre-write protected state was captured before this plan was created:

- `models/registry/active_models.json`: SHA256 `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: SHA256 `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: SHA256 `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: SHA256 `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`
- `models/baselines`: 2692 files, 7392576274 bytes
- `db`: 7 files, 28330 bytes
- `models/checkpoints`: 787 files, 4920830666 bytes
- `models/production`: 8 files, 328265276 bytes
- `models/eval`: 128 files, 388267659 bytes

Process scan before writing found no matching train/eval/gate/AWS/private-data process patterns.

The only planned write for this task is this file:

`docs/plans/20260612_autonomous_simulation_capability_audit_and_demo_plan.md`

## Reviewer

Independent read-only reviewer verdict pending. Expected verdict set:

- `AUTONOMOUS_SIMULATION_CAPABILITY_PLAN_APPROVED`
- `AUTONOMOUS_SIMULATION_CAPABILITY_PLAN_NEEDS_FIXES`
- `AUTONOMOUS_SIMULATION_CAPABILITY_PLAN_BLOCKED`

## Final Planned Classification

`AUTONOMOUS_SIMULATION_ALREADY_COVERED_BY_EXISTING_RUNTIME`
