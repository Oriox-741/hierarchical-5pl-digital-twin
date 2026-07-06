# Thesis Advisor Project Capability Report

Date: 2026-06-12

Status: advisor-facing capability package. This report is documentation only. No training, offline evaluation, long-run gate, dataset download, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output edit, private-data ingest, or training config creation was performed.

## Executive Summary

CODEX PROJE is a Python-native research and prototype system for autonomous 5PL logistics decisioning inside a digital-twin simulator. It combines a SimPy/Gymnasium 5PL environment, a synchronized PPO+DQN reinforcement-learning policy, an eight-scenario stress-test suite, production-style registry/promotion controls, and monitoring/governance tooling.

The current project production model is:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- runtime family: `torch_joint`
- DQN architecture: `hierarchical_v1`
- initialization: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: `73`
- continuous action dimension: `5`
- external discrete action count: `48`
- production state: active in registry and copy-only promoted into `models/production`
- gate status: 250k PASS, 500k PASS, 1M PASS
- equal-budget residual-watch gate: PASS
- ops bundle: `OPS_BUNDLE_CLEAN`

Advisor-safe thesis claim:

> This project demonstrates an autonomous decision-control architecture for a 5PL digital-twin logistics environment. It can sense simulated operational state, choose continuous strategic controls and discrete tactical logistics actions, evaluate itself under multiple stress scenarios, and manage model promotion/monitoring through a guarded MLOps workflow. It is simulator-production-ready inside the project contract, but it is not yet a real-world autonomous logistics deployment because company data, live TMS/WMS integration, and real fleet/reorder economics remain unvalidated.

## Project Definition

The project is a research-grade autonomous control tower for fifth-party logistics. It is not just a vehicle-routing solver. It models an integrated logistics operating loop:

- Sense: state snapshots, telemetry abstractions, inventory/order/fleet state.
- Think: neural policies, reward logic, feasibility checks, action decomposition.
- Act: a 5PL digital twin environment that applies dispatch, routing, fleet, reorder, inventory, capacity, and disruption dynamics.
- Learn: synchronized PPO+DQN training, curriculum, replay, exact resume, scenario metrics, and gates.
- Govern: registry, production promotion, monitoring, artifact-health reporting, company-data schema validation, and ops bundle checks.

The current strongest truth-source docs are:

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `docs/runs/20260612_1m_learning_efficiency_and_sufficiency_audit.md`
- `docs/plans/20260612_autonomous_simulation_capability_audit_and_demo_plan.md`
- `docs/runs/20260611_full_project_chronology_and_future_roadmap.md`
- `docs/plans/20260611_codex_project_future_strategy_plan.md`

The root `README.MD` is historically useful but stale because it still references `physical_reality_v2`; current truth is the v5 route-candidate visibility contract in the dashboard, handoff, configs, and run reports.

## What Problem It Solves

The project addresses this research problem:

> Can a reinforcement-learning based control tower coordinate inventory, dispatch, route choice, fleet mode, and reorder behavior in a 5PL digital twin while staying inside operational safety and governance constraints?

Within the simulator, it solves a joint decision problem:

- When should the system dispatch versus hold?
- Which route family should be used?
- Should primary or secondary fleet be used?
- Should reorder mode be none, conservative, aggressive, or emergency?
- How should continuous strategic controls tune inventory, capacity, and operating posture?
- Does the policy remain safe under baseline, demand spike, high holding cost, lead-time volatility, premium SLA pressure, route disruption, vehicle scarcity, and mixed stress?

## What The System Can Do Now

The product can currently do the following inside the project scope:

1. Run an autonomous policy loop in a 5PL digital-twin simulator.
2. Build 73-dimensional observations from inventory, order, route, fleet, and stress-state signals.
3. Produce joint continuous and discrete decisions:
   - PPO continuous action dimension 5,
   - DQN external discrete action count 48.
4. Decode tactical actions into:
   - dispatch or hold,
   - shortest, low-congestion, or high-resilience route,
   - primary or secondary fleet,
   - none, conservative, aggressive, or emergency reorder.
5. Train torch joint PPO+DQN policies through curriculum learning.
6. Initialize hierarchical DQN from a flat production teacher using bounded distillation.
7. Resume exactly from future checkpoints with replay and RNG state.
8. Evaluate trained policies across eight stress scenarios.
9. Enforce long-run gates, hard blockers, dispatch-success checks, service thresholds, and action-quality checks.
10. Register candidates, switch active registry entries, and perform copy-only production promotion under explicit approvals.
11. Load the active torch joint model through `PolicyService` and `torch_joint_runtime`.
12. Generate artifact-health, synthetic monitoring, synthetic company-data schema, and production ops bundle reports.
13. Use public Amazon Last Mile sample data to partially validate route-choice proxy methods for action 24 and action 32.
14. Provide an audit trail, rollback plan, and governance rules for future work.

## What The System Cannot Do Yet

The project does not yet do these things:

1. It is not connected to a real TMS, WMS, ERP, carrier platform, or live operations system.
2. It does not execute real dispatches, reorders, vehicle assignments, or inventory purchases.
3. It does not ingest private company data.
4. It does not prove universal real-world logistics optimality.
5. It does not fully validate `secondary_fleet` economics or reorder-none economics without company data.
6. It does not replace human operational approval for real deployment.
7. It does not justify blind 3M, 5M, or 10M training runs.
8. It does not claim peer-reviewed state-of-the-art performance without external benchmarks, ablations, and company-data validation.

## Architecture Overview

### Digital Twin And Simulator

Key files:

- `src/act/env_5pl.py`
- `src/act/sim_engine.py`
- `src/act/entities.py`
- `src/act/inventory_dynamics.py`
- `src/act/routing_physics.py`
- `src/act/resources.py`
- `src/act/safety_projector.py`
- `src/act/observation_builder.py`
- `configs/eval_scenarios/*.json`

The simulator supports:

- inventory and safety-stock dynamics,
- pending/delivered/late orders,
- rolling stochastic demand,
- premium and urgent demand,
- primary and secondary vehicles,
- dispatch feasibility,
- route traversal, congestion, disruption, and resilience,
- lead-time volatility,
- high holding-cost pressure,
- vehicle scarcity and capacity shock,
- mixed stress.

The current autonomous simulation capability is already covered by `env_5pl` and `real_world_scenario_arena`; a separate synthetic sandbox would duplicate the existing environment.

### Decisioning And Policy Runtime

Key files:

- `src/think/joint_policies.py`
- `src/act/discrete_action_mapper.py`
- `src/orchestration/torch_joint_runtime.py`
- `src/orchestration/policy_service.py`
- `src/orchestration/runtime_loop.py`

The policy is a synchronized torch joint bundle:

- PPO handles continuous strategic controls.
- DQN handles tactical dispatch decisions.
- `hierarchical_v1` keeps the external 48-action contract but internally factorizes Q values into dispatch, route, mode, and reorder heads.

The active runtime path is `PolicyService.predict_joint(...)`, which serves `algorithm=torch_joint` when the active registry points to the hierarchical model.

### Learning And Training

Key files:

- `src/learn/train_joint_torch.py`
- `src/learn/joint_buffers.py`
- `src/learn/curriculum.py`
- `src/learn/joint_metrics.py`
- `src/learn/hierarchical_dqn_initialization.py`
- `src/learn/teacher_retention.py`

Capabilities:

- synchronized PPO+DQN training,
- curriculum stages,
- DQN replay buffer,
- PPO rollout buffer,
- exact resume with replay/RNG state,
- checkpoint metadata,
- flat-teacher distillation for hierarchical initialization,
- teacher-retention diagnostics and failed branch evidence.

The current production model was not learned from scratch in one isolated run. It was warm-started from a passing flat production parent, distilled into a hierarchical DQN, then continued through a 250k -> 500k -> 1M exact-resume ladder.

### Evaluation And Gating

Key files:

- `src/eval/real_world_scenario_arena.py`
- `src/eval/evaluate_real_world_scenarios.py`
- `src/eval/scenario_metrics.py`
- `src/eval/long_run_gate.py`
- `src/eval/check_long_run_gate.py`
- `tests/eval/*`

Evaluation capabilities:

- eight scenario suite,
- deterministic CPU evaluation support,
- episode-level metrics,
- scenario summaries,
- hard-blocker detection,
- service/lateness/dispatch thresholds,
- long-run production-comparison gate,
- residual action-quality warnings.

Current gate evidence:

- 250k hierarchical gate: PASS.
- 500k hierarchical gate: PASS.
- 1M hierarchical gate: PASS.
- 1M offline scenario verdicts: 8/8 PASS.
- Hard blockers: zero.
- Equal-budget old-production vs hierarchical residual-watch gate: PASS.

### Production Lifecycle

Key files:

- `src/learn/register_torch_joint_candidate.py`
- `src/learn/promote_torch_joint_production.py`
- `src/learn/model_registry.py`
- `models/registry/active_models.json`
- `models/registry/models.jsonl`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`

Capabilities:

- candidate registration,
- explicit active-registry activation,
- copy-only production promotion,
- protected manifests and hashes,
- rollback documentation,
- old production retention for audit.

Current active IDs:

- PPO active: `17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- DQN active: `f87e10d6-479f-44fc-99d1-6925bc9cb346`

Current candidate IDs:

- PPO candidate: `3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- DQN candidate: `71af6701-ac3a-40f2-bdd9-cc80868d5a1c`

### Monitoring And Governance

Key files:

- `scripts/production_artifact_health_report.py`
- `scripts/monitoring_report_generator.py`
- `scripts/company_data_intake_validator.py`
- `scripts/run_production_ops_bundle.py`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`
- `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`
- `docs/runbooks/20260611_hierarchical_v1_artifact_health_report_spec.md`
- `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`

Current operational tool status:

- artifact health: `ARTIFACT_HEALTH_CLEAN`
- synthetic monitoring report: `MONITORING_REPORT_CLEAN`
- synthetic company-data schema validator: `COMPANY_DATA_SCHEMA_READY`
- production ops bundle: `OPS_BUNDLE_CLEAN`

These are safe-now governance tools. They do not ingest private company data, do not run training, and do not update production.

## Autonomous Simulation Explanation

The project is autonomous in simulation because the trained policy can observe digital-twin state and act repeatedly without manual decisions inside each episode:

1. `env_5pl` creates the simulated operational world.
2. `observation_builder` converts that world into a 73-dimensional vector.
3. PPO emits continuous controls.
4. DQN emits a 0..47 tactical action.
5. `discrete_action_mapper` decodes that action into logistics decisions.
6. The environment applies safety projection and updates the simulation.
7. Scenario metrics accumulate service, lateness, dispatch, route, no-work, and hard-blocker signals.

This is autonomous simulator decisioning. It is not yet autonomous real-world execution.

## PPO/DQN Explanation For Advisor

PPO and DQN split the control problem into two layers:

- PPO is the strategic continuous controller. It outputs continuous values that tune operating posture, such as capacity/inventory/speed-style controls within the simulator.
- DQN is the tactical discrete controller. It selects one of 48 logistics action combinations.

The 48 DQN actions are structured as:

`2 dispatch choices x 3 route choices x 2 fleet choices x 4 reorder choices = 48`

In `hierarchical_v1`, the DQN is no longer a single flat 48-way head. It has internal factor heads:

- dispatch head,
- route head,
- mode/fleet head,
- reorder head.

It still returns an external 48-Q vector, so existing runtime/eval contracts remain compatible.

## Current Production Model

Current production state is verified through the dashboard, handoff, final production audit, active registry, production manifest, artifact-health report, and ops bundle:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- production directory: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- exact resume capable: true
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: 73
- continuous action dimension: 5
- external discrete action count: 48
- training step: 1,000,000
- eval verdict: PASS
- hard blocker status: zero
- long-run gate verdict: PASS
- residual watches accepted: true
- baseline update: false
- DB mutation: false
- registry mutation during production copy: false

Production files:

| File | SHA256 |
|---|---|
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` |
| `production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` |

## Evaluation Summary

The current 1M hierarchical model passed all project gates. Equal-budget comparison against old production used the same scenario directory, 20 episodes per scenario, seed 42, CPU deterministic execution, and the same gate logic.

| Scenario | Old production service/success | Hierarchical 1M service/success | Result |
|---|---:|---:|---|
| `baseline_normal` | 0.945 / 0.998 | 0.990 / 1.000 | PASS |
| `demand_spike_volatility` | 0.840 / 1.000 | 0.859 / 1.000 | PASS |
| `high_holding_cost` | 0.931 / 0.997 | 0.966 / 0.999 | PASS |
| `lead_time_volatility` | 0.914 / 0.998 | 1.000 / 0.999 | PASS |
| `mixed_stress` | 0.940 / 0.997 | 0.942 / 0.999 | PASS |
| `premium_sla_pressure` | 0.982 / 0.759 | 1.000 / 0.999 | PASS |
| `route_disruption_congestion` | 0.904 / 0.862 | 0.901 / 0.999 | PASS |
| `vehicle_scarcity_capacity_shock` | 0.908 / 0.998 | 0.949 / 1.000 | PASS |

Residual watches:

- route disruption action 32 concentration,
- mixed stress action 24 concentration,
- top-action concentration warnings,
- mixed-success route-failure warnings.

These are accepted for monitoring because the equal-budget gate passed, hard blockers were zero, and the watched action concentrations were not paired with no-current/no-unassigned/failed-noop explosions.

## Academic Positioning

The project sits at the intersection of:

- 5PL digital twins,
- reinforcement learning for logistics control,
- hierarchical/factorized action-space RL,
- multi-objective reward design,
- simulation-to-real calibration,
- safe MLOps for autonomous decision systems.

It is not merely a VRP solver. A VRP solver typically optimizes route sequences under fixed assumptions. This project coordinates route choice alongside inventory, dispatch, fleet mode, reorder behavior, disruptions, and governance.

It is also not yet a peer-reviewed SOTA claim. To make that kind of claim, the project would need:

- benchmark comparisons,
- ablations,
- multiple seeds and statistical reporting,
- company-data validation,
- external reproducibility,
- clearer thesis experiment protocols.

Current academic contribution is best described as a working research prototype and thesis artifact for safe, governed autonomous decisioning in a 5PL digital twin.

## Real-World Validity

The project is close to real operations in structure, not yet in deployment:

- It models realistic logistics concerns: service, lateness, inventory, holding cost, route disruption, vehicle scarcity, premium SLA pressure, and demand spikes.
- It uses a production-like model lifecycle: registry, active model pointers, production manifests, rollback plans, monitoring specs, and artifact-health checks.
- It has public route-side validation support from the Amazon Last Mile small sample.

But it still needs company data for:

- order streams,
- dispatch attempts,
- delivery outcomes,
- route failures,
- carrier/fleet availability,
- inventory/reorder events,
- costs,
- failure taxonomies,
- TMS/WMS/ERP integration semantics.

Advisor-safe wording:

> The project is a validated simulator-production prototype. It is realistic enough for thesis research and controlled demonstrations, but real-world autonomy remains a future validation phase.

## Risk And Limitation Register

| Risk | Current status | Mitigation |
|---|---|---|
| Action 24/32 concentration | Accepted monitoring watch. | Monitor rates and paired failure counters. |
| Simulation-to-real gap | Open. | Company data calibration and KPI mapping required. |
| Secondary-fleet economics | Not proven. | Require carrier/fleet/cost data. |
| Reorder-none economics | Not proven. | Require inventory/reorder/cost data. |
| No live TMS/WMS integration | Not implemented. | Future productization layer and integration tests. |
| Finite scenario suite | Eight scenarios plus equal-budget assurance. | Add validated scenarios only when evidence requires. |
| Complex env/trainer modules | High-complexity core. | Keep tests, docs, and no-blind-training governance. |
| Protected lifecycle complexity | Deliberately strict. | Use runbooks and explicit approvals. |
| Blind longer training | Not justified. | Trigger-based 1.5M/2M only if monitoring/data mismatch appears. |

## Future Roadmap

Immediate:

- Use the ops bundle and monitoring docs to keep production state auditable.
- Maintain dashboard truth-source links.
- Use advisor package for thesis explanation.

Data-dependent:

- Collect company data under the request package.
- Validate schema with the intake validator.
- Map company KPIs to simulator metrics.
- Calibrate route/fleet/reorder economics.

Research-dependent:

- Consider hierarchical v2 only if current residual watches become operationally risky.
- Consider exact-resume 1.5M/2M only under a concrete trigger.
- Do not start blind 3M/5M/10M.

Productization:

- Add API/service wrappers around `PolicyService`.
- Add operator dashboard and health endpoint.
- Add live telemetry ingestion only after privacy/schema approval.
- Add deployment safety gates and shadow-mode testing.

## Advisor Q&A

### Bu proje nedir?

It is a 5PL digital-twin control-tower prototype that uses reinforcement learning to make logistics decisions in simulation and wraps the resulting model in production-style governance.

### Tam olarak hangi problemi cozer?

It studies how an autonomous agent can coordinate inventory, dispatch, route choice, fleet mode, and reorder behavior under logistics stress while avoiding unsafe shortcuts.

### Simulasyonda ne kadar otonom?

Inside the simulator, it is fully closed-loop: observe state, choose actions, apply actions, update the digital twin, and repeat until the episode ends.

### Gercek dunyaya ne kadar yakin?

It captures many real logistics concepts but is not live-connected. Company data is required before making real-world operational claims.

### Why only 1M steps?

The 1M run was not blank-slate learning. It built on a passing production parent, flat-teacher distillation, hierarchical action decomposition, exact replay/RNG resume, and a repeated stress-scenario curriculum. Under this bounded simulator/eval contract, 1M was enough to pass the defined gates.

### Is it fully autonomous?

It is autonomous in the digital twin. It is not yet authorized for autonomous real-world execution.

### Is it real-world valid?

Partially. Route-side behavior has some public-data plausibility support. Fleet, reorder, cost, and operational failure semantics need company data.

### What does PPO do?

PPO controls continuous strategic levers in the simulator.

### What does DQN do?

DQN chooses the discrete logistics action: dispatch/hold, route family, fleet mode, and reorder mode.

### What are action 24 and action 32?

- Action 24: dispatch + shortest route + secondary fleet + no reorder.
- Action 32: dispatch + low-congestion route + secondary fleet + no reorder.

They are accepted residual watches, not ignored issues.

### Why no 3M now?

Previous longer or continuation branches sometimes got worse. More training is not automatically safer. The current model passed; future training should require a monitoring or data trigger.

### What remains for company data?

Company data must validate real order patterns, dispatch failures, vehicle/carrier availability, inventory/reorder economics, route failures, and cost tradeoffs.

### What makes this academically meaningful?

The project combines a digital twin, hierarchical RL action decomposition, scenario-based evaluation, exact-resume stability, and MLOps governance in a single traceable research artifact.

## Final Advisor-Safe Wording

CODEX PROJE is a research-grade autonomous 5PL digital-twin control system. It can train and run a joint PPO+DQN policy that makes continuous strategic and discrete tactical logistics decisions, evaluate that policy under realistic simulated stress scenarios, and manage the approved model through registry, production promotion, monitoring, and governance reports. The current hierarchical v1 1M model is production-promoted within the project and passed the simulator/evaluation gates. The correct limitation is that this proves simulator-production readiness, not real-world autonomous deployment readiness; company data and integration validation are still required before operational use.
