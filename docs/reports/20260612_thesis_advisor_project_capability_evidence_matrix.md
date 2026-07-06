# Thesis Advisor Project Capability Evidence Matrix

Date: 2026-06-12

## Purpose

Advisor-safe claim support for the CODEX PROJE capability report. This matrix separates what the repository proves from what remains a limitation.

## Evidence Matrix

| Claim | Evidence file/path | Supporting metric or fact | Confidence | Limitation | Advisor-safe wording |
|---|---|---|---|---|---|
| The current project production model is hierarchical v1 1M. | `docs/00_PROJECT_DASHBOARD.md`; `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`; `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json` | `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`; `dqn_architecture=hierarchical_v1`; `training_step=1000000` | High | Production is project-scope, not live company deployment. | The project has a promoted simulator-production model. |
| The active registry points to hierarchical v1 artifacts. | `models/registry/active_models.json`; `docs/runs/20260611_final_production_state_readonly_audit.md`; artifact-health report | Active PPO/DQN paths point to `joint_torch_v5_prod_hierarchical_v1_1m_20260611` final artifacts. | High | Registry is local filesystem-backed. | Runtime routing is configured to the hierarchical v1 model in the local project registry. |
| Production copy exists and matches manifest. | `models/production/.../production_manifest.json`; `docs/runs/20260611_final_production_state_readonly_audit.md` | Joint hash `C1E756...`; PPO hash `2D4C...`; DQN hash `9DB0...`; manifest hash `FDAF...` | High | Raw `.pt` bodies were not parsed in this scan. | The production directory contains the approved files with recorded hashes. |
| The model passed the hierarchical ladder. | `docs/runs/20260611_hierarchical_dqn_ladder_report.md`; release handoff | 250k PASS, 500k PASS, 1M PASS; no `--allow-empty-replay-resume`. | High | Ladder is within the simulator/eval contract. | The model passed the project's staged 250k/500k/1M acceptance path. |
| The 1M candidate passed all scenario gates. | `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`; `models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios/scenario_summary.json` | 8/8 PASS, hard blockers zero, long-run gate PASS. | High | Scenario family is finite. | The model passed all defined simulator scenarios and hard-blocker checks. |
| Equal-budget residual-watch assurance passed. | `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`; `models/eval/equal_budget_*` summaries | Both old production and hierarchical had 8 scenarios, 160 rows, 20 episodes/scenario, seed 42; equal-budget gate PASS. | High | Still simulator-only. | The residual watches were rechecked under a fair equal-budget comparison and accepted for monitoring. |
| Route action 32 concentration is accepted for monitoring. | Equal-budget residual-watch report; monitoring runbook | Route disruption action 32 attempts `2737 / 5760 = 47.5%`; service delta `-0.0037` inside gate tolerance; no failed-noop explosion. | Medium-high | Real congestion data still needed. | Action 32 concentration is a watch, not a blocking defect under current evidence. |
| Mixed action 24 concentration is accepted for monitoring. | Equal-budget residual-watch report; monitoring runbook | Mixed stress action 24 attempts `3342 / 5760 = 58.0%`; mixed service slightly better than old production; no no-current/no-unassigned/failed-noop pairing. | Medium-high | Secondary-fleet/no-reorder economics unvalidated. | Action 24 concentration is acceptable in simulation but must be monitored and validated with company data. |
| The autonomous simulator loop already exists. | `docs/plans/20260612_autonomous_simulation_capability_audit_and_demo_plan.md`; `src/act/env_5pl.py`; `src/eval/real_world_scenario_arena.py` | `env_5pl` steps the digital twin; arena repeatedly selects PPO/DQN actions and applies them. | High | It is autonomous in simulation, not live operations. | The project has a closed-loop autonomous simulation path. |
| The policy uses obs 73 and action 48. | Production manifest; `src/act/observation_builder.py`; `src/act/discrete_action_mapper.py` | `observation_dim=73`; `action_dim=48`; action count is 2 x 3 x 2 x 4. | High | Feature/action semantics are simulator-defined. | The decision contract is fixed and inspectable. |
| The DQN architecture is hierarchical/factorized. | `src/think/joint_policies.py`; hierarchical action architecture report | `HierarchicalDQNQNetwork` composes dispatch, route, mode, reorder heads into external Q values. | High | It is v1, not proof of optimal factorization. | The project reduces flat action-pocket risk through a hierarchical DQN design. |
| The first hierarchical candidate was warm-started, not trained from scratch. | `docs/runs/20260611_hierarchical_dqn_initialization_decision.md`; hierarchical config; `src/learn/hierarchical_dqn_initialization.py` | `flat_teacher_distillation_v1`, 4096 sampled observations, 512 distillation steps, PPO/trunk transfer. | High | Distillation uses sampled observations, not company data. | The final 1M benefited from a production-teacher warm start. |
| Exact resume was required for continuation. | `docs/runs/20260610_level2_continuation_exact_resume_patch_report.md`; `src/learn/train_joint_torch.py`; ladder report | Resume restores replay/RNG; 500k and 1M startup logs restored replay sizes. | High | Exact resume alone did not fix all earlier branches. | Stable continuation depends on replay/RNG state, which is now persisted for new checkpoints. |
| Longer blind training is not automatically safer. | `docs/runs/20260612_1m_learning_efficiency_and_sufficiency_audit.md`; `docs/runs/20260610_level2_teacher_retention_vs_hierarchical_architecture_decision.md` | Nextgen, V2.4, V2.5, and teacher-retention branches show longer or different continuation can fail. | High | Future training may be justified by a trigger. | More steps are a research decision, not an automatic improvement. |
| Artifact health is clean. | `reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`; Step 2 report | `ARTIFACT_HEALTH_CLEAN`; registry, manifest, production hashes, process scan PASS. | High | Snapshot from report time. | The promoted artifact set has a clean local health report. |
| Synthetic monitoring generator is available. | `scripts/monitoring_report_generator.py`; Step 3 report; monitoring JSON | `MONITORING_REPORT_CLEAN` from synthetic fixture; action 24/32 watch fields included. | High | Not live telemetry. | Monitoring report shape exists and is ready for future telemetry-like inputs. |
| Synthetic company-data intake validator is available. | `scripts/company_data_intake_validator.py`; Step 4 report; company-data JSON | `COMPANY_DATA_SCHEMA_READY`, 7 table families, required columns and join keys checked. | High | Uses synthetic fixtures only. | The project can validate expected company-data schema before ingesting real data. |
| Ops bundle is clean. | `scripts/run_production_ops_bundle.py`; `reports/ops/20260611_hierarchical_v1_ops_bundle_report.json` | `OPS_BUNDLE_CLEAN`; artifact health clean, monitoring clean, company schema ready. | High | Local wrapper, not live SRE monitoring. | A local read-only ops bundle can summarize production health. |
| Public route data supports route-side plausibility only. | `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`; real-world calibration plan | Amazon small sample supports route proxy readiness for action 24/32 route components. | Medium | Does not validate secondary fleet or reorder economics. | Public route data partially supports route-choice plausibility, not full action economics. |
| Company data is required for full real-world validation. | Real-world calibration plan; company-data request package; monitoring runbook | Needed fields include orders, dispatch attempts, deliveries, routes, fleet, inventory, costs. | High | Data not available in repo. | Real-world autonomy cannot be claimed until company data validates the simulator assumptions. |
| The project has test coverage across key layers. | `tests/act`, `tests/eval`, `tests/learn`, `tests/orchestration`, `tests/real_world_calibration`, `tests/think` | 70 test files; coverage includes reward physics, scenario evaluator, long-run gate, registry, runtime, monitoring, ops bundle. | Medium-high | This scan did not run tests; it inventoried and summarized them. | The repository contains focused tests for the simulator, learning, evaluation, runtime, and governance layers. |
| The root README is not current truth. | `README.MD`; dashboard and handoff docs | README references `physical_reality_v2`; current truth is v5 route-candidate visibility. | High | README still useful for historical architecture. | Use dashboard/handoff/runbooks as current truth; use README as background only. |

## Production State Evidence

- Current production identity is supported by `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`, `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`, and the active registry files.
- The production artifact set is hierarchical v1, initialized by `flat_teacher_distillation_v1`, exact-resume capable, and bound to `physical_reality_v5_route_candidate_visibility`, obs 73, action 48.
- Protected production and registry artifacts were treated as immutable evidence in this scan.

## Evaluation Evidence

- Scenario evidence comes from the hierarchical 1M offline scenario summaries, the hierarchical ladder report, the long-run gate result, and the equal-budget old-production versus hierarchical residual-watch comparison.
- The advisor-safe claim is that the model passed the defined simulator scenarios and gates, not that it is proven optimal in real operations.

## Validation And Monitoring Evidence

- Artifact health, synthetic monitoring, synthetic company-data schema validation, and the local ops bundle are supported by the Step 2 through Step 5 reports and generated JSON outputs.
- Public Amazon Last Mile work supports route-choice plausibility for action 24/32 components only; it does not validate fleet or reorder economics.

## Limitations And Boundaries

- No private company data has been ingested, and no real TMS/WMS deployment is represented in the repository.
- Binary checkpoint bodies were not parsed during this repository scan.
- Older V2/V2.1/V2.2/V2.3/V2.4 documents remain useful postmortem evidence but are not the current production truth source.

## Component Map

| Component | Code files | Docs/evidence | Purpose | Maturity | Risk | Advisor/demo potential |
|---|---|---|---|---|---|---|
| 5PL simulator | `src/act/env_5pl.py`, `src/act/sim_engine.py`, `src/act/entities.py`, `src/act/routing_physics.py` | autonomous simulation plan, eval scenario docs | Simulates logistics operations and stressors. | High for simulator research | High complexity | Good technical demo |
| Observation builder | `src/act/observation_builder.py` | v5 contract docs | Builds obs 73. | High | Feature realism depends on company data | Good architecture diagram |
| Discrete action mapper | `src/act/discrete_action_mapper.py` | calibration plan | Maps 0..47 to dispatch/route/fleet/reorder. | High | Action concentration watches | Excellent advisor explanation |
| Joint policies | `src/think/joint_policies.py` | hierarchical architecture report | PPO actor/critic and DQN flat/hierarchical networks. | High | No external benchmark yet | Good technical deep dive |
| Trainer | `src/learn/train_joint_torch.py` | ladder report, exact-resume patch report | Synchronized PPO+DQN training and checkpointing. | High | Very complex, training not safe-now | Explain, not demo live |
| Hierarchical initialization | `src/learn/hierarchical_dqn_initialization.py` | initialization decision/report | Flat teacher distillation warm start. | High | Synthetic state bank, no company data | Good thesis contribution |
| Scenario arena | `src/eval/real_world_scenario_arena.py` | eval reports | Closed-loop simulator evaluation. | High | Offline eval must be explicitly approved | Explain with existing outputs |
| Long-run gate | `src/eval/long_run_gate.py`, `src/eval/check_long_run_gate.py` | gate results | Read-only scenario/production-comparison gate. | High | Gate family finite | Strong governance story |
| PolicyService | `src/orchestration/policy_service.py` | runtime audits | Active registry inference service. | Medium-high | Not yet integrated with live TMS/WMS | Good productization story |
| Torch runtime | `src/orchestration/torch_joint_runtime.py` | runtime tests/audits | Load/predict torch joint model. | High | CPU/local runtime evidence | Good smoke demo |
| Artifact health | `scripts/production_artifact_health_report.py` | Step 2 report | Validate protected production state. | High | Snapshot-based | Strong ops story |
| Monitoring generator | `scripts/monitoring_report_generator.py` | Step 3 report | Generate monitoring report from synthetic/telemetry-like rows. | Medium-high | Synthetic-only now | Good future telemetry story |
| Company-data validator | `scripts/company_data_intake_validator.py` | Step 4 report | Validate future company-data schema. | Medium-high | No private data yet | Good real-world bridge |
| Ops bundle | `scripts/run_production_ops_bundle.py` | Step 5 report | Aggregate read-only ops reports. | High | Local wrapper | Good advisor demo |
| Public calibration scripts | `scripts/real_world_calibration/*` | Amazon analysis | Route proxy metrics and schema probe. | Medium | Public data partial only | Good calibration evidence |

## Source File List

Primary truth and capability files used:

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `docs/runs/20260612_1m_learning_efficiency_and_sufficiency_audit.md`
- `docs/plans/20260612_autonomous_simulation_capability_audit_and_demo_plan.md`
- `docs/runs/20260611_full_project_chronology_and_future_roadmap.md`
- `docs/plans/20260611_codex_project_future_strategy_plan.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`
- `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`
- `docs/runbooks/20260611_hierarchical_v1_artifact_health_report_spec.md`
- `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`
- `docs/runs/20260611_step2_artifact_health_cli_report.md`
- `docs/runs/20260611_step3_monitoring_report_generator_report.md`
- `docs/runs/20260611_step4_company_data_intake_validator_report.md`
- `docs/runs/20260611_step5_production_ops_bundle_report.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`
- `src/act/env_5pl.py`
- `src/act/observation_builder.py`
- `src/act/discrete_action_mapper.py`
- `src/think/joint_policies.py`
- `src/learn/train_joint_torch.py`
- `src/learn/hierarchical_dqn_initialization.py`
- `src/eval/real_world_scenario_arena.py`
- `src/eval/long_run_gate.py`
- `src/orchestration/policy_service.py`
- `src/orchestration/torch_joint_runtime.py`
- `scripts/production_artifact_health_report.py`
- `scripts/monitoring_report_generator.py`
- `scripts/company_data_intake_validator.py`
- `scripts/run_production_ops_bundle.py`
- `scripts/real_world_calibration/amazon_last_mile_schema_probe.py`
- `scripts/real_world_calibration/route_proxy_metrics.py`
- `configs/eval_scenarios/*.json`
- `configs/training_joint_curriculum_v5_prod_hierarchical_v1_*.json`
- `models/registry/active_models.json`
- `models/registry/models.jsonl`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`
- `reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`
- `reports/monitoring/20260611_hierarchical_v1_synthetic_monitoring_report.json`
- `reports/company_data/20260611_company_data_synthetic_schema_report.json`
- `reports/ops/20260611_hierarchical_v1_ops_bundle_report.json`

## Do Not Overclaim

Avoid these claims:

- "The model is proven optimal in real logistics."
- "The system is deployed in a real company."
- "Action 24 and 32 are proven economically optimal."
- "Public route data validates fleet and reorder decisions."
- "1M steps prove no more training is ever useful."
- "The project is a peer-reviewed SOTA logistics benchmark result."

Use these instead:

- "The model is simulator-production-ready under the project contract."
- "The system is active/promoted in the local project registry and production artifact structure."
- "Action 24 and 32 are accepted monitoring watches under equal-budget simulation evidence."
- "Public route data partially supports route-choice plausibility."
- "Company data is required for fleet/reorder economics and real deployment validation."
- "Further training should be trigger-based, not blind scale-up."
