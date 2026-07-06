# Operation Walkthrough Traceability Audit

Date: 2026-06-12

## Scope

Prepared Turkish thesis-advisor walkthrough files for one detailed operation narrative, one simplified version, and diagram notes.

Allowed writes used:

- `docs/reports/20260612_tez_danismani_operasyon_walkthrough_detayli.md`
- `docs/reports/20260612_tez_danismani_operasyon_walkthrough_sade_versiyon.md`
- `docs/reports/20260612_tez_danismani_operasyon_walkthrough_diyagram_notlari.md`
- `docs/runs/20260612_operation_walkthrough_traceability_audit.md`
- `docs/00_PROJECT_DASHBOARD.md`

## Files Read

Project state and advisor context:

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/reports/20260612_tez_danismani_turkce_proje_kabiliyet_raporu.md`
- `docs/reports/20260612_tez_danismani_turkce_sade_ozet.md`
- `docs/reports/20260612_tez_danismani_turkce_sunum_notlari.md`
- `docs/reports/20260612_thesis_advisor_project_capability_report.md`
- `docs/runs/20260612_1m_learning_efficiency_and_sufficiency_audit.md`
- `docs/plans/20260612_autonomous_simulation_capability_audit_and_demo_plan.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runs/20260611_step3_monitoring_report_generator_report.md`
- `reports/monitoring/20260611_hierarchical_v1_synthetic_monitoring_report.json`

Core implementation files:

- `src/act/discrete_action_mapper.py`
- `src/act/action_projector.py`
- `src/act/safety_projector.py`
- `src/act/observation_builder.py`
- `src/act/env_5pl.py`
- `src/act/routing_physics.py`
- `src/act/inventory_dynamics.py`
- `src/think/joint_policies.py`
- `src/eval/real_world_scenario_arena.py`
- `src/eval/evaluate_real_world_scenarios.py`
- `src/orchestration/policy_service.py`
- `src/orchestration/torch_joint_runtime.py`
- `scripts/monitoring_report_generator.py`

Scenario and evidence files:

- `configs/eval_scenarios/baseline_normal.json`
- `configs/eval_scenarios/demand_spike_volatility.json`
- `configs/eval_scenarios/high_holding_cost.json`
- `configs/eval_scenarios/lead_time_volatility.json`
- `configs/eval_scenarios/mixed_stress.json`
- `configs/eval_scenarios/premium_sla_pressure.json`
- `configs/eval_scenarios/route_disruption_congestion.json`
- `configs/eval_scenarios/vehicle_scarcity_capacity_shock.json`
- `models/eval/equal_budget_hierarchical_v1_20ep_seed42_20260611/scenario_summary.json`
- `models/eval/equal_budget_hierarchical_v1_20ep_seed42_20260611/episode_metrics.jsonl`
- `models/eval/equal_budget_oldprod_20ep_seed42_20260611/scenario_summary.json`
- `models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios/scenario_summary.json`
- `models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios/episode_metrics.jsonl`
- `reports/ops/20260611_hierarchical_v1_ops_bundle_report.json`

## Exact Step-Level Trace Found?

No.

The evaluator loop in `src/eval/real_world_scenario_arena.py` calls PPO and DQN each step, applies the joint action through `env.step(...)`, and feeds `reward` plus `info` into `EpisodeMetricAccumulator`. The persisted eval outputs are `episode_metrics.jsonl`, `scenario_summary.json`, `scenario_summary.csv`, and report markdown. The available eval artifacts contain per-episode and per-scenario aggregate action/failure/reward counters, not full step-level raw actions, raw PPO vectors, DQN Q/logit values, selected route paths, or per-step `info` payloads.

Therefore the operation walkthroughs explicitly label action 24 and action 32 narratives as **kodla uyumlu temsili operasyon ornegi**, not as actual logged step traces.

## How Examples Were Chosen

Action 24 was selected because it is the accepted mixed-stress residual watch:

- decode: `dispatch + shortest + secondary_fleet + none`;
- equal-budget mixed stress scenario PASS;
- service `0.9423472160631124`;
- action 24 attempts `3342`;
- action 24 percentage `0.5802083333333333`;
- action 24 no-current-work `0`;
- action 24 no-unassigned `0`;
- action 24 failed-noop `0`;
- action 24 dispatch success ratio `1.0`;
- residual risk remains around concentration, no_vehicle, already_assigned, and route_failure watch counters.

Action 32 was selected as the route-disruption comparison:

- decode: `dispatch + low_congestion + secondary_fleet + none`;
- equal-budget route disruption scenario PASS;
- service `0.9007785656472469`;
- action 32 attempts `2737`;
- action 32 no-current-work `0`;
- action 32 no-unassigned `0`;
- action 32 failed-noop `0`;
- action 32 no_vehicle `0`;
- action 32 dispatch success ratio `1.0`;
- residual risk remains around concentration, route_failure, already_assigned, and real-world secondary-fleet economics.

## No-Mutation Proof: Pre-Write Protected Hashes

Individual protected files before documentation writes:

| Path | Size | SHA256 |
| --- | ---: | --- |
| `models/registry/active_models.json` | `314` | `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A` |
| `models/registry/models.jsonl` | `14142` | `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt` | `303413903` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json` | `4084` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` |

Protected tree profiles before documentation writes:

| Root | File count | Size | Profile SHA256 |
| --- | ---: | ---: | --- |
| `models/production` | `8` | `328265276` | `B50B76B4138EB9F3C484FB065B297A2295B2FD4F8361860C473F4D9D6E69F35A` |
| `models/baselines` | `2692` | `7392576274` | `10E9BE7A3E97164D3278FBFA02AA4A10726ED918D49A0E900DCBBA765F4D43F3` |
| `db` | `7` | `28330` | `802070D256321A513B3F50FA9F9010090253319E6811354DB1C7D4C5E1DB2045` |
| `models/checkpoints` | `787` | `4920830666` | `AA2C94E8CD4606790E2C249F638D5490C50159E4C5AB8C08B155BDD4B0FF18D3` |
| `models/eval` | `128` | `388267659` | `0E9335CB500140912A9EA09D58A563882B3DC8F00AC0C3832DDA02FDABE4D77C` |

Pre-write process scan found no Python/AWS process rows.

## Verification

Completed final verification after writing:

- required report files exist;
- required sections are present;
- no forbidden train/eval/gate/AWS process visible;
- protected file hashes unchanged;
- protected tree profiles unchanged except allowed docs paths;
- independent reviewer verdict captured.

## Independent Reviewer Verdict

OPERATION_WALKTHROUGH_APPROVED

## Final Classification

THESIS_OPERATION_WALKTHROUGH_READY
