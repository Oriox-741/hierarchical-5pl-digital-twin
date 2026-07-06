# Hierarchical V1 Production Assurance Audit

Date: 2026-06-11

Executive verdict: `HIERARCHICAL_V1_PRODUCTION_ASSURANCE_WARNINGS_ONLY`

Reviewer verdict: `PRODUCTION_ASSURANCE_AUDIT_WARNINGS_ONLY`

## Scope

This was a read-only assurance audit for the current production state:

`joint_torch_v5_prod_hierarchical_v1_1m_20260611`

No training, offline evaluation, registry update, production mutation, baseline mutation, DB mutation, checkpoint mutation, or eval-output edit was performed. The only write was this audit report.

## Source Documents And Artifacts Read

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `docs/runs/20260611_hierarchical_v1_production_promotion_audit.md`
- `docs/runs/20260611_hierarchical_v1_active_registry_activation_audit.md`
- `docs/runs/20260611_hierarchical_dqn_ladder_report.md`
- `docs/runs/20260611_hierarchical_v1_1m_candidate_review_readiness.md`
- `docs/runs/20260611_hierarchical_v1_candidate_registration_audit.md`
- `docs/runs/20260611_hierarchical_v1_registry_candidate_readiness_patch_report.md`
- `docs/runbooks/20260611_hierarchical_v1_production_promotion_plan.md`
- `docs/runbooks/torch_joint_runtime_promotion_checklist.md`
- `docs/runbooks/torch_joint_production_promotion_plan.md`
- `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios`
- `models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`

## Executive Finding

The production state is operationally consistent:

- Active registry points to the hierarchical v1 1M PPO/DQN final artifacts.
- Candidate and active rows exist and carry the expected hierarchical metadata.
- Production directory contains exactly the approved files.
- Production manifest matches the handoff and active registry IDs.
- Production checkpoint metadata matches the v5/73/48 hierarchical contract.
- DQN replay state and RNG state are present.
- Hierarchical internal heads are present in both DQN q-network and target q-network.
- `load_torch_joint_policy` loads the production checkpoint on CPU.
- `PolicyService.predict_joint(..., fallback_to_heuristic=False)` serves `algorithm=torch_joint` with zero SB3 loader calls.
- Eval artifacts are complete, 8/8 scenarios PASS, 160 episode rows are present, all hard blockers are zero, and long-run gate decision is PASS.

The audit is not classified clean because accepted residual warnings remain real:

- `route_disruption_congestion` service remains below production: `0.901` vs `0.929`.
- `mixed_stress` service remains below production: `0.942` vs `0.951`.
- Action concentration remains high in several scenarios, especially action `24` in `mixed_stress` and action `32` in `route_disruption_congestion`.
- Mixed-success route-failure warnings remain.

These warnings are nonfatal under the current gate evidence: no hard blockers, no threshold failures, no no-current/no-unassigned explosion, no failed-noop explosion, and all scenario verdicts remain PASS.

## Registry Consistency

`models/registry/active_models.json`:

```json
{
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt",
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt"
}
```

Registry hashes:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`

Rows for logical model `joint_torch_v5_prod_hierarchical_v1_1m_20260611`:

- candidate rows: `2`
- active rows: `2`
- PPO candidate id: `3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- DQN candidate id: `71af6701-ac3a-40f2-bdd9-cc80868d5a1c`
- PPO active id: `17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- DQN active id: `f87e10d6-479f-44fc-99d1-6925bc9cb346`

Both active rows include:

- `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `training_step=1000000`
- `eval_verdict=PASS`
- `hard_blocker_status=zero`
- `long_run_gate_verdict=PASS`
- `exact_resume_capable=true`
- `contract=physical_reality_v5_route_candidate_visibility`
- `obs_dim=73`
- `action_count=48`
- `external_discrete_action_count=48`

Result: registry and runtime metadata are consistent.

## Production Artifact Consistency

Production directory exists:

`models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611`

It contains exactly:

- `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `joint_torch_latest.pt`
- `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `production_manifest.json`

Production file hashes:

| File | SHA256 | Bytes |
| --- | --- | ---: |
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` | `303413903` |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` | `4903475` |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` | `4903475` |
| `production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` | `4084` |

Manifest fields match the handoff and active registry:

- `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- `active_ppo_id=17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- `active_dqn_id=f87e10d6-479f-44fc-99d1-6925bc9cb346`
- `candidate_ppo_id=3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- `candidate_dqn_id=71af6701-ac3a-40f2-bdd9-cc80868d5a1c`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `contract=physical_reality_v5_route_candidate_visibility`
- `observation_dim=73`
- `action_dim=48`
- `training_step=1000000`
- `eval_verdict=PASS`
- `hard_blocker_status=zero`
- `long_run_gate_verdict=PASS`
- `residual_watches_accepted=true`
- `baseline_update=false`
- `db_mutation=false`
- `registry_mutation=false`
- `human_approver=egeme`

Result: no production artifact mismatch found.

## Checkpoint Metadata

Production checkpoint:

`models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`

Validated metadata:

- `checkpoint_version=torch_joint_policy_v1`
- `artifact_kind=joint_final`
- `global_step=1000000`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `resume_state.exact_resume_capable=true`
- DQN replay state present: `true`
- DQN replay size: `500000`
- RNG state present: `true`
- contract `physical_reality_v5_route_candidate_visibility`
- observation dimension `73`
- discrete action count `48`

Internal head metadata:

```json
{
  "dispatch": 2,
  "route": 3,
  "mode": 2,
  "reorder": 4,
  "composition": "dispatch + reorder + dispatch_mask(route + mode)"
}
```

Both `model_state_dicts.dqn_q_network` and `model_state_dicts.dqn_target_q_network` contain:

- `dispatch_head.weight`
- `dispatch_head.bias`
- `route_head.weight`
- `route_head.bias`
- `mode_head.weight`
- `mode_head.bias`
- `reorder_head.weight`
- `reorder_head.bias`

Result: hierarchical architecture metadata and internal heads are present.

## Runtime Smoke

`load_torch_joint_policy` loaded the production checkpoint read-only on CPU.

Deterministic zero-observation action:

- continuous length: `5`
- continuous finite: `true`
- continuous bounded in `[-1, 1]`: `true`
- discrete action: `1`
- discrete legal in `0..47`: `true`

`PolicyService.predict_joint(..., fallback_to_heuristic=False)`:

- returned `algorithm=torch_joint`
- continuous length: `5`
- continuous finite and bounded: `true`
- discrete action: `1`
- discrete legal in `0..47`: `true`
- active joint path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`
- active PPO path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- active DQN path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- SB3 PPO loader calls: `0`
- SB3 DQN loader calls: `0`

Result: runtime serves the Torch joint hierarchical checkpoint, not SB3 fallback.

## Eval Artifact Validation

Hierarchical 1M eval directory:

`models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`

Required files present:

- `scenario_summary.json`
- `scenario_summary.csv`
- `episode_metrics.jsonl`
- `real_world_evaluation_report.md`

Eval integrity:

- scenarios: `8`
- episode rows: `160`
- scenario PASS count: `8`
- hard-blocker failures total: `0`
- gate decision: `PASS`
- gate fatal failures: `[]`
- gate warning count: `18`

Explicit hard-blocker counters were zero in every scenario:

- `fake_dispatch_credit`
- `customer_revisited`
- `route_failure_positive_dispatch_credit`
- `nan_inf_detected`
- `dqn_local_negative_positive_train_rows`
- `no_current_work_dqn_delivery_credit`
- `hold_delivery_credit_leak`
- `action8_route_or_delivery_credit_leak`
- `unsafe_24_25_candidate_credit`
- `no_work_positive_dqn_local`
- `emergency_zero_useful_positive_credit`

Result: eval artifacts are complete and hard blockers are clean.

## Scenario Statistics Vs Production

Primary service/lateness/dispatch/no-work values come from the 1M long-run gate comparison. Delivered, route-failure, no-vehicle, and already-assigned values come from eval summary artifacts; the historical production eval has `3` episodes per scenario while hierarchical 1M has `20`, so those counts are raw artifact counts and should not be read as same-sample totals.

| Scenario | svc P->H(delta) | late P->H | disp P->H | success P->H | no-work P->H | delivered P/H | routefail P/H | no_vehicle P/H | already P/H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_normal` | 0.938->0.990 (+0.052) | 0.000->0.000 | 0.487->0.372 | 1.000->1.000 | 0->0 | 466/2907 | 1/3 | 1/1 | 221/1113 |
| `demand_spike_volatility` | 0.843->0.859 (+0.015) | 0.177->0.027 | 0.728->0.793 | 1.000->1.000 | 0->0 | 1114/8911 | 0/13 | 73/833 | 441/3115 |
| `high_holding_cost` | 0.943->0.966 (+0.023) | 0.000->0.000 | 0.387->0.449 | 1.000->0.999 | 0->2 | 399/2919 | 0/8 | 2/0 | 139/1384 |
| `lead_time_volatility` | 0.936->1.000 (+0.064) | 0.000->0.000 | 0.455->0.360 | 1.000->0.999 | 0->1 | 440/3124 | 1/2 | 0/2 | 183/459 |
| `mixed_stress` | 0.951->0.942 (-0.009) | 0.000->0.000 | 0.911->0.987 | 0.996->0.999 | 6->8 | 1246/8391 | 1/10 | 45/293 | 482/3797 |
| `premium_sla_pressure` | 0.980->1.000 (+0.020) | 0.000->0.000 | 0.427->0.357 | 0.769->0.999 | 178->1 | 451/3101 | 0/2 | 0/2 | 107/447 |
| `route_disruption_congestion` | 0.929->0.901 (-0.028) | 0.000->0.008 | 0.575->0.499 | 0.822->0.999 | 171->2 | 479/2926 | 1/8 | 0/0 | 203/1498 |
| `vehicle_scarcity_capacity_shock` | 0.914->0.949 (+0.035) | 0.000->0.001 | 0.731->0.806 | 1.000->1.000 | 0->0 | 744/5000 | 0/8 | 2/2 | 328/2573 |

Improvements:

- Service improves over production in six of eight scenarios.
- Premium no-work falls from `178` to `1`.
- Route-disruption no-work falls from `171` to `2`.
- Demand-spike lateness improves from `0.177` to `0.027`.
- Dispatch success is near-perfect across all H1M scenarios.

Regressions/watches:

- `route_disruption_congestion` service drops by `-0.028`.
- `mixed_stress` service drops by `-0.009`.
- Route-failure artifact counts are higher in H1M, partly on a larger episode sample.
- H1M has higher raw already-assigned context rows, again on a larger episode sample.

## Residual Watch Deep Dive

| Watch | Evidence | Classification |
| --- | --- | --- |
| `route_disruption_congestion` service below production | service `0.901` vs production `0.929`; lateness `0.008`; route failures `8`; no-work improves `171->2`; hard blockers PASS; threshold failures none | warning, nonfatal, accepted |
| `mixed_stress` service below production | service `0.942` vs production `0.951`; lateness `0.000`; route failures `10`; no-work `6->8`; hard blockers PASS; threshold failures none | warning, nonfatal, accepted |
| top-action concentration | action `24` reaches `58.0%` in `mixed_stress`; action `32` reaches `47.5%` in `route_disruption_congestion`; action `1` reaches `63.8%` in `lead_time_volatility` | warning; no fatal blocker or no-op explosion |
| mixed-success route-failure warnings | H1M route-failure artifact counts remain nonzero in all scenarios except none; mixed stress `10`, route disruption `8`, vehicle scarcity `8` | warning; no positive-credit hard blocker |

Conclusion: residual watches are acceptable under the current approved gate and human acceptance, but they are real production monitors. This is why the audit classification is `WARNINGS_ONLY` rather than `CLEAN`.

## Action-Quality Audit

| Scenario | H1M top actions | max action pct | action24/25 conc | failed/no-current/no-unassigned | routefail/no_vehicle | known bad pocket notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `baseline_normal` | 0:2549(44.3%), 1:1071(18.6%), 25:919(16.0%), 24:544(9.4%), 29:268(4.7%) | 44.3% | 25.4% | 0/0/0 | 3/1 | 32:39(0.7%), 33:117(2.0%), 45:1(0.0%) |
| `demand_spike_volatility` | 29:1820(31.6%), 37:1124(19.5%), 1:1119(19.4%), 33:577(10.0%), 45:414(7.2%) | 31.6% | 4.2% | 0/0/0 | 13/833 | 32:7(0.1%), 33:577(10.0%), 41:369(6.4%), 45:414(7.2%) |
| `high_holding_cost` | 0:2745(47.7%), 24:1674(29.1%), 32:466(8.1%), 1:431(7.5%), 28:333(5.8%) | 47.7% | 29.7% | 2/2/0 | 8/0 | 32:466(8.1%), 33:27(0.5%) |
| `lead_time_volatility` | 1:3673(63.8%), 44:1116(19.4%), 36:750(13.0%), 32:82(1.4%), 37:76(1.3%) | 63.8% | 0.0% | 1/1/0 | 2/2 | 32:82(1.4%), 45:47(0.8%) |
| `mixed_stress` | 24:3342(58.0%), 32:1361(23.6%), 28:653(11.3%), 36:321(5.6%), 0:75(1.3%) | 58.0% | 58.0% | 4/4/4 | 10/293 | 32:1361(23.6%), 41:3(0.1%), 45:3(0.1%) |
| `premium_sla_pressure` | 1:2409(41.8%), 28:1809(31.4%), 0:1296(22.5%), 29:204(3.5%), 37:29(0.5%) | 41.8% | 0.0% | 1/1/0 | 2/2 | none |
| `route_disruption_congestion` | 0:2779(48.2%), 32:2737(47.5%), 1:109(1.9%), 36:78(1.4%), 24:23(0.4%) | 48.2% | 0.5% | 2/2/0 | 8/0 | 32:2737(47.5%), 33:2(0.0%) |
| `vehicle_scarcity_capacity_shock` | 25:1369(23.8%), 29:1157(20.1%), 41:863(15.0%), 33:778(13.5%), 1:745(12.9%) | 23.8% | 26.5% | 0/0/0 | 8/2 | 32:5(0.1%), 33:778(13.5%), 41:863(15.0%), 45:18(0.3%) |

Known bad action-pocket aggregate across all H1M scenarios:

| Action | attempts | pct of all H actions | failed | no_current | no_unassigned | route_failure | no_vehicle | already_assigned |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 0.00% | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | 4697 | 10.19% | 0 | 0 | 0 | 5 | 64 | 2906 |
| 33 | 1501 | 3.26% | 0 | 0 | 0 | 1 | 13 | 781 |
| 41 | 1235 | 2.68% | 0 | 0 | 0 | 15 | 59 | 913 |
| 45 | 483 | 1.05% | 0 | 0 | 0 | 2 | 231 | 451 |
| 46 | 0 | 0.00% | 0 | 0 | 0 | 0 | 0 | 0 |

Action-quality conclusion:

- No action `16` or `46` pocket is active.
- Known action `32` is high in route disruption and mixed stress, but it has no failed-noop, no-current, or no-unassigned signal.
- Known action `33`, `41`, and `45` appear under demand/vehicle-scarcity pressure with nonzero no-vehicle/already-assigned context, but not with no-current/no-unassigned or credit-leak hard blockers.
- No new fatal dominant pocket was found. The existing top-action concentration watch remains valid and should be monitored.

## Episode-Level Outlier Audit

| Scenario | worst service ep/service/late | worst lateness ep/service/late | worst no-work ep/count | worst routefail ep/count | suspicious? |
| --- | --- | --- | --- | --- | --- |
| `baseline_normal` | 6/0.974/0.000 | 0/0.993/0.000 | 0/0 | 0/1 | no fatal outlier |
| `demand_spike_volatility` | 10/0.845/0.020 | 1/0.863/0.075 | 0/0 | 0/1 | low service, lateness |
| `high_holding_cost` | 4/0.938/0.000 | 0/0.963/0.000 | 0/1 | 0/1 | no fatal outlier |
| `lead_time_volatility` | 0/1.000/0.000 | 0/1.000/0.000 | 9/1 | 9/1 | no fatal outlier |
| `mixed_stress` | 19/0.925/0.000 | 7/0.931/0.004 | 11/4 | 0/1 | no fatal outlier |
| `premium_sla_pressure` | 0/1.000/0.000 | 0/1.000/0.000 | 9/1 | 9/1 | no fatal outlier |
| `route_disruption_congestion` | 13/0.876/0.000 | 18/0.887/0.080 | 0/1 | 0/1 | low service, lateness |
| `vehicle_scarcity_capacity_shock` | 19/0.921/0.000 | 6/0.947/0.012 | 0/0 | 2/1 | no fatal outlier |

Episode-level conclusion:

- Two scenarios have visible outliers: demand spike and route disruption.
- The route-disruption worst service episode falls below `0.88` and the worst lateness episode reaches `0.080`.
- Demand-spike has a worst-service episode at `0.845` and worst lateness `0.075`, but scenario-level service remains above production and the gate passes.
- No single episode shows a no-work or route-failure explosion.

## Distribution Sanity

| Scenario | route top P -> H | fleet top P -> H | reorder top P -> H |
| --- | --- | --- | --- |
| `baseline_normal` | low_congestion:413(48%), high_resilience:247(29%) -> shortest:5577(97%), low_congestion:182(3%) | secondary_fleet:696(81%), primary_fleet:168(19%) -> secondary_fleet:5239(91%), primary_fleet:521(9%) | none:436(50%), conservative:428(50%) -> none:3363(58%), conservative:2397(42%) |
| `demand_spike_volatility` | shortest:446(52%), low_congestion:304(35%) -> shortest:3253(56%), low_congestion:1724(30%) | secondary_fleet:514(59%), primary_fleet:350(41%) -> primary_fleet:3374(59%), secondary_fleet:2386(41%) | conservative:534(62%), none:329(38%) -> conservative:5666(98%), none:94(2%) |
| `high_holding_cost` | low_congestion:370(43%), high_resilience:273(32%) -> shortest:5259(91%), low_congestion:501(9%) | primary_fleet:447(52%), secondary_fleet:417(48%) -> secondary_fleet:5378(93%), primary_fleet:382(7%) | conservative:500(58%), none:364(42%) -> none:5222(91%), conservative:538(9%) |
| `lead_time_volatility` | shortest:340(39%), high_resilience:268(31%) -> shortest:3684(64%), high_resilience:1168(20%) | primary_fleet:560(65%), secondary_fleet:304(35%) -> secondary_fleet:3771(65%), primary_fleet:1989(35%) | none:436(50%), conservative:425(49%) -> conservative:3796(66%), none:1964(34%) |
| `mixed_stress` | low_congestion:444(51%), high_resilience:361(42%) -> shortest:4072(71%), low_congestion:1682(29%) | secondary_fleet:598(69%), primary_fleet:266(31%) -> secondary_fleet:4783(83%), primary_fleet:977(17%) | conservative:449(52%), none:308(36%) -> none:5752(100%), conservative:8(0%) |
| `premium_sla_pressure` | shortest:402(47%), low_congestion:280(32%) -> shortest:5718(99%), low_congestion:42(1%) | primary_fleet:645(75%), secondary_fleet:219(25%) -> secondary_fleet:3705(64%), primary_fleet:2055(36%) | none:535(62%), conservative:326(38%) -> none:3118(54%), conservative:2642(46%) |
| `route_disruption_congestion` | high_resilience:657(76%), shortest:193(22%) -> shortest:2924(51%), low_congestion:2817(49%) | secondary_fleet:459(53%), primary_fleet:405(47%) -> secondary_fleet:5675(99%), primary_fleet:85(1%) | conservative:521(60%), none:336(39%) -> none:5643(98%), conservative:117(2%) |
| `vehicle_scarcity_capacity_shock` | shortest:375(43%), low_congestion:257(30%) -> shortest:3909(68%), low_congestion:970(17%) | primary_fleet:433(50%), secondary_fleet:431(50%) -> secondary_fleet:4294(75%), primary_fleet:1466(25%) | conservative:558(65%), none:297(34%) -> conservative:5117(89%), none:643(11%) |

Distribution conclusion:

- H1M is more concentrated than prior production in several scenarios.
- Route-disruption shifts away from production's high-resilience-heavy profile toward shortest/low-congestion.
- Mixed stress chooses almost all `none` reorder mode despite high dispatch, which aligns with the residual service watch.
- These shifts are not runtime bugs, but they are production monitoring targets.

## Protected Path Audit

Protected hashes:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- production joint: `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- production PPO: `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996`
- production DQN: `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D`
- production manifest: `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Protected path profiles:

| Path | File count | Total bytes | Status |
| --- | ---: | ---: | --- |
| `models\baselines` | `2692` | `7392576274` | unchanged |
| `db` | `7` | `28330` | unchanged |
| `models\checkpoints` | `787` | `4920830666` | unchanged |
| `models\eval` | `120` | `354070196` | unchanged |

Process scan:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

Result: no protected drift was found.

## Obsidian / Project Memory

`docs/00_PROJECT_DASHBOARD.md` points to:

- final release handoff: `docs/releases/20260611_hierarchical_v1_1m_production_handoff`
- current production checkpoint: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- production promotion audit: `docs/runs/20260611_hierarchical_v1_production_promotion_audit`

No dashboard edit was needed.

## Risk Classification

No evidence of:

- hidden registry/runtime mismatch,
- missing production artifact,
- manifest/handoff mismatch,
- missing replay/RNG state,
- missing hierarchical heads,
- SB3 loader fallback,
- failed hard blocker,
- failed scenario verdict,
- no-current/no-unassigned action-quality explosion,
- failed-noop action explosion,
- active action `16` or `46` pocket.

Warnings that remain:

- route-disruption service below production,
- mixed-stress service below production,
- action concentration in `mixed_stress`, `route_disruption_congestion`, and `lead_time_volatility`,
- route-failure and mixed-success route-failure warnings,
- demand-spike and route-disruption episode outliers.

Classification before independent review:

`HIERARCHICAL_V1_PRODUCTION_ASSURANCE_WARNINGS_ONLY`

## Independent Review

Reviewer verdict:

```text
PRODUCTION_ASSURANCE_AUDIT_WARNINGS_ONLY
```

## Final Classification

`HIERARCHICAL_V1_PRODUCTION_ASSURANCE_WARNINGS_ONLY`
