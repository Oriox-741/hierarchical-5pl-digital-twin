# Chapter 3 Architecture Truth Table

Generated UTC: `2026-06-23T00:16:34.509949+00:00`

Official thesis title: `Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration`

All values below are source-grounded. Derived values are marked as derived; missing values are not guessed.

## Source-Code Module Map

| Component | Chapter 3 role | Source | Status |
| --- | --- | --- | --- |
| Observation builder | Builds normalized 73-feature state | src/act/observation_builder.py:103 | current source truth |
| Action projector | Maps five PPO controls from [-1, 1] to physical controls | src/act/action_projector.py:14 | current source truth |
| Discrete mapper | Encodes/decodes 48 DQN actions | src/act/discrete_action_mapper.py:40 | current source truth |
| Environment | Gymnasium 5PL digital twin, reward and physical action execution | src/act/env_5pl.py:140 | current source truth |
| Joint policies | PPO actor/critic and hierarchical_v1 DQN heads | src/think/joint_policies.py:222 | current source truth |
| Torch runtime | Loads joint checkpoint and validates output sizes/bounds | src/orchestration/torch_joint_runtime.py:67 | current source truth |
| Policy service | Resolves active registry pair and returns torch_joint predictions | src/orchestration/policy_service.py:205 | current source truth |

## 73-Feature Observation Structure

Total observation dimension: `73` from ordered feature tuple groups. Source: src/act/observation_builder.py:103 to src/act/observation_builder.py:111.

| Feature group | Count | Index range | Ordered feature names | Source | Value status |
| --- | --- | --- | --- | --- | --- |
| LEGACY_OBSERVATION_FEATURES | 32 | 0..31 | time_ratio, pending_order_ratio, delivered_order_ratio, late_delivery_ratio, failed_order_ratio, service_level, transport_cost_ratio, on_hand_inventory_share, reserved_inventory_share, in_transit_inventory_share, backlog_inventory_share, safety_stock_inventory_share, demand_mean_pressure, demand_variance_pressure, lead_time_mean_pressure, lead_time_variance_pressure, vehicle_utilization, active_vehicle_ratio, capacity_pressure, disruption_score, network_safety_potential, hub_count_ratio, customer_count_ratio, route_arc_count_ratio, db_asset_count_ratio, db_avg_speed_ratio, db_avg_battery_ratio, db_avg_sensor_quality_ratio, db_safety_potential, db_disruption_score, db_congestion_score, bias | src/act/observation_builder.py:15 | explicit tuple ordering; total is derived from tuple lengths |
| REALITY_OBSERVATION_FEATURES | 12 | 32..43 | inventory_coverage_ratio, stockout_risk, safety_stock_target_gap, demand_volatility_pressure, lead_time_volatility_pressure, backlog_age_pressure, urgent_order_ratio, speed_cost_exposure, mean_speed_ratio, premium_fleet_exposure, route_disruption_pressure, capacity_slack | src/act/observation_builder.py:50 | explicit tuple ordering; total is derived from tuple lengths |
| DISPATCH_FEASIBILITY_OBSERVATION_FEATURES | 5 | 44..48 | normalized_available_vehicle_count, normalized_dispatchable_order_count, feasible_dispatch_opportunity, feasible_dispatch_ratio, vehicle_availability_pressure | src/act/observation_builder.py:65 | explicit tuple ordering; total is derived from tuple lengths |
| REAL_WORLD_STRESS_OBSERVATION_FEATURES | 8 | 49..56 | normalized_holding_cost_pressure, normalized_stockout_penalty_pressure, capacity_shock_pressure, scenario_route_disruption_pressure, route_cost_pressure, premium_sla_pressure, supplier_delay_pressure, scenario_lead_time_volatility_pressure | src/act/observation_builder.py:73 | explicit tuple ordering; total is derived from tuple lengths |
| ROUTE_CANDIDATE_OBSERVATION_FEATURES | 16 | 57..72 | shortest_candidate_score, low_congestion_candidate_score, high_resilience_candidate_score, shortest_candidate_score_gap, low_congestion_candidate_score_gap, high_resilience_candidate_score_gap, route_pressure_reliability_share, route_pressure_congestion_share, route_pressure_balance, route_alt_pressure_imbalance, shortest_candidate_near_best, shortest_secondary_safe_context, shortest_secondary_brittle_risk, secondary_fleet_feasible_dispatch_ratio, secondary_fleet_vehicle_availability_pressure, useful_dispatch_opportunity | src/act/observation_builder.py:84 | explicit tuple ordering; total is derived from tuple lengths |

## Ordered Observation Feature Index

| Index | Feature | Group | Source | Value status |
| --- | --- | --- | --- | --- |
| 0 | time_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 1 | pending_order_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 2 | delivered_order_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 3 | late_delivery_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 4 | failed_order_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 5 | service_level | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 6 | transport_cost_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 7 | on_hand_inventory_share | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 8 | reserved_inventory_share | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 9 | in_transit_inventory_share | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 10 | backlog_inventory_share | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 11 | safety_stock_inventory_share | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 12 | demand_mean_pressure | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 13 | demand_variance_pressure | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 14 | lead_time_mean_pressure | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 15 | lead_time_variance_pressure | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 16 | vehicle_utilization | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 17 | active_vehicle_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 18 | capacity_pressure | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 19 | disruption_score | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 20 | network_safety_potential | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 21 | hub_count_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 22 | customer_count_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 23 | route_arc_count_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 24 | db_asset_count_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 25 | db_avg_speed_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 26 | db_avg_battery_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 27 | db_avg_sensor_quality_ratio | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 28 | db_safety_potential | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 29 | db_disruption_score | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 30 | db_congestion_score | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 31 | bias | LEGACY_OBSERVATION_FEATURES | src/act/observation_builder.py:15 | explicit ordered tuple member |
| 32 | inventory_coverage_ratio | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 33 | stockout_risk | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 34 | safety_stock_target_gap | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 35 | demand_volatility_pressure | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 36 | lead_time_volatility_pressure | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 37 | backlog_age_pressure | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 38 | urgent_order_ratio | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 39 | speed_cost_exposure | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 40 | mean_speed_ratio | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 41 | premium_fleet_exposure | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 42 | route_disruption_pressure | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 43 | capacity_slack | REALITY_OBSERVATION_FEATURES | src/act/observation_builder.py:50 | explicit ordered tuple member |
| 44 | normalized_available_vehicle_count | DISPATCH_FEASIBILITY_OBSERVATION_FEATURES | src/act/observation_builder.py:65 | explicit ordered tuple member |
| 45 | normalized_dispatchable_order_count | DISPATCH_FEASIBILITY_OBSERVATION_FEATURES | src/act/observation_builder.py:65 | explicit ordered tuple member |
| 46 | feasible_dispatch_opportunity | DISPATCH_FEASIBILITY_OBSERVATION_FEATURES | src/act/observation_builder.py:65 | explicit ordered tuple member |
| 47 | feasible_dispatch_ratio | DISPATCH_FEASIBILITY_OBSERVATION_FEATURES | src/act/observation_builder.py:65 | explicit ordered tuple member |
| 48 | vehicle_availability_pressure | DISPATCH_FEASIBILITY_OBSERVATION_FEATURES | src/act/observation_builder.py:65 | explicit ordered tuple member |
| 49 | normalized_holding_cost_pressure | REAL_WORLD_STRESS_OBSERVATION_FEATURES | src/act/observation_builder.py:73 | explicit ordered tuple member |
| 50 | normalized_stockout_penalty_pressure | REAL_WORLD_STRESS_OBSERVATION_FEATURES | src/act/observation_builder.py:73 | explicit ordered tuple member |
| 51 | capacity_shock_pressure | REAL_WORLD_STRESS_OBSERVATION_FEATURES | src/act/observation_builder.py:73 | explicit ordered tuple member |
| 52 | scenario_route_disruption_pressure | REAL_WORLD_STRESS_OBSERVATION_FEATURES | src/act/observation_builder.py:73 | explicit ordered tuple member |
| 53 | route_cost_pressure | REAL_WORLD_STRESS_OBSERVATION_FEATURES | src/act/observation_builder.py:73 | explicit ordered tuple member |
| 54 | premium_sla_pressure | REAL_WORLD_STRESS_OBSERVATION_FEATURES | src/act/observation_builder.py:73 | explicit ordered tuple member |
| 55 | supplier_delay_pressure | REAL_WORLD_STRESS_OBSERVATION_FEATURES | src/act/observation_builder.py:73 | explicit ordered tuple member |
| 56 | scenario_lead_time_volatility_pressure | REAL_WORLD_STRESS_OBSERVATION_FEATURES | src/act/observation_builder.py:73 | explicit ordered tuple member |
| 57 | shortest_candidate_score | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 58 | low_congestion_candidate_score | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 59 | high_resilience_candidate_score | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 60 | shortest_candidate_score_gap | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 61 | low_congestion_candidate_score_gap | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 62 | high_resilience_candidate_score_gap | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 63 | route_pressure_reliability_share | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 64 | route_pressure_congestion_share | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 65 | route_pressure_balance | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 66 | route_alt_pressure_imbalance | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 67 | shortest_candidate_near_best | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 68 | shortest_secondary_safe_context | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 69 | shortest_secondary_brittle_risk | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 70 | secondary_fleet_feasible_dispatch_ratio | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 71 | secondary_fleet_vehicle_availability_pressure | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |
| 72 | useful_dispatch_opportunity | ROUTE_CANDIDATE_OBSERVATION_FEATURES | src/act/observation_builder.py:84 | explicit ordered tuple member |

## Five PPO Controls

| Index | Physical control | Projection rule | Source |
| --- | --- | --- | --- |
| 0 | reorder_fraction | unit interval from clipped PPO element 0 | src/act/action_projector.py:76 |
| 1 | dispatch_intensity | unit interval from clipped PPO element 1 | src/act/action_projector.py:77 |
| 2 | speed_multiplier | 1.0 + action[2] * max_speed_delta_fraction | src/act/action_projector.py:78 |
| 3 | safety_stock_multiplier | 1.0 + unit_interval(action[3]) * (max_safety_stock_multiplier - 1.0) | src/act/action_projector.py:79 |
| 4 | capacity_buffer_fraction | unit_interval(action[4]) * max_capacity_buffer_fraction | src/act/action_projector.py:80 |

## Hierarchical DQN Heads

| Head | Cardinality | Values | Network source | Enum source |
| --- | --- | --- | --- | --- |
| dispatch_head | 2 | HOLD, DISPATCH | src/think/joint_policies.py:243 | src/act/discrete_action_mapper.py:10 |
| route_head | 3 | SHORTEST, LOW_CONGESTION, HIGH_RESILIENCE | src/think/joint_policies.py:244 | src/act/discrete_action_mapper.py:15 |
| mode_head | 2 | SECONDARY_FLEET, PRIMARY_FLEET | src/think/joint_policies.py:245 | src/act/discrete_action_mapper.py:21 |
| reorder_head | 4 | NONE, CONSERVATIVE, AGGRESSIVE, EMERGENCY | src/think/joint_policies.py:246 | src/act/discrete_action_mapper.py:26 |

## 48-Action Encoding

| Item | Value | Source | Limit/derivation |
| --- | --- | --- | --- |
| Dispatch values | `["HOLD", "DISPATCH"]` | src/act/discrete_action_mapper.py:10 | 2 values |
| Route values | `["SHORTEST", "LOW_CONGESTION", "HIGH_RESILIENCE"]` | src/act/discrete_action_mapper.py:15 | 3 values |
| Mode/fleet values | `["SECONDARY_FLEET", "PRIMARY_FLEET"]` | src/act/discrete_action_mapper.py:21 | 2 values |
| Reorder values | `["NONE", "CONSERVATIVE", "AGGRESSIVE", "EMERGENCY"]` | src/act/discrete_action_mapper.py:26 | 4 values |
| External action count | `48` | src/act/discrete_action_mapper.py:40 | 2*3*2*4 = 48 |
| Decode formula | reorder = value % 4; mode = (value//4) % 2; route = ... % 3; dispatch = remaining | src/act/discrete_action_mapper.py:72 | explicit decode code |
| Encode formula | (((dispatch * len(RouteDecision)) + route) * len(ModeDecision) + mode) * len(ReorderDecision) + reorder | src/act/discrete_action_mapper.py:87 | explicit encode code |

## Hierarchical Q-Value Composition

| Formula | Source | Status |
| --- | --- | --- |
| dispatch_component + reorder_component + dispatch_mask(route + mode) | src/think/joint_policies.py:261 | explicit source formula |

## Runtime Prediction Path

| Runtime fact | Value | Source | Status |
| --- | --- | --- | --- |
| Production logical model | `joint_torch_v5_prod_hierarchical_v1_1m_20260611` | models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json $.logical_model_id | explicit production manifest |
| DQN architecture | `hierarchical_v1` | models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json $.dqn_architecture | explicit production manifest |
| Init method | `flat_teacher_distillation_v1` | models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json $.hierarchical_init_method | explicit production manifest |
| Contract | `physical_reality_v5_route_candidate_visibility` | models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json $.contract | explicit production manifest |
| Runtime checkpoint version/artifact checks | checkpoint_version and artifact_kind are validated before load | src/orchestration/torch_joint_runtime.py:92 | source validation |
| Runtime observation/action checks | obs shape 73, continuous len 5 finite [-1,1], discrete 0..47 | src/orchestration/torch_joint_runtime.py:36 | source validation |
| PolicyService active path | returns algorithm torch_joint when active joint checkpoint resolves | src/orchestration/policy_service.py:205 | source validation |

## Eight Scenario Configurations

| Scenario id | Name | Seeds | Scenario JSON episode_count | Environment overrides | Pass/fail thresholds | Source | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_normal | Baseline Normal Operations | `[42, 43, 44]` | `3` | `{"rolling_demand_max_orders_per_step": 2, "rolling_demand_probability": 0.35, "service_level_target": 0.95, "urgent_order_probability": 0.25}` | `{"fail_on_customer_revisited": true, "fail_on_fake_dispatch_credit": true, "fail_on_nan_inf": true, "fail_on_no_vehicle_available": false, "fail_on_route_failure_positive_dispatch_credit": true, "fail_on_stuck_assigned_in_transit_orders": true, "max_action_24_25_concentration_fail": 0.4, "max_action_24_25_concentration_warn": 0.4, "max_true_lateness_pressure": 0.05, "min_delivered_delta": 0.45, "min_dispatch_success_per_attempt": 0.7, "min_service_level": 0.93, "require_operational_degradation_for_action_concentration_fail": true}` | configs/eval_scenarios/baseline_normal.json $.scenario_id/name/seeds/episode_count/environment_overrides/pass_fail_thresholds | Scenario JSON episode_count is 3; final equal-budget summaries contain 160 rows across 8 scenarios (20 per scenario) in packaged eval summary evidence. |
| demand_spike_volatility | Demand Spike / Demand Volatility Scenario | `[62, 63, 64]` | `3` | `{"clustered_spike_windows": [{"end_step": 72, "multiplier": 2.0, "start_step": 24}], "order_units_multiplier": 1.5, "rolling_demand_probability_multiplier": 1.85, "rolling_demand_volume_multiplier": 2.0, "urgent_order_probability_multiplier": 1.4}` | `{"fail_on_customer_revisited": true, "fail_on_fake_dispatch_credit": true, "fail_on_nan_inf": true, "fail_on_route_failure_positive_dispatch_credit": true, "max_action_24_25_concentration_fail": 0.5, "max_action_24_25_concentration_warn": 0.5, "max_true_lateness_pressure": 0.25, "min_dispatch_success_per_attempt": 0.5, "min_service_level": 0.6, "require_operational_degradation_for_action_concentration_fail": true}` | configs/eval_scenarios/demand_spike_volatility.json $.scenario_id/name/seeds/episode_count/environment_overrides/pass_fail_thresholds | Scenario JSON episode_count is 3; final equal-budget summaries contain 160 rows across 8 scenarios (20 per scenario) in packaged eval summary evidence. |
| high_holding_cost | High Holding-Cost Scenario | `[52, 53, 54]` | `3` | `{"excess_inventory_penalty_multiplier": 2.0, "holding_cost_multiplier": 2.25, "planned_replenishment_cost_multiplier": 2.0, "service_level_target": 0.95}` | `{"fail_on_customer_revisited": true, "fail_on_fake_dispatch_credit": true, "fail_on_nan_inf": true, "fail_on_route_failure_positive_dispatch_credit": true, "max_action_24_25_concentration_fail": 0.4, "max_action_24_25_concentration_warn": 0.4, "max_true_lateness_pressure": 0.08, "min_service_level": 0.9, "require_operational_degradation_for_action_concentration_fail": true}` | configs/eval_scenarios/high_holding_cost.json $.scenario_id/name/seeds/episode_count/environment_overrides/pass_fail_thresholds | Scenario JSON episode_count is 3; final equal-budget summaries contain 160 rows across 8 scenarios (20 per scenario) in packaged eval summary evidence. |
| lead_time_volatility | Lead-Time Volatility / Supplier Delay Scenario | `[72, 73, 74]` | `3` | `{"lead_time_mean_multiplier": 1.8, "lead_time_variance_multiplier": 3.0, "service_level_target": 0.95, "supplier_delay_probability": 0.45, "urgent_order_probability": 0.3}` | `{"fail_on_customer_revisited": true, "fail_on_fake_dispatch_credit": true, "fail_on_nan_inf": true, "fail_on_route_failure_positive_dispatch_credit": true, "max_action_24_25_concentration_fail": 0.4, "max_action_24_25_concentration_warn": 0.4, "max_true_lateness_pressure": 0.1, "min_service_level": 0.9, "require_operational_degradation_for_action_concentration_fail": true}` | configs/eval_scenarios/lead_time_volatility.json $.scenario_id/name/seeds/episode_count/environment_overrides/pass_fail_thresholds | Scenario JSON episode_count is 3; final equal-budget summaries contain 160 rows across 8 scenarios (20 per scenario) in packaged eval summary evidence. |
| mixed_stress | Mixed Stress Scenario | `[112, 113, 114]` | `3` | `{"capacity_shock_severity": 0.4, "congestion_multiplier": 2.5, "fleet_capacity_multiplier": 0.6, "holding_cost_multiplier": 2.0, "inventory_shortfall_penalty_multiplier": 1.5, "order_units_multiplier": 1.45, "rolling_demand_probability_multiplier": 1.7, "rolling_demand_volume_multiplier": 2.0, "route_disruption_probability": 0.55, "stockout_penalty_multiplier": 1.6, "transport_cost_penalty_scale": 180.0, "traversal_cost_multiplier": 1.4, "urgent_order_probability_multiplier": 1.8, "vehicle_availability_multiplier": 0.5}` | `{"fail_on_customer_revisited": true, "fail_on_fake_dispatch_credit": true, "fail_on_nan_inf": true, "fail_on_route_failure_positive_dispatch_credit": true, "max_action_24_25_concentration_fail": 0.5, "max_action_24_25_concentration_warn": 0.5, "max_true_lateness_pressure": 0.2, "min_dispatch_success_per_attempt": 0.5, "min_service_level": 0.7, "require_operational_degradation_for_action_concentration_fail": true}` | configs/eval_scenarios/mixed_stress.json $.scenario_id/name/seeds/episode_count/environment_overrides/pass_fail_thresholds | Scenario JSON episode_count is 3; final equal-budget summaries contain 160 rows across 8 scenarios (20 per scenario) in packaged eval summary evidence. |
| premium_sla_pressure | Premium SLA Pressure Scenario | `[102, 103, 104]` | `3` | `{"premium_lateness_penalty_multiplier": 1.8, "premium_sla_ratio": 0.55, "primary_fleet_penalty": 0.08, "service_target": 0.98, "urgent_due_window_multiplier": 0.65}` | `{"fail_on_customer_revisited": true, "fail_on_fake_dispatch_credit": true, "fail_on_nan_inf": true, "fail_on_route_failure_positive_dispatch_credit": true, "max_action_24_25_concentration_fail": 0.4, "max_action_24_25_concentration_warn": 0.4, "max_true_lateness_pressure": 0.08, "min_dispatch_success_per_attempt": 0.5, "min_service_level": 0.92, "require_operational_degradation_for_action_concentration_fail": true}` | configs/eval_scenarios/premium_sla_pressure.json $.scenario_id/name/seeds/episode_count/environment_overrides/pass_fail_thresholds | Scenario JSON episode_count is 3; final equal-budget summaries contain 160 rows across 8 scenarios (20 per scenario) in packaged eval summary evidence. |
| route_disruption_congestion | Route Disruption / Congestion Scenario | `[92, 93, 94]` | `3` | `{"congestion_multiplier": 3.0, "disruption_risk_multiplier": 1.8, "high_resilience_route_penalty": 0.04, "low_congestion_route_penalty": 0.01, "route_disruption_probability": 0.6, "shortest_route_risk_multiplier": 2.0, "transport_cost_penalty_scale": 175.0, "traversal_cost_multiplier": 1.5}` | `{"fail_on_customer_revisited": true, "fail_on_fake_dispatch_credit": true, "fail_on_nan_inf": true, "fail_on_route_failure_positive_dispatch_credit": true, "max_action_24_25_concentration_fail": 0.4, "max_action_24_25_concentration_warn": 0.4, "max_true_lateness_pressure": 0.12, "min_dispatch_success_per_attempt": 0.5, "min_service_level": 0.88, "require_operational_degradation_for_action_concentration_fail": true}` | configs/eval_scenarios/route_disruption_congestion.json $.scenario_id/name/seeds/episode_count/environment_overrides/pass_fail_thresholds | Scenario JSON episode_count is 3; final equal-budget summaries contain 160 rows across 8 scenarios (20 per scenario) in packaged eval summary evidence. |
| vehicle_scarcity_capacity_shock | Vehicle Scarcity / Capacity Shock Scenario | `[82, 83, 84]` | `3` | `{"capacity_shock_severity": 0.45, "fleet_capacity_multiplier": 0.55, "rolling_demand_max_orders_per_step": 3, "rolling_demand_probability": 0.45, "service_level_target": 0.93, "vehicle_availability_multiplier": 0.5}` | `{"fail_on_customer_revisited": true, "fail_on_fake_dispatch_credit": true, "fail_on_nan_inf": true, "fail_on_route_failure_positive_dispatch_credit": true, "max_action_24_25_concentration_fail": 0.5, "max_action_24_25_concentration_warn": 0.5, "max_true_lateness_pressure": 0.15, "min_capacity_buffer_fraction": 0.03, "min_dispatch_success_per_attempt": 0.5, "min_service_level": 0.85, "require_operational_degradation_for_action_concentration_fail": true}` | configs/eval_scenarios/vehicle_scarcity_capacity_shock.json $.scenario_id/name/seeds/episode_count/environment_overrides/pass_fail_thresholds | Scenario JSON episode_count is 3; final equal-budget summaries contain 160 rows across 8 scenarios (20 per scenario) in packaged eval summary evidence. |
