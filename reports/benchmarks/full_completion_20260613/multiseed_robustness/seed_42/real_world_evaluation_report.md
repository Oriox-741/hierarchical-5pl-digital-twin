# Real-World Scenario Evaluation Report

Checkpoint: `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`
Contract: `physical_reality_v5_route_candidate_visibility`
Observation dim: `73`

delivered_delta is a backward-compatible alias for mean_step_delivered_delta; final_step_delivered_delta is reported separately.
Global hard blockers override scenario threshold pass.

| Scenario | Verdict | Scenario Threshold Verdict | Hard-Blocker Verdict | Threshold Failures | Hard-Blocker Failures | Threshold Warnings | Unsupported Overrides | Partial Overrides |
|---|---:|---:|---:|---|---|---|---|---|
| baseline_normal | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 3 | none | none |
| demand_spike_volatility | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 13 | none | clustered_spike_windows |
| high_holding_cost | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 6 | none | none |
| lead_time_volatility | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 1 | none | none |
| mixed_stress | PASS | PASS | PASS | none | none | action_24_25_concentration 0.580 > warning 0.500; mixed_success_route_failure_steps present: 10 | none | none |
| premium_sla_pressure | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 1 | none | none |
| route_disruption_congestion | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 6 | none | none |
| vehicle_scarcity_capacity_shock | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 8 | none | none |

## Hard-Blocker Telemetry

| Scenario | fake_dispatch_credit | customer_revisited | route_failure_positive_dispatch_credit | nan_inf_detected | dqn_local_negative_positive_train_rows | no_current_work_dqn_delivery_credit | hold_delivery_credit_leak | action8_route_or_delivery_credit_leak | unsafe_24_25_candidate_credit | no_work_positive_dqn_local | emergency_zero_useful_positive_credit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| demand_spike_volatility | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| high_holding_cost | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lead_time_volatility | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mixed_stress | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| premium_sla_pressure | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| route_disruption_congestion | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 0 | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Premium Reward Guard Telemetry

| Scenario | premium_sla_fleet_credit_blocked_no_current_work | premium_primary_adaptation_credit_blocked_no_current_work | premium_fleet_credit_allowed |
|---|---:|---:|---:|
| baseline_normal | 0 | 0 | 467 |
| demand_spike_volatility | 0 | 0 | 3371 |
| high_holding_cost | 0 | 0 | 335 |
| lead_time_volatility | 1 | 1 | 393 |
| mixed_stress | 0 | 0 | 965 |
| premium_sla_pressure | 1 | 1 | 2054 |
| route_disruption_congestion | 0 | 0 | 84 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 1376 |

## Route/Premium/Mixed Diagnostic Telemetry

mixed_stress remains a watch unless candidate-quality telemetry proves avoidable overconservatism.

| Scenario | action31 attempts | action31 successes | action31 failed/no-op | action31 success ratio | Top failed/no-op actions | Successful dispatch routes | Failed dispatch routes | Hold routes | Premium hold under useful dispatch | Premium missed reorder opportunity | Shortest near-best not selected | High-resilience selected when shortest near-best |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| baseline_normal | 0 | 0 | 0 | None | [] | {"high_resilience": 1, "low_congestion": 182, "shortest": 1957} | {} | {"shortest": 3620} | 0 | 0 | 0 | 0 |
| demand_spike_volatility | 0 | 0 | 0 | None | [] | {"high_resilience": 783, "low_congestion": 1724, "shortest": 2063} | {} | {"shortest": 1190} | 0 | 0 | 0 | 0 |
| high_holding_cost | 0 | 0 | 0 | None | [{"action_id": 24, "dispatch_attempts": 1674, "dispatch_success_ratio": 0.998805256869773, "failed_noop": 2, "successes": 1672}] | {"low_congestion": 501, "shortest": 2081} | {"shortest": 2} | {"shortest": 3176} | 0 | 0 | 0 | 0 |
| lead_time_volatility | 0 | 0 | 0 | None | [{"action_id": 36, "dispatch_attempts": 750, "dispatch_success_ratio": 0.9986666666666667, "failed_noop": 1, "successes": 749}] | {"high_resilience": 1168, "low_congestion": 907} | {"low_congestion": 1} | {"shortest": 3684} | 0 | 0 | 0 | 0 |
| mixed_stress | 0 | 0 | 0 | None | [{"action_id": 36, "dispatch_attempts": 321, "dispatch_success_ratio": 0.9906542056074766, "failed_noop": 3, "successes": 318}, {"action_id": 28, "dispatch_attempts": 653, "dispatch_success_ratio": 0.998468606431853, "failed_noop": 1, "successes": 652}] | {"high_resilience": 6, "low_congestion": 1679, "shortest": 3994} | {"low_congestion": 3, "shortest": 1} | {"shortest": 77} | 0 | 0 | 0 | 0 |
| premium_sla_pressure | 0 | 0 | 0 | None | [{"action_id": 28, "dispatch_attempts": 1809, "dispatch_success_ratio": 0.9994472084024323, "failed_noop": 1, "successes": 1808}] | {"low_congestion": 42, "shortest": 2012} | {"shortest": 1} | {"shortest": 3705} | 0 | 4 | 0 | 0 |
| route_disruption_congestion | 0 | 0 | 0 | None | [{"action_id": 24, "dispatch_attempts": 23, "dispatch_success_ratio": 0.9565217391304348, "failed_noop": 1, "successes": 22}, {"action_id": 40, "dispatch_attempts": 19, "dispatch_success_ratio": 0.9473684210526315, "failed_noop": 1, "successes": 18}] | {"high_resilience": 18, "low_congestion": 2817, "shortest": 35} | {"high_resilience": 1, "shortest": 1} | {"shortest": 2888} | 0 | 0 | 0 | 0 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 0 | None | [] | {"high_resilience": 881, "low_congestion": 970, "shortest": 2790} | {} | {"shortest": 1119} | 0 | 0 | 0 | 0 |

### Route Candidate Quality

| Scenario | route_candidate_score_mean_by_route | selected_route_candidate_score_mean_by_route | selected_route_rank_counts | selected_route_best_count | selected_route_near_best_count | selected_route_margin_to_best_mean | shortest_near_best_but_not_selected_count | high_resilience_selected_when_shortest_near_best_count | high_resilience_selected_when_not_best_count | mixed_route_overconservative_candidate_count |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| demand_spike_volatility | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| high_holding_cost | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| lead_time_volatility | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| mixed_stress | {"high_resilience": 0.6426558635821862, "low_congestion": 0.5795649922347249, "shortest": 0.0} | {"high_resilience": 0.7419520504772059, "low_congestion": 0.4898813947591292, "shortest": 0.0} | {"1": 127, "2": 1646, "3": 3987} | 127 | 1653 | 0.4988308256850129 | 0 | 0 | 0 | 0 |
| premium_sla_pressure | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| route_disruption_congestion | {"high_resilience": 0.1341676657875814, "low_congestion": 0.1373878626629599, "shortest": 0.0} | {"high_resilience": 0.6873624755485667, "low_congestion": 0.2658074934488473, "shortest": 0.0} | {"1": 5707, "2": 18, "3": 35} | 5707 | 2156 | 0.005124043898927706 | 0 | 0 | 18 | 0 |
| vehicle_scarcity_capacity_shock | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |

### Premium SLA Responsiveness

| Scenario | premium_pressure_steps | premium_hold_rate | premium_dispatch_rate | premium_missed_useful_dispatch_rate | premium_no_vehicle_steps | premium_already_assigned_steps | premium_no_unassigned_steps | premium_reorder_none_steps | premium_reorder_conservative_steps | premium_reorder_aggressive_steps | premium_reorder_emergency_steps | premium_missed_useful_reorder_rate | premium_service_pressure_steps | premium_late_or_at_risk_backlog_steps | hold_primary_under_premium_pressure_rate | dispatch_primary_under_premium_pressure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| demand_spike_volatility | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| high_holding_cost | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| lead_time_volatility | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| mixed_stress | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| premium_sla_pressure | 5760 | 0.6432291666666666 | 0.3567708333333333 | 0.0 | 2 | 447 | 0 | 3118 | 2642 | 0 | 0 | 1.0 | 3351 | 2729 | 0.0 | 2055 |
| route_disruption_congestion | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| vehicle_scarcity_capacity_shock | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
