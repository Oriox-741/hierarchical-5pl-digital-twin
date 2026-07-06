# Real-World Scenario Evaluation Report

Checkpoint: `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`
Contract: `physical_reality_v5_route_candidate_visibility`
Observation dim: `73`

delivered_delta is a backward-compatible alias for mean_step_delivered_delta; final_step_delivered_delta is reported separately.
Global hard blockers override scenario threshold pass.

| Scenario | Verdict | Scenario Threshold Verdict | Hard-Blocker Verdict | Threshold Failures | Hard-Blocker Failures | Threshold Warnings | Unsupported Overrides | Partial Overrides |
|---|---:|---:|---:|---|---|---|---|---|
| baseline_normal | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 3 | none | none |
| demand_spike_volatility | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 14 | none | clustered_spike_windows |
| high_holding_cost | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 7 | none | none |
| lead_time_volatility | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 1 | none | none |
| mixed_stress | PASS | PASS | PASS | none | none | action_24_25_concentration 0.578 > warning 0.500; mixed_success_route_failure_steps present: 10 | none | none |
| premium_sla_pressure | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 1 | none | none |
| route_disruption_congestion | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 7 | none | none |
| vehicle_scarcity_capacity_shock | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 9 | none | none |

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
| baseline_normal | 0 | 0 | 465 |
| demand_spike_volatility | 0 | 0 | 3387 |
| high_holding_cost | 0 | 0 | 343 |
| lead_time_volatility | 1 | 1 | 387 |
| mixed_stress | 0 | 0 | 956 |
| premium_sla_pressure | 1 | 1 | 2040 |
| route_disruption_congestion | 0 | 0 | 85 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 1388 |

## Route/Premium/Mixed Diagnostic Telemetry

mixed_stress remains a watch unless candidate-quality telemetry proves avoidable overconservatism.

| Scenario | action31 attempts | action31 successes | action31 failed/no-op | action31 success ratio | Top failed/no-op actions | Successful dispatch routes | Failed dispatch routes | Hold routes | Premium hold under useful dispatch | Premium missed reorder opportunity | Shortest near-best not selected | High-resilience selected when shortest near-best |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| baseline_normal | 0 | 0 | 0 | None | [] | {"high_resilience": 1, "low_congestion": 186, "shortest": 1971} | {} | {"shortest": 3602} | 0 | 0 | 0 | 0 |
| demand_spike_volatility | 0 | 0 | 0 | None | [] | {"high_resilience": 780, "low_congestion": 1717, "shortest": 2057} | {} | {"shortest": 1206} | 0 | 0 | 0 | 0 |
| high_holding_cost | 0 | 0 | 0 | None | [{"action_id": 24, "dispatch_attempts": 1688, "dispatch_success_ratio": 0.9988151658767772, "failed_noop": 2, "successes": 1686}] | {"low_congestion": 505, "shortest": 2105} | {"shortest": 2} | {"shortest": 3148} | 0 | 0 | 0 | 0 |
| lead_time_volatility | 0 | 0 | 0 | None | [{"action_id": 36, "dispatch_attempts": 748, "dispatch_success_ratio": 0.9986631016042781, "failed_noop": 1, "successes": 747}] | {"high_resilience": 1156, "low_congestion": 905} | {"low_congestion": 1} | {"shortest": 3698} | 0 | 0 | 0 | 0 |
| mixed_stress | 0 | 0 | 0 | None | [{"action_id": 36, "dispatch_attempts": 319, "dispatch_success_ratio": 0.9905956112852664, "failed_noop": 3, "successes": 316}, {"action_id": 28, "dispatch_attempts": 647, "dispatch_success_ratio": 0.9984544049459042, "failed_noop": 1, "successes": 646}] | {"high_resilience": 6, "low_congestion": 1694, "shortest": 3974} | {"low_congestion": 3, "shortest": 1} | {"shortest": 82} | 0 | 0 | 0 | 0 |
| premium_sla_pressure | 0 | 0 | 0 | None | [{"action_id": 28, "dispatch_attempts": 1797, "dispatch_success_ratio": 0.9994435169727324, "failed_noop": 1, "successes": 1796}] | {"low_congestion": 43, "shortest": 1997} | {"shortest": 1} | {"shortest": 3719} | 0 | 4 | 0 | 0 |
| route_disruption_congestion | 0 | 0 | 0 | None | [{"action_id": 24, "dispatch_attempts": 23, "dispatch_success_ratio": 0.9565217391304348, "failed_noop": 1, "successes": 22}, {"action_id": 40, "dispatch_attempts": 17, "dispatch_success_ratio": 0.9411764705882353, "failed_noop": 1, "successes": 16}] | {"high_resilience": 16, "low_congestion": 2855, "shortest": 37} | {"high_resilience": 1, "shortest": 1} | {"shortest": 2850} | 0 | 0 | 0 | 0 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 0 | None | [] | {"high_resilience": 883, "low_congestion": 957, "shortest": 2823} | {} | {"shortest": 1097} | 0 | 0 | 0 | 0 |

### Route Candidate Quality

| Scenario | route_candidate_score_mean_by_route | selected_route_candidate_score_mean_by_route | selected_route_rank_counts | selected_route_best_count | selected_route_near_best_count | selected_route_margin_to_best_mean | shortest_near_best_but_not_selected_count | high_resilience_selected_when_shortest_near_best_count | high_resilience_selected_when_not_best_count | mixed_route_overconservative_candidate_count |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| demand_spike_volatility | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| high_holding_cost | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| lead_time_volatility | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| mixed_stress | {"high_resilience": 0.6398922155427781, "low_congestion": 0.5770726573082668, "shortest": 0.0} | {"high_resilience": 0.7419520504772059, "low_congestion": 0.49042822364156996, "shortest": 0.0} | {"1": 132, "2": 1661, "3": 3967} | 132 | 1668 | 0.4946303391846867 | 0 | 0 | 0 | 0 |
| premium_sla_pressure | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| route_disruption_congestion | {"high_resilience": 0.14812254591845883, "low_congestion": 0.15167767790007133, "shortest": 0.0} | {"high_resilience": 0.6885865438793577, "low_congestion": 0.2909991459150545, "shortest": 0.0} | {"1": 5707, "2": 16, "3": 37} | 5707 | 2254 | 0.005408835394267618 | 0 | 0 | 16 | 0 |
| vehicle_scarcity_capacity_shock | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |

### Premium SLA Responsiveness

| Scenario | premium_pressure_steps | premium_hold_rate | premium_dispatch_rate | premium_missed_useful_dispatch_rate | premium_no_vehicle_steps | premium_already_assigned_steps | premium_no_unassigned_steps | premium_reorder_none_steps | premium_reorder_conservative_steps | premium_reorder_aggressive_steps | premium_reorder_emergency_steps | premium_missed_useful_reorder_rate | premium_service_pressure_steps | premium_late_or_at_risk_backlog_steps | hold_primary_under_premium_pressure_rate | dispatch_primary_under_premium_pressure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| demand_spike_volatility | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| high_holding_cost | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| lead_time_volatility | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| mixed_stress | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| premium_sla_pressure | 5760 | 0.6456597222222222 | 0.35434027777777777 | 0.0 | 2 | 440 | 0 | 3119 | 2641 | 0 | 0 | 1.0 | 3336 | 2712 | 0.0 | 2041 |
| route_disruption_congestion | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| vehicle_scarcity_capacity_shock | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
