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
| mixed_stress | PASS | PASS | PASS | none | none | action_24_25_concentration 0.581 > warning 0.500; mixed_success_route_failure_steps present: 9 | none | none |
| premium_sla_pressure | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 1 | none | none |
| route_disruption_congestion | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 7 | none | none |
| vehicle_scarcity_capacity_shock | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 7 | none | none |

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
| baseline_normal | 0 | 0 | 508 |
| demand_spike_volatility | 0 | 0 | 3346 |
| high_holding_cost | 0 | 0 | 340 |
| lead_time_volatility | 1 | 1 | 376 |
| mixed_stress | 0 | 0 | 942 |
| premium_sla_pressure | 1 | 1 | 2037 |
| route_disruption_congestion | 0 | 0 | 89 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 1396 |

## Route/Premium/Mixed Diagnostic Telemetry

mixed_stress remains a watch unless candidate-quality telemetry proves avoidable overconservatism.

| Scenario | action31 attempts | action31 successes | action31 failed/no-op | action31 success ratio | Top failed/no-op actions | Successful dispatch routes | Failed dispatch routes | Hold routes | Premium hold under useful dispatch | Premium missed reorder opportunity | Shortest near-best not selected | High-resilience selected when shortest near-best |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| baseline_normal | 0 | 0 | 0 | None | [] | {"high_resilience": 2, "low_congestion": 223, "shortest": 1980} | {} | {"shortest": 3555} | 0 | 0 | 0 | 0 |
| demand_spike_volatility | 0 | 0 | 0 | None | [] | {"high_resilience": 705, "low_congestion": 1759, "shortest": 2034} | {} | {"shortest": 1262} | 0 | 0 | 0 | 0 |
| high_holding_cost | 0 | 0 | 0 | None | [{"action_id": 24, "dispatch_attempts": 1682, "dispatch_success_ratio": 0.9988109393579072, "failed_noop": 2, "successes": 1680}, {"action_id": 25, "dispatch_attempts": 36, "dispatch_success_ratio": 0.9722222222222222, "failed_noop": 1, "successes": 35}] | {"low_congestion": 473, "shortest": 2098} | {"shortest": 3} | {"shortest": 3186} | 0 | 0 | 0 | 0 |
| lead_time_volatility | 0 | 0 | 0 | None | [{"action_id": 36, "dispatch_attempts": 736, "dispatch_success_ratio": 0.998641304347826, "failed_noop": 1, "successes": 735}] | {"high_resilience": 1161, "low_congestion": 897} | {"low_congestion": 1} | {"shortest": 3701} | 0 | 0 | 0 | 0 |
| mixed_stress | 0 | 0 | 0 | None | [{"action_id": 36, "dispatch_attempts": 318, "dispatch_success_ratio": 0.9937106918238994, "failed_noop": 2, "successes": 316}, {"action_id": 28, "dispatch_attempts": 630, "dispatch_success_ratio": 0.9984126984126984, "failed_noop": 1, "successes": 629}] | {"high_resilience": 10, "low_congestion": 1697, "shortest": 3974} | {"low_congestion": 2, "shortest": 1} | {"shortest": 76} | 0 | 0 | 0 | 0 |
| premium_sla_pressure | 0 | 0 | 0 | None | [{"action_id": 28, "dispatch_attempts": 1787, "dispatch_success_ratio": 0.9994404029099049, "failed_noop": 1, "successes": 1786}] | {"low_congestion": 41, "shortest": 1996} | {"shortest": 1} | {"shortest": 3722} | 0 | 4 | 0 | 0 |
| route_disruption_congestion | 0 | 0 | 0 | None | [{"action_id": 24, "dispatch_attempts": 24, "dispatch_success_ratio": 0.9583333333333334, "failed_noop": 1, "successes": 23}, {"action_id": 40, "dispatch_attempts": 17, "dispatch_success_ratio": 0.9411764705882353, "failed_noop": 1, "successes": 16}] | {"high_resilience": 16, "low_congestion": 2850, "shortest": 37} | {"high_resilience": 1, "shortest": 1} | {"shortest": 2855} | 0 | 0 | 0 | 0 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 0 | None | [] | {"high_resilience": 872, "low_congestion": 947, "shortest": 2845} | {} | {"shortest": 1096} | 0 | 0 | 0 | 0 |

### Route Candidate Quality

| Scenario | route_candidate_score_mean_by_route | selected_route_candidate_score_mean_by_route | selected_route_rank_counts | selected_route_best_count | selected_route_near_best_count | selected_route_margin_to_best_mean | shortest_near_best_but_not_selected_count | high_resilience_selected_when_shortest_near_best_count | high_resilience_selected_when_not_best_count | mixed_route_overconservative_candidate_count |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| demand_spike_volatility | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| high_holding_cost | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| lead_time_volatility | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| mixed_stress | {"high_resilience": 0.6368517683802367, "low_congestion": 0.5743306972079947, "shortest": 0.0} | {"high_resilience": 0.7644505831329367, "low_congestion": 0.48692221459427615, "shortest": 0.0} | {"1": 133, "2": 1662, "3": 3965} | 133 | 1674 | 0.49189945094499316 | 0 | 0 | 0 | 0 |
| premium_sla_pressure | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| route_disruption_congestion | {"high_resilience": 0.1547171123225447, "low_congestion": 0.15843052239600783, "shortest": 0.0} | {"high_resilience": 0.6897443295638672, "low_congestion": 0.30515045302649657, "shortest": 0.0} | {"1": 5707, "2": 16, "3": 37} | 5707 | 2321 | 0.005408917408490386 | 0 | 0 | 16 | 0 |
| vehicle_scarcity_capacity_shock | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |

### Premium SLA Responsiveness

| Scenario | premium_pressure_steps | premium_hold_rate | premium_dispatch_rate | premium_missed_useful_dispatch_rate | premium_no_vehicle_steps | premium_already_assigned_steps | premium_no_unassigned_steps | premium_reorder_none_steps | premium_reorder_conservative_steps | premium_reorder_aggressive_steps | premium_reorder_emergency_steps | premium_missed_useful_reorder_rate | premium_service_pressure_steps | premium_late_or_at_risk_backlog_steps | hold_primary_under_premium_pressure_rate | dispatch_primary_under_premium_pressure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| demand_spike_volatility | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| high_holding_cost | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| lead_time_volatility | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| mixed_stress | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| premium_sla_pressure | 5760 | 0.6461805555555555 | 0.35381944444444446 | 0.0 | 2 | 450 | 0 | 3044 | 2716 | 0 | 0 | 1.0 | 3321 | 2702 | 0.0 | 2038 |
| route_disruption_congestion | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| vehicle_scarcity_capacity_shock | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
