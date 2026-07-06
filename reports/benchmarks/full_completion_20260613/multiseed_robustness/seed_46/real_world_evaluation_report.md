# Real-World Scenario Evaluation Report

Checkpoint: `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`
Contract: `physical_reality_v5_route_candidate_visibility`
Observation dim: `73`

delivered_delta is a backward-compatible alias for mean_step_delivered_delta; final_step_delivered_delta is reported separately.
Global hard blockers override scenario threshold pass.

| Scenario | Verdict | Scenario Threshold Verdict | Hard-Blocker Verdict | Threshold Failures | Hard-Blocker Failures | Threshold Warnings | Unsupported Overrides | Partial Overrides |
|---|---:|---:|---:|---|---|---|---|---|
| baseline_normal | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 4 | none | none |
| demand_spike_volatility | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 15 | none | clustered_spike_windows |
| high_holding_cost | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 6 | none | none |
| lead_time_volatility | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 1 | none | none |
| mixed_stress | PASS | PASS | PASS | none | none | action_24_25_concentration 0.584 > warning 0.500; mixed_success_route_failure_steps present: 10 | none | none |
| premium_sla_pressure | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 1 | none | none |
| route_disruption_congestion | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 8 | none | none |
| vehicle_scarcity_capacity_shock | PASS | PASS | PASS | none | none | mixed_success_route_failure_steps present: 6 | none | none |

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
| baseline_normal | 0 | 0 | 506 |
| demand_spike_volatility | 0 | 0 | 3326 |
| high_holding_cost | 0 | 0 | 341 |
| lead_time_volatility | 1 | 1 | 379 |
| mixed_stress | 0 | 0 | 915 |
| premium_sla_pressure | 1 | 1 | 2028 |
| route_disruption_congestion | 0 | 0 | 87 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 1375 |

## Route/Premium/Mixed Diagnostic Telemetry

mixed_stress remains a watch unless candidate-quality telemetry proves avoidable overconservatism.

| Scenario | action31 attempts | action31 successes | action31 failed/no-op | action31 success ratio | Top failed/no-op actions | Successful dispatch routes | Failed dispatch routes | Hold routes | Premium hold under useful dispatch | Premium missed reorder opportunity | Shortest near-best not selected | High-resilience selected when shortest near-best |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| baseline_normal | 0 | 0 | 0 | None | [] | {"high_resilience": 2, "low_congestion": 221, "shortest": 1949} | {} | {"shortest": 3588} | 0 | 0 | 0 | 0 |
| demand_spike_volatility | 0 | 0 | 0 | None | [] | {"high_resilience": 687, "low_congestion": 1768, "shortest": 2034} | {} | {"shortest": 1271} | 0 | 0 | 0 | 0 |
| high_holding_cost | 0 | 0 | 0 | None | [{"action_id": 24, "dispatch_attempts": 1673, "dispatch_success_ratio": 0.9988045427375971, "failed_noop": 2, "successes": 1671}, {"action_id": 25, "dispatch_attempts": 34, "dispatch_success_ratio": 0.9705882352941176, "failed_noop": 1, "successes": 33}] | {"low_congestion": 496, "shortest": 2088} | {"shortest": 3} | {"shortest": 3173} | 0 | 0 | 0 | 0 |
| lead_time_volatility | 0 | 0 | 0 | None | [{"action_id": 36, "dispatch_attempts": 732, "dispatch_success_ratio": 0.9986338797814208, "failed_noop": 1, "successes": 731}] | {"high_resilience": 1153, "low_congestion": 896} | {"low_congestion": 1} | {"shortest": 3710} | 0 | 0 | 0 | 0 |
| mixed_stress | 0 | 0 | 0 | None | [{"action_id": 36, "dispatch_attempts": 309, "dispatch_success_ratio": 0.9935275080906149, "failed_noop": 2, "successes": 307}, {"action_id": 28, "dispatch_attempts": 613, "dispatch_success_ratio": 0.99836867862969, "failed_noop": 1, "successes": 612}] | {"high_resilience": 10, "low_congestion": 1695, "shortest": 3973} | {"low_congestion": 2, "shortest": 1} | {"shortest": 79} | 0 | 0 | 0 | 0 |
| premium_sla_pressure | 0 | 0 | 0 | None | [{"action_id": 28, "dispatch_attempts": 1776, "dispatch_success_ratio": 0.9994369369369369, "failed_noop": 1, "successes": 1775}] | {"low_congestion": 40, "shortest": 1988} | {"shortest": 1} | {"shortest": 3731} | 0 | 4 | 0 | 0 |
| route_disruption_congestion | 0 | 0 | 0 | None | [{"action_id": 24, "dispatch_attempts": 24, "dispatch_success_ratio": 0.9583333333333334, "failed_noop": 1, "successes": 23}, {"action_id": 40, "dispatch_attempts": 19, "dispatch_success_ratio": 0.9473684210526315, "failed_noop": 1, "successes": 18}] | {"high_resilience": 18, "low_congestion": 2825, "shortest": 38} | {"high_resilience": 1, "shortest": 1} | {"shortest": 2877} | 0 | 0 | 0 | 0 |
| vehicle_scarcity_capacity_shock | 0 | 0 | 0 | None | [{"action_id": 41, "dispatch_attempts": 869, "dispatch_success_ratio": 0.998849252013809, "failed_noop": 1, "successes": 868}] | {"high_resilience": 884, "low_congestion": 931, "shortest": 2878} | {"high_resilience": 1} | {"shortest": 1066} | 0 | 0 | 0 | 0 |

### Route Candidate Quality

| Scenario | route_candidate_score_mean_by_route | selected_route_candidate_score_mean_by_route | selected_route_rank_counts | selected_route_best_count | selected_route_near_best_count | selected_route_margin_to_best_mean | shortest_near_best_but_not_selected_count | high_resilience_selected_when_shortest_near_best_count | high_resilience_selected_when_not_best_count | mixed_route_overconservative_candidate_count |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| demand_spike_volatility | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| high_holding_cost | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| lead_time_volatility | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| mixed_stress | {"high_resilience": 0.6344316884940151, "low_congestion": 0.572148201629961, "shortest": 0.0} | {"high_resilience": 0.7644505831329367, "low_congestion": 0.4858436497975061, "shortest": 0.0} | {"1": 136, "2": 1660, "3": 3964} | 136 | 1672 | 0.48996620593538737 | 0 | 0 | 0 | 0 |
| premium_sla_pressure | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| route_disruption_congestion | {"high_resilience": 0.1548279337894633, "low_congestion": 0.15854400372094365, "shortest": 0.0} | {"high_resilience": 0.6925267054377868, "low_congestion": 0.30726699612423286, "shortest": 0.0} | {"1": 5704, "2": 18, "3": 38} | 5704 | 2299 | 0.005560102426798611 | 0 | 0 | 18 | 0 |
| vehicle_scarcity_capacity_shock | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"high_resilience": 0.0, "low_congestion": 0.0, "shortest": 0.0} | {"1": 5760} | 5760 | 0 | 0.0 | 0 | 0 | 0 | 0 |

### Premium SLA Responsiveness

| Scenario | premium_pressure_steps | premium_hold_rate | premium_dispatch_rate | premium_missed_useful_dispatch_rate | premium_no_vehicle_steps | premium_already_assigned_steps | premium_no_unassigned_steps | premium_reorder_none_steps | premium_reorder_conservative_steps | premium_reorder_aggressive_steps | premium_reorder_emergency_steps | premium_missed_useful_reorder_rate | premium_service_pressure_steps | premium_late_or_at_risk_backlog_steps | hold_primary_under_premium_pressure_rate | dispatch_primary_under_premium_pressure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| demand_spike_volatility | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| high_holding_cost | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| lead_time_volatility | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| mixed_stress | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| premium_sla_pressure | 5760 | 0.6477430555555556 | 0.35225694444444444 | 0.0 | 2 | 442 | 0 | 3006 | 2754 | 0 | 0 | 1.0 | 3307 | 2694 | 0.0 | 2029 |
| route_disruption_congestion | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
| vehicle_scarcity_capacity_shock | 0 | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | 0 | 0 | None | 0 |
