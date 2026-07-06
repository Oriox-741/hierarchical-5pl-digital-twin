# Company Replay Bridge to Model Policy Spec - 2026-06-14

## Classification

`COMPANY_REPLAY_BRIDGE_SPEC_READY`

## Purpose

This bridge defines how future approved company historical logs should be transformed into CODEX replay rows and, only if the required logging exists, into OPE-ready trajectories. No private company data was read or ingested in this sprint.

## CODEX Policy Surface

- Logical model: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Observation dim: `73`
- Continuous control dim: `5`
- Discrete action count: `48`
- Action 24: `dispatch + shortest + secondary_fleet + none`
- Action 32: `dispatch + low_congestion + secondary_fleet + none`

## Minimum Company Tables for High-Confidence Replay

| Table | Required fields |
|---|---|
| orders | `order_id`, timestamps, promised window, delivered outcome, order lines, priority/SLA |
| dispatch_attempts | `dispatch_attempt_id`, `order_id`, `decision_time`, `action_id`, `dispatch_status`, failure reason, `route_id`, `vehicle_id`, `carrier_id` |
| routes | planned/actual travel time, route type, origin/destination, route failure, congestion/disruption labels |
| fleet | `vehicle_id`, `carrier_id`, `fleet_type`, availability, acceptance, capacity, cost |
| inventory | SKU/site stock at decision, reserved/in-transit, stockout after decision, reorder type |
| costs | primary/secondary fleet cost, stockout cost, holding cost, lateness penalty, transport cost |

## Extra Requirements for OPE

Proxy replay becomes OPE only if logs include:

- behavior action probability or full behavior-policy distribution
- candidate policy probability/distribution on the same states
- ordered trajectories: `s_t`, `a_t`, `r_t`, `s_next`, `done`
- action support for candidate choices, especially action 24/32 and close alternatives
- pre-registered reward/cost definition
- decision-time alternatives if propensities are not logged

## Confidence Rules

- High: most CODEX state fields can be reconstructed; behavior action/outcome are logged.
- Medium: dispatch/route/fleet fields exist but inventory/cost/alternatives are partial.
- Low: only order timing/location proxies exist.

## No-Overclaim Rules

Public replay and company replay without propensity remain descriptive. They can report policy action tendencies and agreement/proxy correlations, but cannot claim counterfactual superiority or policy value.

Machine-readable spec:

`reports/benchmarks/historical_replay_20260614/company_replay_bridge/company_replay_bridge_spec.json`
