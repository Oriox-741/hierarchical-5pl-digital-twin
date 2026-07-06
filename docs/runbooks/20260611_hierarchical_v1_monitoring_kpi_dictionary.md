# Hierarchical V1 Monitoring KPI Dictionary

Date: 2026-06-11

Final classification: `STEP1_MONITORING_KPI_DICTIONARY_READY`

## Scope

This dictionary defines the monitoring metrics for the current production
hierarchical v1 1M model. It is a read-only operating reference.

This document does not authorize training, offline eval, long-run gates,
registry mutation, production mutation, baseline mutation, DB mutation,
checkpoint mutation, eval-output edits, dataset downloads, or training configs.

## Current Production Context

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- runtime family: `torch_joint`
- DQN architecture: `hierarchical_v1`
- initialization method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: `73`
- continuous action dimension: `5`
- external discrete action count: `48`
- production checkpoint:
  `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- status: active registry plus copy-only production promotion complete
- residual-watch status: accepted for monitoring
- equal-budget residual-watch gate: `PASS`

## Action Decode For 24 And 32

The external discrete action space remains `0..47`.

Watched actions:

- action `24` = `dispatch + shortest + secondary_fleet + none`
- action `32` = `dispatch + low_congestion + secondary_fleet + none`

Interpretation boundaries:

- The `shortest` and `low_congestion` route components are partially supported
  by Amazon Last Mile route-proxy evidence.
- The `secondary_fleet` and `none` reorder components are not validated by
  public route-only data.
- Company fleet, cost, dispatch, and inventory data are required to validate
  secondary-fleet and reorder economics.

## Equal-Budget Baselines For Route Action 32 And Mixed Action 24

Equal-budget residual-watch evidence used the same scenario set, `20` episodes
per scenario, seed `42`, CPU deterministic inference, and old production as the
comparator.

| Watch | Old production | Hierarchical v1 1M | Interpretation |
| --- | ---: | ---: | --- |
| `route_disruption_congestion` service | `0.904` | `0.901` | slight `-0.0037` delta, inside gate tolerance |
| `route_disruption_congestion` action `32` attempts | `39` | `2737` | real concentration shift |
| `route_disruption_congestion` action `32` per-step rate | `0.006771` | `0.475174` | accepted watch |
| action `32` no-current | `1` | `0` | no hard-blocker growth |
| action `32` no-unassigned | `1` | `0` | no hard-blocker growth |
| action `32` failed-noop | `1` | `0` | no hard-blocker growth |
| action `32` route-failure | `0` | `5` | watch, not blocker |
| action `32` no-vehicle | `0` | `0` | no growth |
| action `32` already-assigned | `30` | `1444` | watch with denominator |
| `mixed_stress` service | `0.940` | `0.942` | equal-budget service watch cleared |
| `mixed_stress` action `24` attempts | `0` | `3342` | real concentration shift |
| `mixed_stress` action `24` per-step rate | `0.000000` | `0.580208` | accepted watch |
| action `24` no-current | `0` | `0` | no hard-blocker growth |
| action `24` no-unassigned | `0` | `0` | no hard-blocker growth |
| action `24` failed-noop | `0` | `0` | no hard-blocker growth |
| action `24` route-failure | `0` | `10` | watch, not blocker |
| action `24` no-vehicle | `0` | `121` | monitor with fleet availability data |
| action `24` already-assigned | `0` | `2010` | watch with denominator |

## Metric Dictionary

Each metric should be reported by monitoring window, operating regime or
scenario analogue, action id when applicable, and denominator.

| Metric | Definition | Numerator | Denominator | Source | Aggregation grain | Warning threshold | Escalation threshold | Caveats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| service by scenario | Share of orders or demand units served within the relevant SLA or scenario target. | served on time or within service target | eligible orders/demand units | production telemetry, scenario summaries | window x regime x customer/SLA segment | below scenario or business threshold | below threshold in two consecutive windows or with action/failure drift | exact SLA semantics must match business definition |
| lateness | Mean and tail lateness after due window. | late minutes or late-order count | delivered or eligible orders | production delivery telemetry | window x regime x SLA segment | mean/tail grows materially above baseline | service falls while lateness rises in two windows | compare same promise-window mix |
| dispatch success | Share of dispatch attempts that result in executable assignment. | successful dispatch attempts | all dispatch attempts | dispatch platform logs | window x regime x action x fleet type | drops below current operating baseline | drops with service/lateness degradation | distinguish acceptance failure from route/inventory failure |
| no-current by action | Dispatch attempt when no current work item exists. | no-current failures for action | action attempts or decision steps | action-quality telemetry | window x action | any count on action `24` or `32` | repeated count or paired service drop | requires exact no-current taxonomy |
| no-unassigned by action | Dispatch attempt when no eligible unassigned order exists. | no-unassigned failures for action | action attempts or decision steps | action-quality telemetry | window x action | any count on action `24` or `32` | repeated count or paired service drop | may overlap platform queue semantics |
| failed-noop by action | Invalid no-op or no-op failure for a chosen action. | failed-noop rows for action | action attempts or decision steps | action-quality telemetry | window x action | any count on action `24` or `32` | repeated count or runtime invalid-action issue | should remain zero for watched dispatch actions |
| route failure by action | Route infeasible, failed, rerouted, or route-plan failure attributed to the action. | route-failure rows for action | action attempts | route planning and dispatch logs | window x action x route type | doubles from equal-budget watch baseline or grows with service/lateness drift | repeated material growth in two windows | separate route infeasibility from customer absence |
| no-vehicle by action | Fleet unavailable or capacity infeasible for the action. | no-vehicle rows for action | action attempts | fleet availability and dispatch logs | window x action x fleet type | action `24` materially exceeds equal-budget mixed baseline of `121` rows or action `32` develops nonzero no-vehicle rows | paired service/lateness or cost issue | requires fleet availability at decision time |
| already-assigned by action | Duplicate assignment, lock conflict, or attempt on committed work. | already-assigned rows for action | action attempts | dispatch/assignment logs | window x action | material growth above equal-budget baseline | repeated growth with service or throughput impact | real systems may suppress duplicates before policy sees them |
| top action concentration | Share of decisions or attempts captured by the most common action in a window/regime. | count of top action | total decisions or attempts | policy telemetry | window x regime | top action rises by at least 10 percentage points above comparable baseline | repeated concentration with service/failure drift | concentration alone is not a blocker |
| action `24` rate | Rate of `dispatch + shortest + secondary_fleet + none`. | action 24 decisions or attempts | total decisions/attempts in comparable regime | policy telemetry | window x mixed-stress-like regime | rises materially above `0.580208` per-step equal-budget baseline with degradation | repeated material rise plus service/lateness/fleet/cost issue | validates route only with public data; fleet/reorder need company data |
| action `32` rate | Rate of `dispatch + low_congestion + secondary_fleet + none`. | action 32 decisions or attempts | total decisions/attempts in comparable regime | policy telemetry | window x route-disruption-like regime | rises materially above `0.475174` per-step equal-budget baseline with degradation | repeated material rise plus service/lateness/route issue | low-congestion label needs route telemetry mapping |
| secondary_fleet rate | Share of dispatches assigned to secondary or overflow fleet. | secondary-fleet dispatches | all dispatches | fleet and carrier logs | window x regime x carrier | conflicts with availability, cost, acceptance, or cancellation data | cost/service tradeoff turns negative or capacity infeasible | cannot be validated from Amazon route-only data |
| reorder none rate | Share of inventory decisions selecting no reorder. | no-reorder decisions | all reorder decisions or relevant order decisions | inventory/replenishment logs | window x SKU/site/regime | conflicts with stockout, backlog, or emergency-order evidence | repeated stockout/backlog impact | requires inventory and demand ground truth |

## Alert Levels

| Level | Meaning | Examples |
| --- | --- | --- |
| `info` | Expected monitoring movement with no operational impact. | action concentration changes without service/failure drift |
| `watch` | Residual watch moved materially but no confirmed business risk. | action `32` rate above baseline with flat service |
| `warning` | Watch plus operational drift. | action `24` rise with no-vehicle or lateness growth |
| `critical` | Hard-blocker, runtime, or protected-state issue. | no-current on action `24/32`, invalid action, non-`torch_joint` runtime |

## Company Data Requirement

Company data is required before closing the main economic validation gaps:

- `secondary_fleet` availability, acceptance, cancellation, and cost.
- `primary_fleet` versus `secondary_fleet` service reliability.
- reorder `none` versus stockout, backlog, emergency order, and holding-cost
  outcomes.
- dispatch failure taxonomy and lock/idempotency semantics.

Amazon Last Mile evidence only supports route-side plausibility for action
`24` and action `32`.

## Source Documents

- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`

## Final Classification

`STEP1_MONITORING_KPI_DICTIONARY_READY`
