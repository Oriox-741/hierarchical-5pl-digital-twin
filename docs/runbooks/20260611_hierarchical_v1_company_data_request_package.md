# Hierarchical V1 Company Data Request Package

Date: 2026-06-11

Final classification: `STEP1_COMPANY_DATA_REQUEST_PACKAGE_READY`

## Purpose And Non-Goals

This package defines the company data needed to validate the current
hierarchical v1 production model against real operations.

Purpose:

- validate action `24` and action `32` under real route, fleet, inventory,
  dispatch, and cost conditions,
- close the secondary-fleet and reorder-economics gaps that public route data
  cannot close,
- map simulator KPIs to business KPIs before any future model research.

Non-goals:

- no private data ingestion in this task,
- no DB mutation,
- no model training,
- no offline eval,
- no long-run gate,
- no registry or production mutation,
- no checkpoint mutation,
- no existing eval-output edit,
- no claim that public data validates secondary-fleet or reorder economics.

## Current Model And Action Context

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- architecture: `hierarchical_v1`
- initialization: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- obs/action: `73 / 48`
- equal-budget residual-watch gate: `PASS`
- accepted watches:
  - route action `32` concentration,
  - mixed action `24` concentration,
  - top-action concentration warnings,
  - mixed-success route-failure warnings.

Watched action decode:

- action `24` = `dispatch + shortest + secondary_fleet + none`
- action `32` = `dispatch + low_congestion + secondary_fleet + none`

## Order Fields

| Field | Required | Purpose |
| --- | --- | --- |
| `order_id` | yes | primary join key across order, dispatch, inventory, and delivery facts |
| `customer_id` | preferred | customer segmentation and repeat-pattern analysis |
| `customer_location_id` | preferred | route clustering and location joins |
| `latitude` / `longitude` | yes or obfuscated equivalent | route-distance and zone calibration |
| `zone_id` | preferred | operating-regime and route-cluster grouping |
| `created_at` | yes | demand arrival and queue age |
| `order_priority` or SLA tier | yes | premium/SLA segmentation |
| `promised_window_start` | yes | service and lateness definition |
| `promised_window_end` | yes | service and lateness definition |
| `order_size` / package count / volume | yes | vehicle capacity and stop workload |
| `sku_id` or product group | preferred | inventory/reorder validation |

## Route Fields

| Field | Required | Purpose |
| --- | --- | --- |
| `planned_route_id` | yes | planned route join |
| `actual_route_id` | preferred | actual route comparison |
| `planned_route_type` | preferred | map to shortest, low-congestion, high-resilience equivalents |
| `planned_distance` | yes | route efficiency and route-choice proxy |
| `actual_distance` | preferred | route deviation |
| `planned_travel_time` | yes | travel-time target |
| `actual_travel_time` | preferred | congestion/reliability calibration |
| `planned_sequence_rank` | preferred | route-order comparison |
| `actual_sequence_rank` | preferred | driver/execution route comparison |
| `route_failure_flag` | yes | route-quality watch |
| `reroute_flag` | preferred | disruption and low-congestion proxy |
| `congestion_delay_flag` | preferred | route-disruption label |
| `route_provider` / routing engine | preferred | compare route policy source |

## Fleet Fields

| Field | Required | Purpose |
| --- | --- | --- |
| `vehicle_id` | yes | vehicle capacity and availability |
| `carrier_id` | preferred | carrier reliability/cost |
| `fleet_type` | yes | primary versus secondary fleet validation |
| `vehicle_capacity` | yes | no-vehicle and load feasibility |
| `available_at_decision` | yes | fleet feasibility at policy time |
| `assigned_at` | yes | dispatch lifecycle |
| `accepted_at` | preferred | carrier acceptance latency |
| `cancelled_at` | preferred | carrier cancellation behavior |
| `fleet_cost` | yes | secondary-fleet economic validation |
| `fleet_reliability_score` | preferred | service/cancellation calibration |

## Inventory Fields

| Field | Required | Purpose |
| --- | --- | --- |
| `sku_id` | yes | inventory join key |
| `site_id` or fulfillment node | yes | stock location |
| `stock_on_hand_at_decision` | yes | reorder necessity |
| `allocated_quantity` | preferred | available-to-promise |
| `backlog_quantity` | yes | no-reorder risk |
| `stockout_flag` | yes | reorder economics |
| `reorder_event_id` | preferred | replenishment action history |
| `reorder_type` | preferred | conservative/aggressive/emergency mapping |
| `supplier_id` | preferred | lead-time source |
| `supplier_lead_time` | yes | lead-time calibration |

## Cost Fields

| Field | Required | Purpose |
| --- | --- | --- |
| `primary_fleet_cost` | yes | baseline fleet economics |
| `secondary_fleet_cost` | yes | validate secondary_fleet concentration |
| `route_cost` | preferred | route-choice economics |
| `lateness_penalty_cost` | yes | SLA tradeoff |
| `stockout_cost` | yes | no-reorder tradeoff |
| `holding_cost` | yes | reorder/holding balance |
| `emergency_reorder_cost` | preferred | emergency replenishment penalty |
| `cancellation_cost` | preferred | carrier/fleet reliability penalty |

## Dispatch Failure Fields

| Field | Required | Purpose |
| --- | --- | --- |
| `dispatch_attempt_id` | yes | attempt-level denominator |
| `decision_timestamp` | yes | policy-time state |
| `dispatch_attempt_timestamp` | yes | action execution time |
| `dispatch_status` | yes | success/failure |
| `dispatch_failure_reason` | yes | no-current/no-unassigned/no-vehicle/route-failure mapping |
| `no_vehicle_flag` | yes | fleet feasibility |
| `already_assigned_flag` | yes | lock/idempotency conflict |
| `route_failure_flag` | yes | route-choice risk |
| `carrier_rejected_flag` | preferred | secondary-fleet reliability |
| `duplicate_assignment_flag` | preferred | already-assigned semantics |

## Join Keys

Minimum viable join keys:

- `order_id` across order, dispatch, delivery, and inventory facts.
- `dispatch_attempt_id` for action-quality denominators.
- `vehicle_id` and/or `carrier_id` for fleet availability and cost.
- `planned_route_id` and `actual_route_id` for route comparison.
- `sku_id` and `site_id` for inventory and reorder validation.
- timestamps with shared timezone semantics.

Stop if these keys cannot be produced or mapped consistently.

## Privacy And Access Assumptions

Preferred access shape:

- read-only extract,
- minimum required columns,
- hashed or surrogate customer identifiers,
- obfuscated coordinates if exact coordinates are sensitive,
- clear data retention window,
- no production DB writes,
- no direct customer PII unless explicitly approved,
- shared field dictionary and owner for each table.

The first extract can be aggregated or de-identified if it preserves join keys
and decision-time ordering.

## Validation Questions For Action 24

Action `24` = `dispatch + shortest + secondary_fleet + none`.

Questions:

1. Under mixed-stress-like regimes, is a shortest or fastest-like route
   operationally common for urgent orders?
2. Was secondary fleet actually available at the policy decision time?
3. Is secondary fleet economically acceptable relative to lateness penalties?
4. Does secondary fleet have comparable acceptance, cancellation, and on-time
   reliability?
5. Was inventory sufficient at decision time, making reorder `none` reasonable?
6. Does no-reorder correlate with stockouts, backlog, or emergency reorders?
7. Are no-vehicle or already-assigned counters explained by real platform
   semantics or by policy overuse?

## Validation Questions For Action 32

Action `32` = `dispatch + low_congestion + secondary_fleet + none`.

Questions:

1. Under route-disruption-like regimes, do planned or actual routes favor
   lower-congestion or more reliable route choices?
2. Does low-congestion routing reduce lateness or reroutes compared with
   shortest-like alternatives?
3. Are route failures materially higher for action `32` or comparable
   low-congestion choices?
4. Was secondary fleet needed because primary fleet was unavailable or slower?
5. Does secondary fleet cost remain acceptable under disruption?
6. Was inventory sufficient, making reorder `none` reasonable?
7. Does route action `32` concentration coincide with service improvement,
   service neutrality, or operational degradation?

## What Amazon Public Sample Supports

The approved Amazon Last Mile small sample supports:

- route schema and travel-time matrix parsing,
- actual driver sequence comparison,
- greedy travel-time proxy comparison,
- directed travel-time asymmetry as a reliability proxy,
- partial route-side plausibility for action `24` and action `32`.

Observed sample facts:

- `13` routes,
- `3,129` packages,
- actual/greedy travel-time ratio mean `0.9673`,
- `9 / 13` actual routes no worse than greedy,
- nontrivial directed travel-time asymmetry,
- no truly tight two-hour windows in the small sample.

## What Company Data Must Validate

Company data is required for:

- secondary-fleet availability,
- secondary-fleet acceptance/cancellation/reliability,
- primary versus secondary fleet cost,
- inventory sufficiency at decision time,
- no-reorder correctness,
- stockout/backlog/emergency reorder impact,
- platform-specific no-current/no-unassigned/no-vehicle/already-assigned
  semantics,
- true SLA penalty and cost tradeoffs.

## Minimum Viable First Extract

A first extract should cover a recent stable production-like window and include:

- order table with `order_id`, location/zone, created time, promised window,
  package volume, SLA tier,
- dispatch attempts with action-equivalent labels if available, decision time,
  dispatch status, failure reason, vehicle/carrier/fleet type,
- delivery outcomes with pickup/delivery timestamps and on-time flag,
- route table with planned/actual travel time and route failure/reroute flags,
- fleet table with availability, capacity, fleet type, and cost,
- inventory table with stock on hand, backlog, stockout, reorder events, and
  lead time.

## Stop Conditions

Stop the calibration effort if:

- join keys are missing,
- timestamps cannot be aligned,
- SLA/lateness semantics are undefined,
- fleet type cannot distinguish primary from secondary,
- fleet cost or availability is unavailable,
- inventory state at decision time is unavailable,
- data access/privacy approval is missing,
- the extract requires DB mutation or production-system writes.

## Source Documents

- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runs/20260611_public_route_proxy_validation_readiness.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`

## Final Classification

`STEP1_COMPANY_DATA_REQUEST_PACKAGE_READY`
