# Company Data Replay Validation Harness Spec - 2026-06-14

## Classification

`COMPANY_REPLAY_SCHEMA_READY`

## Purpose

The replay validation harness defines the minimum future company-data fields needed to test whether action 24 and action 32 are operationally justified. It is synthetic-only in this sprint and does not ingest private company data.

Implementation:

- `scripts/company_data_replay_validator.py`

Synthetic report:

- `reports/benchmarks/public_data_expansion_20260614/company_data_validation_harness/company_data_harness_report.json`

## Required Table Families

| Table | Required fields |
|---|---|
| `dispatch_attempts` | `dispatch_attempt_id`, `order_id`, `action_id`, `dispatch_status`, `dispatch_failure_reason`, `fleet_type`, `carrier_id`, `vehicle_id`, `route_id` |
| `fleet` | `vehicle_id`, `carrier_id`, `fleet_type`, `accepted`, `fleet_cost` |
| `inventory` | `order_id`, `sku_id`, `stock_on_hand_at_decision`, `stockout_after_decision`, `reorder_type` |
| `costs` | `order_id`, `primary_fleet_cost`, `secondary_fleet_cost`, `stockout_cost`, `holding_cost`, `lateness_penalty_cost` |
| `routes` | `route_id`, `planned_route_type`, `route_failure_flag`, `actual_travel_time`, `planned_travel_time` |

## Questions It Can Answer When Real Data Is Approved

- Did action 24/32 dispatch attempts succeed or fail?
- Were secondary-fleet choices economically justified?
- Did no-reorder choices cause stockouts?
- Were shortest and low-congestion route choices reliable?
- Did action 24/32 failures map to no-current, no-unassigned, failed-noop, no-vehicle, or route-failure modes?

## Non-Goals

- No private data ingestion in this sprint.
- No DB writes.
- No training, eval, gate, registry update, baseline update, or checkpoint mutation.
- No calibration conclusions from synthetic fixtures.

