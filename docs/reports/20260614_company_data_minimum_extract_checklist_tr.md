# Company Data Minimum Extract Checklist - 2026-06-14

## Classification

`COMPANY_DATA_MINIMUM_EXTRACT_CHECKLIST_READY`

| Table | Minimum fields | Why |
|---|---|---|
| orders | order_id, created/promised/pickup/delivery timestamps, SLA/priority, SKU/order lines | service, lateness, demand pressure |
| dispatch_attempts | dispatch_attempt_id, order_id, decision_time, action_id, status, failure_reason | action validation and dispatch failure causality |
| routes | route_id, planned route type, planned/actual travel time, failure flag, congestion/disruption | action 24/32 route-side validation |
| fleet | vehicle_id, carrier_id, fleet_type, availability, accepted, capacity, cost | secondary fleet economics |
| inventory | order_id, sku_id/site_id, stock on hand, reserved, in transit, stockout after decision, reorder type | reorder/no-reorder validation |
| costs | primary/secondary fleet cost, stockout, holding, lateness, transport costs | reward/cost model |

PII can be removed. Join keys and timestamp ordering must remain stable.
