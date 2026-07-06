# Company Data And Benchmark Execution Plan

Date: 2026-06-13

Scope: company-data and benchmark execution planning only. This file does not ingest private data, download datasets, run benchmarks, train models, run offline eval, run long-run gates, mutate registry, mutate production, mutate baselines, mutate DB, mutate checkpoints, edit existing eval outputs, or create training configs.

## Executive Summary

The project's strongest current evidence is simulator-based. The next evidence layer should come from:

1. company data request readiness;
2. public and synthetic benchmark design;
3. company historical replay after approved data exists;
4. ablations and multi-seed tests only with separate approval.

The main open validation gaps are action 24 and action 32:

- action 24 = `dispatch + shortest + secondary_fleet + none`
- action 32 = `dispatch + low_congestion + secondary_fleet + none`

Public data can help validate the route component. Company data is required for secondary_fleet, reorder none, dispatch failure, and cost economics.

## Company Data Table Families

Required table families:

| Table family | Minimum purpose | Primary joins |
| --- | --- | --- |
| `orders` | order creation, SLA, demand, package size | `order_id`, `customer_location_id`, `sku_id` |
| `dispatch_attempts` | policy/action equivalent and failure taxonomy | `dispatch_attempt_id`, `order_id`, `vehicle_id`, `route_id` |
| `deliveries_outcomes` | pickup, delivery, on-time, lateness | `order_id`, `dispatch_attempt_id` |
| `routes` | planned/actual route, time/distance/congestion | `route_id`, `order_id` |
| `fleet_carriers` | primary/secondary availability, cost, reliability | `vehicle_id`, `carrier_id` |
| `inventory_reorder` | stock state, reorder events, stockouts, backlog | `sku_id`, `site_id`, `order_id` |
| `costs` | fleet, route, lateness, stockout, holding, reorder costs | `order_id`, `vehicle_id`, `sku_id`, `site_id` |

## Minimum Viable Extract

The first data extract should be small, read-only, and de-identified where possible. It should cover one stable historical operating window.

| Field | Required | Why needed | Action component validated | Join/privacy requirement |
| --- | --- | --- | --- | --- |
| `order_id` | yes | universal join | dispatch feasibility | stable hash ok |
| `created_at` | yes | demand timing and queue age | dispatch/hold | timezone required |
| `promised_window_start` | yes | due-window start | route and service | timestamp semantics |
| `promised_window_end` | yes | due-window end | route and service | timestamp semantics |
| `delivery_time` | yes | actual service/lateness | all | timestamp semantics |
| `delivered_on_time` | yes | service label | all | derived ok if definition supplied |
| `dispatch_attempt_id` | yes | attempt denominator | dispatch quality | stable id |
| `decision_timestamp` | yes | policy-time state | all | must precede dispatch/pickup |
| `dispatch_status` | yes | dispatch success/failure | dispatch | controlled labels |
| `dispatch_failure_reason` | yes | no-current/no-unassigned/no_vehicle/route_failure mapping | action quality | controlled labels preferred |
| `vehicle_id` | yes | vehicle join and capacity | fleet | surrogate ok |
| `carrier_id` | preferred | carrier reliability/cost | fleet | surrogate ok |
| `fleet_type` | yes | primary vs secondary | secondary_fleet | primary/secondary label |
| `planned_route_id` | yes | route plan join | route | stable route id |
| `planned_travel_time` | yes | expected route time | shortest/low_congestion | seconds/minutes |
| `actual_travel_time` | preferred | realized route performance | route | seconds/minutes |
| `stock_on_hand_at_decision` | yes | inventory adequacy | reorder none | SKU/site join |
| `stockout_flag` | yes | inventory failure | reorder none | boolean or event |
| `primary_fleet_cost` | yes | primary cost baseline | secondary_fleet | aggregate/banded ok |
| `secondary_fleet_cost` | yes | overflow cost | secondary_fleet | aggregate/banded ok |

Minimum viable first answer:

- Can rows be joined by `order_id` and `dispatch_attempt_id`?
- Can promised windows and delivery time define service/lateness?
- Can dispatch failure reasons map to simulator counters?
- Can primary vs secondary fleet be distinguished?
- Can inventory state at decision time be joined?
- Can fleet and SLA costs be compared enough to judge action 24/32 economics?

## How To Request The Data From A Company

Recommended request sequence:

1. Start with a non-technical email explaining that the first ask is read-only, historical, and de-identified.
2. Attach a one-page table-family summary rather than a database dump request.
3. Ask for a header-only/schema sample first if the data owner is uncertain.
4. Ask for a small historical window after schema alignment, such as one stable week or one representative operating cycle.
5. Run the synthetic/schema validator against matching synthetic or header-only fixtures before any private data is handled.
6. Stop if privacy, ownership, join keys, timestamp semantics, or data retention rules are unclear.

Initial package contents:

- project purpose in plain language;
- minimum viable extract table;
- ideal extract table;
- anonymization instructions;
- join-key and timestamp requirements;
- list of fields not needed at first, such as customer names, phone numbers, addresses, payment data, or free-text notes;
- data-owner questions;
- next-meeting agenda.

Data-owner questions:

- Which system owns orders, dispatch attempts, routes, fleet/carrier status, inventory, and costs?
- Can stable surrogate keys be created across those systems?
- Are promised delivery windows and actual delivery timestamps available with timezone semantics?
- Are dispatch failure reasons controlled labels or free text?
- Can primary and secondary fleet/carrier usage be distinguished?
- Can stock state and reorder events be reconstructed at decision time?
- Which cost fields can be shared as raw, aggregated, or banded values?

## Ideal Extract

Ideal additions:

| Field group | Fields | Why it matters |
| --- | --- | --- |
| Location | customer/location/zone, depot/site, obfuscated lat/lng | route and demand clustering |
| Package | volume, weight, package count, product group | vehicle capacity and workload |
| Lifecycle | carrier accepted/cancelled timestamps, pickup time | dispatch and fleet reliability |
| Route | planned/actual distance, actual route id, reroute flag, congestion/delay flag | route proxy validation |
| Failure taxonomy | route failure reason, no_vehicle, already_assigned, duplicate assignment, carrier rejected | action-quality mapping |
| Inventory | reorder event id, reorder type, supplier id, supplier lead time, backlog | reorder/no-reorder economics |
| Costs | route cost, holding cost, stockout cost, SLA penalty, emergency reorder cost, cancellation cost | business objective calibration |

## Field-By-Field Rationale

| Field | Why needed | Model question | Action 24/32 validation | Privacy/anonymization |
| --- | --- | --- | --- | --- |
| `order_id` | joins all tables | Was there current work? | dispatch | hash stable across tables |
| `dispatch_attempt_id` | attempt-level denominator | How many attempts failed? | dispatch quality | surrogate ok |
| `vehicle_id` | capacity/availability | Was fleet available? | secondary_fleet | surrogate ok |
| `carrier_id` | reliability and cost | Which carrier class worked? | secondary_fleet | surrogate ok |
| `route_id` | route plan/outcome join | Which route was selected? | shortest/low_congestion | non-sensitive route id ok |
| `sku_id` | inventory join | Was no reorder safe? | reorder none | product group ok if sensitive |
| `site_id` | inventory location | Was stock available locally? | reorder none | site/region ok |
| event timestamps | temporal ordering | Did state exist before decision? | all | timezone-normalized |
| `fleet_type` | primary/secondary label | Is secondary use realistic? | secondary_fleet | controlled label |
| cost fields | business tradeoffs | Is the policy economical? | secondary_fleet/reorder | aggregate/banded acceptable |

## Data-Quality Rules

Run these before any calibration interpretation:

- required columns present;
- required tables present;
- timestamps parse;
- promised window start <= promised window end;
- decision <= dispatch <= pickup <= delivery where present;
- non-negative costs;
- `fleet_type` in `primary|secondary`;
- `reorder_type` in `none|conservative|aggressive|emergency`;
- no duplicate primary keys;
- no blank required join keys;
- shared timezone semantics documented.

Stop if any required join key or SLA timestamp definition is missing.

## Benchmark Table

| Benchmark | Purpose | Safe now? | Needs approval? | Required inputs | Expected outputs | Stop conditions | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Internal equal-budget comparator | Preserve current old-prod vs hierarchical evidence | read-only yes | rerun requires eval/gate approval | existing summaries | comparison table | no fresh output dir approval | P0 |
| OR-Tools routing baseline | Classical route optimization comparison | plan only | dependency/implementation/run approval | synthetic or approved public instances | route time/distance/service report | dependency or mapping unclear | P1 |
| Rule-based dispatch baselines | Transparent heuristic comparator | plan only | implementation/eval approval | existing scenarios | service/lateness/action-quality report | would mutate eval outputs | P1 |
| Amazon full route proxy | Public route-side validation | plan only | download/analysis approval | approved Amazon files | route proxy metrics | size/license/access unclear | P1 |
| Solomon VRPTW | Classic time-window route benchmark | plan only | download/run approval | Solomon instances | route feasibility/time-window report | mapping too artificial | P2 |
| Homberger VRPTW | Larger time-window stress | plan only | download/run approval | Homberger instances | scale feasibility report | runtime/size too high | P2 |
| CVRPLIB/CVRP | Capacity route sanity | plan only | download/run approval | CVRPLIB instances | capacity/route report | no meaningful mapping | P3 |
| Multi-seed robustness | Statistical stability of scenario gates | plan only | offline eval/gate approval | current scenarios, seed list | mean/sd/quantiles | no fresh dirs or approval | P1 |
| Runtime latency | Productization runtime readiness | plan only | implementation/report approval | synthetic observations, active runtime | latency/memory report | service/private data/protected write required | P1 |
| Ablation | Causal design contribution | no | training/eval approval | new planned branches | flat/hier/distillation comparisons | no architecture/training approval | P2/P3 |
| Historical company replay | Strongest pre-live real-world test | no | company data approval | approved company extract | policy-vs-history report | missing privacy/join keys | P0 after data |

## Detailed Benchmark Specifications

### OR-Tools Baseline

- Purpose: compare route-side decisions against a known solver.
- Proves: route/time/distance/service tradeoff under compatible instances.
- Cannot prove: reorder, inventory, or secondary-fleet economics unless extended.
- Required data: synthetic simulator instance export or approved public routes.
- Required code: instance exporter, OR-Tools adapter, comparator.
- Safe now: plan only.
- Metrics: route time, distance, late orders, infeasible orders, vehicle count.
- Stop conditions: new dependency not approved, instance mapping unfaithful, output path not fresh.

### Rule-Based Baselines

Candidate rules:

- FIFO + shortest + primary;
- earliest due date + shortest;
- premium-first;
- low-congestion under disruption;
- stock-threshold reorder.

Purpose:

- give advisor-readable comparisons against human-understandable policies.

Safe now:

- design only.

Needs approval:

- implementation and any simulator/eval run.

### Amazon Full Route Proxy

Purpose:

- scale up the current small-sample route plausibility work.

Metrics:

- actual vs greedy/shortest ratio;
- actual vs reliability/asymmetry proxy;
- time-window stress buckets;
- route concentration under stress;
- invalid sequence scores.

Limit:

- route side only. No secondary_fleet, reorder, or cost validation.

### Solomon / Homberger VRPTW

Purpose:

- evaluate time-window route feasibility in standard benchmarks.

Limit:

- synthetic route benchmarks are not 5PL inventory/fleet/reorder systems.

### CVRPLIB/CVRP

Purpose:

- test capacity and route-cost logic.

Limit:

- lacks time windows and dispatch/reorder semantics in many instances.

### Multi-Seed Robustness

Purpose:

- quantify variance in the current eight scenario suite.

Approval:

- requires fresh offline eval outputs and likely gate checks.

Metrics:

- service mean/sd/min/max;
- lateness tail;
- dispatch success;
- action 24/32 rate distribution;
- hard blockers.

### Runtime Latency

Purpose:

- quantify whether the existing production runtime path is suitable for interactive decision support.

What it proves:

- local CPU cold-load and warm-prediction latency for the current `torch_joint` policy;
- `PolicyService.predict_joint` response behavior on synthetic observations;
- action decode overhead for the 48-action external contract.

What it cannot prove:

- live network latency;
- TMS/WMS/ERP integration latency;
- database performance;
- throughput under real concurrent users.

Required data:

- synthetic observations only.

Required code:

- a small benchmark runner and test fixture if later approved.

Safe now:

- plan only.

Needs approval:

- implementation and any persistent report output.

Output metrics:

- model cold-load time;
- warm prediction p50/p95/p99;
- continuous action shape/finite checks;
- discrete action validity;
- memory footprint if feasible;
- error count.

Stop conditions:

- benchmark would start a long-lived service;
- benchmark would read private company data;
- benchmark would mutate registry, production, baselines, DB, checkpoints, or existing eval outputs;
- output path is not fresh and explicitly approved.

### Ablation

Purpose:

- show why hierarchical v1 and distillation mattered.

Requires:

- new training/eval plans, separate approvals, and fresh dirs.

Not safe-now:

- no training or config creation in this task.

### Historical Company Replay

Purpose:

- compare policy choices against historical company outcomes before live deployment.

Requires:

- approved company data, privacy review, schema validation, replay design.

Stop:

- missing join keys, timestamp ambiguity, no data owner approval.

## Benchmark Priority Roadmap

1. P0 now: company-data request package and data dictionary handoff.
2. P1 next: benchmark design doc for OR-Tools/rule-based/public route benchmarks.
3. P1 after approval: bounded Amazon larger sample, if size/license approved.
4. P1/P2 after approval: OR-Tools or rule-based implementation.
5. P1 after implementation/report approval: runtime latency benchmark.
6. P1 after eval approval: multi-seed robustness.
7. P0 after company data: historical replay.
8. P2/P3 only after evidence: ablations or training extensions.

## Stop Conditions Across All Benchmarks

Stop immediately if:

- command would train;
- command would run offline eval/gate without approval;
- command would download data without size/approval;
- output dir is not fresh;
- protected hashes drift;
- company data appears without approval;
- benchmark cannot preserve the v5/73/48 contract where required;
- benchmark interpretation would overclaim fleet/reorder economics from route-only data.

## Final Classification

`COMPANY_DATA_AND_BENCHMARK_EXECUTION_PLAN_READY`
