# Real-World Calibration and Logistics Validity Plan

Date: 2026-06-11

## Scope

This is a research and validation plan for the current production model:
`joint_torch_v5_prod_hierarchical_v1_1m_20260611`.

Hard constraints for this work:

- Do not train.
- Do not run offline eval.
- Do not update registry.
- Do not mutate `active_models.json` or `models.jsonl`.
- Do not mutate production, baselines, DB, checkpoints, or existing eval outputs.
- Do not start 3M/5M/10M/100M.

## Executive Conclusion

Actions 24 and 32 are logically plausible logistics choices, but they are not
proven real-world optimal by the simulation alone.

Current evidence supports this interpretation:

- Equal-budget residual-watch eval passed for old production and hierarchical
  production using the same 20 episodes per scenario and seed 42.
- Both models passed all 8 scenarios.
- Equal-budget long-run gate passed.
- Hard blockers stayed zero.
- The no-current/no-unassigned/failed-noop explosion was absent.
- No hidden bug was suspected in the residual-watch audit.

The remaining action concentration is therefore best classified as an
operational calibration question:

- Public data can help validate route-choice behavior, time-window pressure,
  service times, realized travel times, route reliability, and shortest-route
  versus realized-driver-route proxies.
- Public data cannot fully validate company-specific secondary-fleet economics,
  dispatch feasibility rules, reorder/stockout/holding-cost tradeoffs, or the
  true business cost of choosing `secondary_fleet + none`.
- Full validation requires company data for fleet, order, inventory, dispatch,
  failure, and cost fields.

## Local Evidence

Production manifest:

- Logical model: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Architecture: `hierarchical_v1`
- Initialization: `flat_teacher_distillation_v1`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`
- External discrete action count: `48`
- Training step: `1000000`
- Eval verdict: `PASS`
- Hard blocker status: `zero`
- Long-run gate verdict: `PASS`
- Residual watches accepted: `true`

Equal-budget residual-watch evidence:

- Old production comparator: 20 episodes per scenario, seed 42.
- Hierarchical production: 20 episodes per scenario, seed 42.
- Both evals: 8/8 scenario PASS.
- Equal-budget gate: PASS.
- `mixed_stress`: hierarchical service slightly exceeded old production in the
  equal-budget comparison.
- `route_disruption_congestion`: hierarchical service was slightly below old
  production but inside gate tolerance.
- Action concentration remained visible:
  - `route_disruption_congestion` action 32: `2737 / 5760 = 47.5%`
  - `mixed_stress` action 24: `3342 / 5760 = 58.0%`
- The concentration did not coincide with no-current/no-unassigned/failed-noop
  collapse.

## Action Decode

The discrete action space is composed from:

- `dispatch`: hold or dispatch.
- `route`: `shortest`, `low_congestion`, `high_resilience`.
- `mode`: `secondary_fleet`, `primary_fleet`.
- `reorder`: `none`, `conservative`, `aggressive`, `emergency`.

Decoded watches:

- Action 24 = `dispatch + shortest + secondary_fleet + none`
- Action 32 = `dispatch + low_congestion + secondary_fleet + none`

## KPI Mapping

| Model / sim metric | Real logistics KPI mapping | Operational question | Caveat |
| --- | --- | --- | --- |
| `service_level` | OTIF, on-time delivery rate, fill rate, customer promise adherence | Are orders delivered within the promised window? | Needs exact promise-window and exception definitions. |
| `lateness`, `true_lateness_pressure` | Late delivery rate, mean/percentile lateness, SLA breach pressure | Is the model trading too much timeliness for other objectives? | Sim lateness must be calibrated to real due-window distributions. |
| `dispatch_success` | Successful dispatch attempt rate, dispatch acceptance, carrier assignment success | Are dispatches executable? | Needs real dispatch failure taxonomy. |
| no-current / no-unassigned | Empty work queue actions, invalid dispatch attempt rate, dispatcher no-op rate | Is the policy attempting work when none exists? | Sim row context may not map one-to-one to a dispatch platform event. |
| `route_failure` | Failed route plan, reroute required, failed attempt, infeasible route rate | Are route choices operationally robust? | Must separate route infeasibility from customer absence or carrier failure. |
| `no_vehicle` | Capacity shortage, carrier unavailable, load not covered, fleet stockout | Is demand exceeding available fleet? | Needs planned capacity and actual availability timestamps. |
| `already_assigned` | Duplicate assignment attempt, lock conflict, idempotency conflict | Is the dispatcher trying to reassign committed work? | Real systems may suppress duplicate attempts before policy sees them. |
| Route distribution | Carrier route mix, shortest/fastest/low-risk route selection, route reliability | Does route choice match operational conditions? | Public datasets can validate route proxies better than exact policy labels. |
| Fleet distribution | Primary versus overflow/secondary carrier mix, outsource rate | Is secondary fleet use cost-realistic? | Requires company carrier contracts, availability, and costs. |
| Reorder distribution | Replenishment rate, emergency order rate, stockout avoidance, holding-cost policy | Is inventory action realistic? | Requires inventory and demand data; public routing data cannot validate it. |
| Delivered per step | Throughput, fulfilled units per hour, stop completion rate | Is throughput stable under stress? | Needs normalization by route length, shift length, and order mix. |

## Action 24: `dispatch + shortest + secondary_fleet + none`

### When It Is Realistic

Action 24 is realistic when:

- Demand is urgent and an order can be dispatched immediately.
- The shortest route is also operationally acceptable.
- Primary fleet is scarce, committed, or slower than overflow/secondary fleet.
- Inventory is already sufficient, so reorder action `none` is correct.
- Secondary fleet cost is acceptable relative to SLA penalties.

A real-world analogue is an overflow carrier or contracted secondary fleet being
used for an urgent job where route simplicity and time-to-dispatch dominate.

### When It Is Risky

Action 24 becomes risky when:

- Secondary fleet is materially more expensive than primary fleet.
- The shortest route is fragile under congestion or time-window pressure.
- No-reorder hides a stockout or future backlog risk.
- The sim underprices secondary-fleet use or holding/stockout cost.
- Secondary fleet has weaker service reliability than modeled.

The main risk is a cost realism gap: the model may be optimizing service while
underweighting secondary-fleet cost, carrier constraints, or inventory effects.

### Data Needed To Validate Or Refute

Useful real data:

- Primary and secondary fleet assignment by order.
- Carrier/fleet costs, acceptance rates, cancellation rates, and SLA outcomes.
- Planned versus actual route choice.
- Dispatch timestamp, pickup timestamp, delivery timestamp.
- Stockout/backlog status at dispatch time.
- Reorder event history and replenishment lead time.
- Failed-attempt reason and route failure reason.

Public routing datasets can partially validate the route component, but not the
secondary-fleet and reorder components.

### Current Interpretation

The current simulation evidence does not indicate a bug or invalid-action
exploit. Action 24 concentrated in `mixed_stress`, but equal-budget service did
not regress and hard blockers stayed zero. Treat it as a valid internal policy
preference pending cost/fleet/inventory calibration.

## Action 32: `dispatch + low_congestion + secondary_fleet + none`

### When It Is Realistic

Action 32 is realistic when:

- Congestion or disruption makes shortest-route dispatch less reliable.
- Low-congestion routing improves time-window adherence.
- Secondary fleet is available and can preserve service under capacity stress.
- Inventory is already available, so reorder is unnecessary.

A real-world analogue is shifting to a lower-risk route and overflow carrier
under route disruption, even if that means using a non-primary fleet resource.

### When It Is Risky

Action 32 becomes risky when:

- Low-congestion route labels are too optimistic or leak future information.
- Secondary fleet reliability is lower than modeled.
- Route disruption costs are miscalibrated.
- The model overuses secondary fleet instead of accepting bounded lateness or
  batching work.
- No-reorder suppresses necessary replenishment under demand pressure.

The main risk is not the low-congestion route itself; that is plausible under
stress. The risk is whether the secondary-fleet/no-reorder pairing is priced and
constrained realistically.

### Data Needed To Validate Or Refute

Useful real data:

- Historical travel-time distributions by route segment and time of day.
- Congestion or delay flags from routing providers or telemetry.
- Planned route versus actual route.
- Failed-route and reroute events.
- Carrier/fleet type used per dispatch.
- Fleet availability and acceptance rate at decision time.
- Cost and SLA penalty data.

Public datasets can validate route reliability proxies. Company data is needed
to validate secondary-fleet and no-reorder choices.

### Current Interpretation

The current simulation evidence does not indicate a bug. Action 32 concentrated
in `route_disruption_congestion`, which is exactly the scenario where
low-congestion routing should become attractive. Equal-budget service was only
slightly below old production and inside gate tolerance. Treat it as a plausible
policy preference with an active calibration watch.

## Public Dataset And Benchmark Review

| Dataset / source | Link | What it contains | License / access | What it can answer | What it cannot answer | Relevance |
| --- | --- | --- | --- | --- | --- | --- |
| Amazon Last Mile Routing Research Challenge | https://registry.opendata.aws/amazon-last-mile-challenges/ | Real route-, stop-, and package-level last-mile features from 9,184 historical Amazon routes in 2018 across five U.S. metro areas. | Public AWS Open Data Registry; CC BY-NC 4.0. | Actual stop sequences, time windows, service times, package dimensions, historical travel-time matrix, route quality proxies. | No company-specific fleet split, carrier cost, inventory/reorder state, or exact dispatch policy. | Strong route-choice and service-time calibration source; partial validation for action 24/32 route component. |
| Amazon Last Mile data paper | https://www.amazon.science/publications/2021-amazon-last-mile-routing-research-challenge-data-set | Dataset description and research context for the Amazon challenge. | Public paper page. | Confirms operational provenance and feature level. | Does not add carrier/inventory/cost fields. | Supports dataset credibility. |
| MIT-CAVE Challenge data structures | https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md | Documented fields for route data, package data, travel times, and actual sequences. Includes time windows, service times, delivered/attempted package status, route scores, and realized travel times. | Public GitHub documentation tied to challenge data. | Defines concrete columns for route-choice proxy validation. | Does not define secondary-fleet economics or inventory state. | Good schema reference for Phase B/C. |
| Waterloo Amazon Challenge ATSP instances | https://www.math.uwaterloo.ca/tsp/amz/data.html | Practical ATSP transformations from Amazon routes, including training/evaluation instances and driver sequences. | CC BY-NC. | Route optimization benchmark and sequence comparison. | Does not validate dispatch mode or reorder behavior. | Useful for algorithmic route-choice proxy checks. |
| Solomon VRPTW benchmark | https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/ | Classic vehicle-routing-with-time-windows instances and best known solutions. | Public benchmark. | Time-window routing, route feasibility, vehicle-count/distance tradeoffs. | Synthetic, not operational telemetry; no fleet split or inventory. | Useful sanity benchmark, not direct real-world validation. |
| Homberger / Gehring extended VRPTW benchmarks | https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/ | Larger VRPTW benchmark instances. | Public benchmark. | Scaling route feasibility and time-window objectives. | Synthetic; no dispatcher, carrier, cost, or reorder events. | Useful for route-policy stress tests only. |
| CVRPLIB | https://galgos.inf.puc-rio.br/cvrplib/ | Public library of capacitated vehicle-routing instances and best known solutions. | Public benchmark site; individual sets may have separate citation terms. | Vehicle capacity and route optimization behavior. | Not real dispatch telemetry; usually lacks congestion, time windows, fleet economics, reorder state. | Partial proxy for capacity and fleet constraints. |
| OpenStreetMap | https://www.openstreetmap.org/copyright | Road-network data under ODbL. | ODbL with share-alike obligations for derived databases. | Road graph, distance, accessibility, route geometry. | No delivery outcomes, customer promise windows, carrier mode, inventory, or actual dispatch decisions. | Base map for route-cost calibration and synthetic routing. |
| OSRM | https://project-osrm.org/ | Routing engine using OpenStreetMap data, with shortest/fastest path and travel-time matrix support. | BSD-style software license; OSM data license applies separately. | Route distance/time proxies and alternative route generation. | Not an outcome dataset; no actual fleet or inventory behavior. | Useful for shortest versus low-congestion route proxy features. |
| openrouteservice | https://openrouteservice.org/ | OSM-based routing, isochrones, and time-distance matrices. | Public service and open-source ecosystem; OSM data license applies. | Travel-time matrices, accessibility, routing alternatives. | No direct last-mile operational outcomes or inventory data. | Useful route-calibration proxy. |
| NYC TLC Trip Record Data | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page | Taxi/FHV trip pickup/dropoff times, zones, distances, fares, and related fields. | Public official city data. | Urban travel-time and congestion proxy by time/location. | Passenger trips are not delivery routes; no time windows, package service times, or fleet/reorder actions. | Supplemental congestion calibration source. |
| BTS freight data | https://www.bts.gov/topics/freight-transportation | U.S. freight indicators, Freight Analysis Framework, Commodity Flow Survey, and macro freight data. | Public U.S. government data. | Macro demand, flow, and freight trend context. | Not last-mile action-level telemetry. | Useful background calibration, not action 24/32 validation. |

## What Public Data Can Prove

Public data can support or refute:

- Whether shortest-route choices are common under low-stress conditions.
- Whether actual high-quality routes deviate from shortest under time-window or
  travel-time pressure.
- Whether service-time and travel-time distributions in the sim are realistic.
- Whether route disruption/congestion assumptions are directionally plausible.
- Whether route-choice concentration is unusual compared with route-quality
  proxies.

Public data can only partially validate action 24 and 32 because those actions
combine route, fleet, and reorder decisions.

## What Public Data Cannot Prove

Public data generally cannot prove:

- Whether this company should use `secondary_fleet` at the observed rate.
- Whether secondary fleet is cheaper or more reliable than modeled.
- Whether `none` reorder is correct under mixed stress.
- Whether real carrier capacity, acceptance, cancellation, and SLA penalty
  curves match the simulator.
- Whether `already_assigned` and dispatch-lock semantics match real production.
- Whether the simulation's reward/cost weights match business economics.

Those require company data.

## Ideal Company Data Contract

| Column | Purpose |
| --- | --- |
| `order_id` | Join key across order, dispatch, inventory, and delivery records. |
| `customer_id` / `customer_location_id` | Customer/location segmentation and route clustering. |
| `latitude`, `longitude`, `zone_id` | Routing, distance, and local congestion calibration. |
| `created_at` | Demand arrival process. |
| `promised_window_start`, `promised_window_end` | SLA and time-window pressure. |
| `actual_dispatch_time` | Dispatch timing and queue delay. |
| `pickup_time` | Carrier pickup latency. |
| `delivery_time` | On-time and lateness computation. |
| `delivered_on_time` | OTIF / service target. |
| `delivery_status` | Delivered, attempted, cancelled, rejected, failed. |
| `failed_attempt_reason` | Route failure versus customer absence versus carrier issue. |
| `vehicle_id` | Vehicle-level capacity and utilization. |
| `carrier_id` | Carrier performance and cost modeling. |
| `fleet_type` | Primary versus secondary fleet label. |
| `planned_route_id` | Planned route choice. |
| `planned_route_type` | Shortest, low-congestion, high-resilience, or operational equivalent. |
| `actual_route_id` | Actual executed route. |
| `planned_distance`, `actual_distance` | Route efficiency and deviation. |
| `planned_travel_time`, `actual_travel_time` | Congestion and route reliability. |
| `congestion_delay_flag` | Delay attribution. |
| `route_failure_flag` | Route infeasibility or reroute event. |
| `no_vehicle_flag` | Capacity shortage event. |
| `already_assigned_flag` | Duplicate/locked assignment event. |
| `inventory_item_id` | Reorder and stockout join key. |
| `stock_on_hand_at_decision` | Reorder necessity. |
| `stockout_flag` | Stockout rate and penalty. |
| `backlog_quantity` | Demand pressure. |
| `reorder_event_id` | Replenishment action. |
| `reorder_type` | Conservative/aggressive/emergency equivalent. |
| `supplier_lead_time` | Lead-time calibration. |
| `holding_cost` | Inventory holding economics. |
| `stockout_cost` | Service penalty calibration. |
| `secondary_fleet_cost` | Fleet-mode tradeoff calibration. |
| `primary_fleet_cost` | Baseline fleet-mode cost. |
| `sla_penalty_cost` | Explicit service tradeoff. |

## Simulation Calibration Targets

| Simulation parameter | Real data source | Calibration target |
| --- | --- | --- |
| Demand arrival rates | `created_at` by zone/product/daypart | Match order arrival intensity and burstiness. |
| SLA due-window distribution | Promised delivery windows | Match due-window width and urgency mix. |
| Lead-time distribution | Supplier/order fulfillment timestamps | Match replenishment and fulfillment delay. |
| Congestion multipliers | Planned versus actual travel time, road segment telemetry | Match travel-time inflation by time/location/stress. |
| Route failure probability | Failed/rerouted route events | Match route infeasibility and disruption rates. |
| Vehicle availability | Vehicle/carrier availability logs | Match capacity shortage frequency. |
| Vehicle capacity | Vehicle manifest and order dimensions | Match load feasibility. |
| Primary versus secondary fleet cost | Carrier contracts and actual invoices | Price secondary-fleet use realistically. |
| Primary versus secondary fleet reliability | Acceptance/cancellation/on-time rates by fleet type | Match service and failure rates. |
| Reorder cost | Purchase/replenishment records | Match reorder action economics. |
| Stockout cost | Lost sale, backorder, SLA penalty data | Match service impact of stockout. |
| Holding cost | Inventory cost accounting | Prevent unrealistic no-reorder or over-reorder behavior. |
| Service time / stop time | Actual stop duration or package service time | Match delivery route throughput. |
| Failed attempt probability | Delivery attempt outcomes | Match attempt failure rates and reasons. |
| Dispatch feasibility | Assignment logs and rejection reasons | Match invalid/no-vehicle/already-assigned rates. |

## Reality-Gap Assessment

### Are Actions 24 And 32 Plausible?

Yes. Both actions are operationally plausible:

- Action 24 is plausible when shortest route is adequate, secondary fleet is the
  fastest available capacity, and no reorder is needed.
- Action 32 is plausible when low-congestion routing protects service under
  disruption and secondary fleet supplies overflow capacity.

### Are They Proven Real-World Optimal?

No. The current evidence proves only that they are effective under the current
simulation and gate semantics. Real-world optimality requires:

- Real secondary-fleet cost/reliability data.
- Real inventory and reorder economics.
- Real route choice and travel-time outcomes.
- Real failure and exception taxonomy.
- Business-approved cost tradeoffs between service, fleet, and inventory.

### Could Secondary-Fleet + No-Reorder Concentration Be A Cost Realism Gap?

Yes. This is the main unresolved risk. If secondary fleet is underpriced, too
available, or too reliable in the simulator, the model may rationally overuse it.
If reorder/stockout/holding costs are mispriced, the model may choose `none`
more often than a real operation would.

### Could Shortest / Low-Congestion Concentration Be Realistic Under Stress?

Yes. Concentrated route choice can be realistic when a scenario strongly favors
one operational strategy. In `route_disruption_congestion`, a high share of
`low_congestion` dispatch is directionally sensible. In `mixed_stress`, a high
share of `shortest` dispatch can be sensible if urgency and vehicle scarcity
make immediate shortest-route execution dominate.

The concentration is suspicious only if it coincides with degraded service,
invalid-action spikes, hidden future information, or unrealistic cost treatment.
The equal-budget audit did not show those failure modes, so this remains a
monitoring and calibration watch.

## First Validation Experiment

Run a route-choice proxy validation using the Amazon Last Mile dataset:

1. Build per-route features from time windows, package service times, historical
   travel-time matrix, actual driver sequence, and route score.
2. Derive route-choice proxies:
   - shortest-distance or shortest-time route,
   - low-travel-time-variance route,
   - lower-backtracking/high-reliability route,
   - actual driver route.
3. Segment by stress:
   - tight time windows,
   - high travel-time matrix asymmetry,
   - long service-time routes,
   - route-score quality buckets.
4. Measure whether high-quality actual routes move away from shortest under
   pressure.
5. Compare the simulator's `shortest` and `low_congestion` route distributions
   against those proxy patterns.

This experiment can validate the route component of actions 24 and 32. It
cannot validate secondary-fleet use or no-reorder behavior.

## Proposed Validation Phases

### Phase A: KPI Mapping And Data Contract

Deliverables:

- Final KPI mapping table.
- Agreed definitions for OTIF, service level, lateness, failed attempt, route
  failure, no vehicle, duplicate assignment, secondary fleet, and reorder.
- Company data extract schema and privacy constraints.

Exit criteria:

- Business and engineering agree the sim metrics map to measurable operational
  KPIs.

### Phase B: Public Dataset Exploratory Analysis

Deliverables:

- Amazon Last Mile route/stop/package schema profile.
- Route-score, time-window, service-time, and travel-time distribution summary.
- Candidate route-choice proxy labels.

Exit criteria:

- We know whether public route data can calibrate the route choice side of the
  simulator.

### Phase C: Route-Choice Proxy Validation

Deliverables:

- Shortest versus low-congestion/resilience proxy comparison.
- Route quality relationship to time-window pressure and travel-time asymmetry.
- Scenario calibration recommendations for congestion and route disruption.

Exit criteria:

- Route-choice concentration is either directionally supported by public route
  evidence or marked as needing company route telemetry.

### Phase D: Fleet / Reorder Calibration With Company Data

Deliverables:

- Secondary-fleet availability, cost, reliability, and acceptance calibration.
- Primary versus secondary fleet outcome comparison.
- Reorder/stockout/holding-cost calibration.
- Dispatch infeasibility and duplicate-assignment rate calibration.

Exit criteria:

- Action 24/32 fleet and reorder components are either supported or flagged as a
  material reality gap.

### Phase E: Optional Sim Parameter Retuning

Deliverables:

- Parameter-only retuning proposal, if evidence supports it.
- Separate gated training plan only if retuning changes the learning target.

Exit criteria:

- No training begins without a separate approved plan.
- No blind 3M run is recommended.

## Decision Gates

| Gate | Pass condition | If failed |
| --- | --- | --- |
| KPI/data-contract gate | Sim metrics have accepted real KPI definitions. | Stop and refine metric definitions. |
| Public route-proxy gate | Public data supports or constrains route-choice assumptions. | Require company route telemetry before retuning. |
| Fleet/reorder gate | Company data supports secondary-fleet and reorder economics. | Treat action 24/32 concentration as a reality gap. |
| Calibration gate | Parameter changes are evidence-backed and bounded. | Do not retune. |
| Training gate | Separate plan exists with explicit approval. | Do not train. |

## What Not To Do

- Do not treat action concentration alone as a bug.
- Do not start 3M/5M/10M/100M.
- Do not train from this research plan.
- Do not retune costs without evidence.
- Do not use public routing benchmarks as proof of fleet or inventory realism.

## Protected No-Mutation Position

This plan is documentation-only. It does not require:

- registry writes,
- production writes,
- baseline writes,
- DB writes,
- checkpoint writes,
- training,
- offline eval,
- edits to existing eval outputs.

Allowed writes for this task:

- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- optional dashboard pointer in `docs/00_PROJECT_DASHBOARD.md`

## Reviewer Verdict

Independent read-only reviewer verdict:

`REAL_WORLD_CALIBRATION_REQUIRES_COMPANY_DATA`

Reviewer scope was read-only and covered action decoding, KPI mapping,
public-versus-company-data limitations, no-training/no-mutation constraints, and
final classification justification.

## Plan Classification

`REAL_WORLD_CALIBRATION_REQUIRES_COMPANY_DATA`

Rationale: the route-choice part can be researched with public data, but the
action 24/32 watch combines route, fleet, and reorder decisions. Full validation
of secondary-fleet and no-reorder concentration requires company data.

## References

- Amazon Last Mile Routing Research Challenge:
  https://registry.opendata.aws/amazon-last-mile-challenges/
- Amazon Science dataset description:
  https://www.amazon.science/publications/2021-amazon-last-mile-routing-research-challenge-data-set
- MIT-CAVE challenge data structures:
  https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md
- Waterloo Amazon Challenge ATSP instances:
  https://www.math.uwaterloo.ca/tsp/amz/data.html
- Solomon VRPTW benchmark:
  https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/
- Homberger VRPTW benchmark:
  https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/
- CVRPLIB:
  https://galgos.inf.puc-rio.br/cvrplib/
- OpenStreetMap copyright and license:
  https://www.openstreetmap.org/copyright
- OSRM:
  https://project-osrm.org/
- openrouteservice:
  https://openrouteservice.org/
- NYC TLC Trip Record Data:
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- BTS Freight Transportation:
  https://www.bts.gov/topics/freight-transportation
