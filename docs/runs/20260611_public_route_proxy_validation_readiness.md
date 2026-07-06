# Public Route-Choice Proxy Validation Readiness

Date: 2026-06-11

Pre-review classification: `PUBLIC_ROUTE_PROXY_VALIDATION_READY`

## Scope

Company data is unavailable. This readiness package validates only the
route-choice side of the current hierarchical v1 residual watches:

- action 24 route component: `shortest`
- action 32 route component: `low_congestion`

It does not attempt to validate:

- `secondary_fleet` economics,
- primary versus secondary carrier constraints,
- reorder or inventory economics,
- company-specific dispatch feasibility or costs.

No training, offline scenario eval, registry update, production mutation,
baseline mutation, DB mutation, checkpoint mutation, existing eval-output edit,
or 3M/5M/10M/100M run was performed.

## Local Context

Current production model:

- logical model: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- DQN architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- obs/action: `73 / 48`

Equal-budget residual-watch state:

- old production and hierarchical production both ran 20 episodes per scenario
  with seed 42
- both passed 8/8 scenarios
- equal-budget long-run gate passed
- remaining watches are monitoring items, not training triggers

Watched route actions:

- action 24 = `dispatch + shortest + secondary_fleet + none`
- action 32 = `dispatch + low_congestion + secondary_fleet + none`

The public proxy validation package therefore focuses only on:

- shortest-route plausibility,
- low-congestion / travel-time-reliability plausibility,
- stress-conditioned route concentration.

## Dataset Decision

Best first target: Amazon Last Mile Routing Research Challenge dataset.

Rationale:

- It is the closest public dataset to actual last-mile route execution.
- It includes real historical Amazon delivery routes rather than synthetic-only
  benchmark instances.
- It includes route-, stop-, package-, time-window-, travel-time-, and actual
  driver sequence data.
- It provides route quality labels that reflect time-window adherence and
  backtracking.

Limitations:

- It does not expose company-specific primary/secondary fleet decisions.
- It does not expose carrier contracts, overflow carrier costs, or acceptance
  rates.
- It does not expose inventory/reorder state.
- It cannot prove that action 24 or 32 is globally optimal, only that the route
  component is or is not directionally plausible.

Secondary data sources:

| Source | Use | Limitation |
| --- | --- | --- |
| MIT-CAVE schema docs | Concrete Amazon file/field definitions | Documentation only; requires dataset for analysis |
| Solomon VRPTW / Homberger | Synthetic time-window route sanity benchmarks | Not real driver behavior and no congestion/fleet/reorder |
| CVRPLIB | Capacity and route optimization benchmark context | Limited/no route telemetry and no inventory/fleet economics |
| OSM + OSRM/openrouteservice | Route distance/time matrix and alternative-route proxies | No delivery outcomes or dispatch decisions |

## Public Access Notes

Important public facts:

- AWS Open Data Registry describes 9,184 historical Amazon routes performed in
  2018 across five U.S. metro areas.
- Open Data Registry resource bucket: `s3://amazon-last-mile-challenges`
- Dataset folder: `almrrc2021/`
- Training folder: `almrrc2021_data_training`, 6,112 routes.
- Evaluation folder: `almrrc2021_data_evaluation`, 3,072 routes.
- License: Creative Commons Attribution-NonCommercial 4.0.
- Waterloo's Amazon Challenge Data page documents the public sync command:
  `aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/ ./almrrc2021/`

No dataset download was performed in this task. `aws` is not installed in the
current environment, so a local size estimate could not be produced.

## Exact Access Instructions

First install AWS CLI if needed, then estimate size:

```powershell
aws s3 ls --summarize --human-readable --recursive --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/
```

If the reported size is acceptable and explicitly approved, download to a fresh
non-production local data directory:

```powershell
New-Item -ItemType Directory -Force data\public | Out-Null
aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/ data\public\almrrc2021\
```

After download, run schema probe:

```powershell
python scripts\real_world_calibration\amazon_last_mile_schema_probe.py data\public\almrrc2021
```

Then run route proxy summary:

```powershell
python scripts\real_world_calibration\route_proxy_metrics.py data\public\almrrc2021 --route-limit 25
```

## Needed Amazon Fields

Required fields:

- `route_data.json`
  - `route_score`
  - `station_code`
  - `date_YYYY_MM_DD`
  - `departure_time_utc`
  - `executor_capacity_cm3`
  - `stops[stop_id].lat`
  - `stops[stop_id].lng`
  - `stops[stop_id].type`
  - `stops[stop_id].zone_id`
- `actual_sequences.json`
  - `actual_sequences[route_id]["actual"][stop_id]`
- `package_data.json`
  - `scan_status`
  - `time_window.start_time_utc`
  - `time_window.end_time_utc`
  - `planned_service_time_seconds`
  - `dimensions`
- `travel_times.json`
  - directed historical average travel time matrix by route and stop pair
- optional:
  - `invalid_sequence_scores.json`
  - `new_*` model-apply / model-score files

## Proxy Labels

Shortest route proxy:

- Uses shortest geometric or route-matrix path length where distances are
  available.
- With Amazon data, point-to-point travel time is present and exact geometry is
  obfuscated, so shortest proxy should be treated as "lowest matrix/path cost"
  rather than literal road distance unless OSM/OSRM is joined separately.

Fastest route proxy:

- Uses lowest directed travel-time sequence cost from `travel_times.json`.
- Initial lightweight proxy can use greedy nearest-neighbor by travel time;
  deeper work can use a TSP/ATSP solver or route-optimization library.

Low-congestion / low-variance proxy:

- Uses directed travel-time asymmetry as a disruption/congestion proxy.
- High asymmetry suggests unstable or directionally constrained travel.
- If multiple daily/time-sliced matrices become available, use travel-time
  variance by arc/time bucket. Amazon's released matrix is historical average,
  so asymmetry is the first safe proxy.

Actual driver route proxy:

- Uses `actual_sequences.json` sorted by actual stop rank.
- Route quality comes from `route_score` in `route_data.json`.

Tight time-window stress proxy:

- Uses package `time_window` lengths.
- Routes with high share of short delivery windows are stress cases.

Travel-time asymmetry / disruption proxy:

- Uses normalized directed-pair gap:
  `abs(t_ij - t_ji) / ((t_ij + t_ji) / 2)`.
- Aggregate by mean and max per route.

Route concentration proxy:

- Computes the top route-label share within scenario or public-data stress
  buckets.
- Public route labels are proxy labels, not simulator action labels.

## Validation Questions

1. Under tight time windows, do high-quality actual routes look shortest-like,
   fastest-like, or reliability-oriented?
2. Under high travel-time asymmetry, do high-quality actual routes avoid brittle
   shortest/fastest proxies?
3. Does route-choice concentration increase in public route data under stress
   buckets?
4. Is action 24's shortest-route component directionally plausible in mixed
   stress when urgency dominates?
5. Is action 32's low-congestion-route component directionally plausible in
   route-disruption stress when directed travel-time asymmetry is high?
6. Are the simulator's concentrated route preferences more concentrated than
   public actual-driver route proxies under similar stress?

## Local Analysis Package

Created:

- `scripts/real_world_calibration/README.md`
- `scripts/real_world_calibration/__init__.py`
- `scripts/real_world_calibration/amazon_last_mile_schema_probe.py`
- `scripts/real_world_calibration/route_proxy_metrics.py`
- `tests/real_world_calibration/test_route_proxy_metrics.py`

The scripts do not require a dataset to import and do not download data. They
only inspect a local path passed by the user.

Pure helper coverage:

- directed sequence travel-time computation
- actual sequence extraction from rank maps
- normalized travel-time asymmetry
- route concentration
- tight time-window stress
- deterministic bootstrap mean interval

## What This Can Validate About Action 24 / 32

Can validate partially:

- whether shortest-like actual routes are plausible under urgent/tight-window
  stress,
- whether low-congestion/reliability-like actual routes are plausible under
  travel-time asymmetry/disruption stress,
- whether route-choice concentration appears in high-quality real routes under
  stress buckets,
- whether route concentration itself is unusual or normal in comparable public
  routing data.

Cannot validate:

- whether `secondary_fleet` should be used at the observed rate,
- whether no reorder is correct,
- whether secondary fleet cost/reliability is realistic,
- whether company dispatch feasibility, carrier acceptance, or inventory costs
  match the simulator.

## Recommended Next Command

Because `aws` is not installed and the dataset size was not locally estimated,
the next safe command is a size check after installing AWS CLI:

```powershell
aws s3 ls --summarize --human-readable --recursive --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/
```

If the size is acceptable, explicitly approve the sync command in a separate
message before download.

## Tests

Focused TDD run:

```powershell
python -m unittest tests.real_world_calibration.test_route_proxy_metrics
```

Result: `6` tests passed.

Relevant contract/config tests:

```powershell
python -m unittest tests.act.test_discrete_action_mapper_contract tests.eval.test_real_world_scenario_configs
```

Result: `5` tests passed.

Compile check:

```powershell
python -m py_compile scripts/real_world_calibration/amazon_last_mile_schema_probe.py scripts/real_world_calibration/route_proxy_metrics.py
```

Result: exit code `0`.

Import-safe no-data script probes:

```powershell
python scripts/real_world_calibration/amazon_last_mile_schema_probe.py .\tmp\nonexistent_almrrc2021_probe
python scripts/real_world_calibration/route_proxy_metrics.py .\tmp\nonexistent_almrrc2021_probe --route-limit 1
```

Result: both commands exited `0` and reported no local data / no route summaries
without attempting download.

## Protected No-Mutation Proof

Task mutations were limited to:

- new route-calibration scripts under `scripts/real_world_calibration`
- new focused tests under `tests/real_world_calibration`
- this readiness report
- optional dashboard pointer

No model, registry, production, baseline, DB, checkpoint, or existing eval-output
path was intentionally written.

## Independent Review

Independent read-only reviewer verdict:

`PUBLIC_ROUTE_PROXY_VALIDATION_READY`

Reviewer scope covered route-only framing, action 24/32 decode, dataset choice,
proxy labels, validation questions, import-safe scripts, tests, and no-mutation
constraints.

## Classification

`PUBLIC_ROUTE_PROXY_VALIDATION_READY`

Rationale: company data is still required for full action 24/32 validation, but
the public route-choice proxy work can proceed safely once the Amazon dataset is
downloaded under separate explicit approval.

## References

- Amazon Last Mile Routing Research Challenge:
  https://registry.opendata.aws/amazon-last-mile-challenges/
- AWS Open Data Registry YAML:
  https://github.com/awslabs/open-data-registry/blob/main/datasets/amazon-last-mile-challenges.yaml
- MIT-CAVE data structures:
  https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md
- Waterloo Amazon Challenge Data:
  https://www.math.uwaterloo.ca/tsp/amz/data.html
- Solomon VRPTW 100-customer benchmark:
  https://www.sintef.no/projectweb/top/vrptw/100-customers/
- Homberger / Gehring 1000-customer benchmark:
  https://www.sintef.no/projectweb/top/vrptw/1000-customers/
- OSRM:
  https://project-osrm.org/
- openrouteservice:
  https://openrouteservice.org/
- OpenStreetMap copyright and license:
  https://www.openstreetmap.org/copyright
