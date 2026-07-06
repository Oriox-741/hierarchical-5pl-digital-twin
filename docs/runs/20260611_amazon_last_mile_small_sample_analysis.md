# Amazon Last Mile Small-Sample Route Proxy Analysis

Date: 2026-06-11

Pre-review classification: `AMAZON_SMALL_SAMPLE_ROUTE_PROXY_READY`

## Scope

This analysis uses only the approved Amazon Last Mile small sample files in:

`data/public/almrrc2021_small/`

Objective:

- inspect schema,
- run lightweight route-choice proxy metrics,
- evaluate whether the route side of action 24 and action 32 is directionally
  plausible in public last-mile route data.

Hard boundaries:

- no full `3.1 GiB` dataset download
- no training
- no project offline eval
- no registry mutation
- no production mutation
- no baseline mutation
- no DB mutation
- no checkpoint mutation
- no existing eval-output edit

## Model Context

Watched production route actions:

- action 24 = `dispatch + shortest + secondary_fleet + none`
- action 32 = `dispatch + low_congestion + secondary_fleet + none`

This public sample can only address:

- `shortest` / fastest-like route plausibility,
- `low_congestion` / reliability-like route plausibility,
- route concentration and stress-bucket feasibility.

It cannot validate:

- `secondary_fleet`,
- primary versus secondary fleet economics,
- reorder decisions,
- stockout or holding-cost economics,
- company-specific dispatch constraints.

## Data Access

Manual S3 listing summary supplied by user:

- Total objects: `42`
- Total size: `3.1 GiB`

AWS CLI used:

```text
C:\Program Files\Amazon\AWSCLIV2\aws.exe
aws-cli/2.35.2 Python/3.14.5 Windows/11 exe/AMD64
```

Approved-file size gate:

- total approved size: `5,249,870` bytes / `5.007` MiB
- status: below `10 MiB`; safe to download

Downloaded only the seven approved files:

- `License.txt`
- `Readme.txt`
- `new_route_data.json`
- `new_package_data.json`
- `new_travel_times.json`
- `new_actual_sequences.json`
- `new_invalid_sequence_scores.json`

Full listing and hashes are recorded in:

`docs/runs/20260611_amazon_last_mile_s3_listing_audit.md`

## File Inventory

| File | Size bytes | SHA256 |
| --- | ---: | --- |
| `License.txt` | `19342` | `48F99D700291C586C53CEED67E424E4513EF0A2BA1B61D1472D4113086D820C9` |
| `Readme.txt` | `422` | `1325A6079AD41DA79C150DD34BF20B866CE26330E3DAC2D28F8753E0F812A54D` |
| `new_actual_sequences.json` | `21937` | `174625CCF3E0AED722F1129FEEE558EE5672372A3757D475072F7E04C300C2DB` |
| `new_invalid_sequence_scores.json` | `884` | `D401C4A1C8D441962D2ADD760ADF14325FE9D08AF5573C0AFC287EEAB18D199B` |
| `new_package_data.json` | `717877` | `27FC132C7F9DE5498301844E27E0D57EBA7898BD2F2A247C51742D5B87400C95` |
| `new_route_data.json` | `178703` | `5B6292CF03ED6ABC1A8E37A61405C05E14751DC2820A0856F04FD14563F362A3` |
| `new_travel_times.json` | `4310705` | `3A96C295D70F1EB372C89126E77E69F60D7BB1B9D38B2A913777D9341D560AB6` |

## Schema Probe

Command:

```powershell
python scripts\real_world_calibration\amazon_last_mile_schema_probe.py data\public\almrrc2021_small --max-routes 3
```

Result:

- `new_route_data.json`: present
- `new_package_data.json`: present
- `new_travel_times.json`: present
- `new_actual_sequences.json`: present
- `new_invalid_sequence_scores.json`: present
- non-`new_` files are absent as expected

Schema summary:

- `new_route_data.json` top-level route IDs contain:
  - `station_code`
  - `date_YYYY_MM_DD`
  - `departure_time_utc`
  - `executor_capacity_cm3`
  - `stops`
- stop payload contains:
  - `lat`
  - `lng`
  - `type`
  - `zone_id`
- `new_package_data.json` route -> stop -> package payload contains:
  - `time_window`
  - `planned_service_time_seconds`
  - `dimensions`
- `new_travel_times.json` route -> origin stop -> destination stop contains
  directed travel-time seconds.
- `new_actual_sequences.json` route -> `actual` contains actual stop ranks.
- `new_invalid_sequence_scores.json` contains one score per route.

Schema-probe compatibility fix:

- `amazon_last_mile_schema_probe.py` now reports
  `new_invalid_sequence_scores.json` explicitly.
- Focused test added:
  `tests.real_world_calibration.test_amazon_last_mile_schema_probe`.

Route score caveat:

- `new_route_data.json` from `model_apply_inputs` does not include
  `route_score`.
- `new_invalid_sequence_scores.json` is available and can be used as the scoring
  sidecar for the model-apply sample.

## Route Count And Coverage

| Artifact | Route count |
| --- | ---: |
| `new_route_data.json` | `13` |
| `new_package_data.json` | `13` |
| `new_travel_times.json` | `13` |
| `new_actual_sequences.json` | `13` |
| `new_invalid_sequence_scores.json` | `13` |

Station-code distribution:

| Station | Routes |
| --- | ---: |
| `DBO2` | `2` |
| `DCH4` | `2` |
| `DLA7` | `4` |
| `DLA8` | `1` |
| `DLA9` | `1` |
| `DSE4` | `2` |
| `DSE5` | `1` |

Stop count:

- min: `105`
- mean: `157.77`
- median: `158`
- max: `193`

Package count:

- total: `3129`
- min per route: `205`
- mean per route: `240.69`
- median per route: `237`
- max per route: `287`

## Time Window Availability

Windowed packages:

- `210` packages had finite time-window lengths.
- `0` packages had windows at or below the current `2 hour` tight-window
  threshold.
- minimum finite window: `14,400` seconds / `4 hours`

Interpretation:

- The small sample can exercise time-window parsing.
- It does not contain truly tight two-hour windows, so it cannot strongly test
  urgent/tight-window stress behavior by itself.

## Travel-Time Matrix Shape

Each route has a directed stop-to-stop travel-time matrix matching the route's
stop count.

Example route:

- route: `RouteID_15baae2d-bf07-4967-956a-173d4036613f`
- stop count: `193`
- first origin sample: `AH`
- destination count from first origin: `193`
- sample travel times from `AH`:
  - `AH -> AH`: `0.0`
  - `AH -> AK`: `287.4`
  - `AH -> AN`: `249.4`
  - `AH -> AU`: `476.9`
  - `AH -> AV`: `421.2`

Travel-time asymmetry:

| Metric | Value |
| --- | ---: |
| min mean pair asymmetry | `0.0513` |
| mean pair asymmetry | `0.0979` |
| median mean pair asymmetry | `0.0920` |
| max mean pair asymmetry | `0.1637` |
| min max pair asymmetry | `0.9359` |
| mean max pair asymmetry | `1.6335` |
| median max pair asymmetry | `1.7647` |
| max max pair asymmetry | `2.0000` |

Interpretation:

- Directed travel-time asymmetry is present and nontrivial.
- This supports a low-congestion / reliability-style proxy for action 32's route
  component.
- It does not provide live congestion labels; asymmetry is a proxy, not proof.

## Actual Sequence Availability

Actual sequences are present for all `13` routes.

Station stops are present and ranked `0` in sampled actual sequences, so the
actual sequence can be compared against simple station-start route proxies.

## Route Proxy Metrics

Command:

```powershell
python scripts\real_world_calibration\route_proxy_metrics.py data\public\almrrc2021_small --route-limit 25
```

The script computed route summaries for all `13` routes.

TDD patch added:

- `greedy_nearest_neighbor_sequence`
- `route_efficiency_ratio`
- invalid sequence score loading in route summaries
- route summary fields:
  - `greedy_travel_time_seconds`
  - `actual_to_greedy_travel_time_ratio`
  - `invalid_sequence_score`

Focused tests:

```powershell
python -m unittest tests.real_world_calibration.test_route_proxy_metrics -v
```

Result: `8` tests passed.

After independent review flagged schema completeness, focused TDD tests were
added for:

- `new_invalid_sequence_scores.json` propagation into route summaries
- `new_invalid_sequence_scores.json` reporting from the schema probe

The updated focused test run passed `10` tests.

Compile check:

```powershell
python -m py_compile scripts\real_world_calibration\amazon_last_mile_schema_probe.py scripts\real_world_calibration\route_proxy_metrics.py
```

Result: exit code `0`.

Actual route travel time:

- min: `6219.7` seconds
- mean: `11705.38` seconds
- median: `11654.3` seconds
- max: `14255.5` seconds

Greedy nearest-neighbor travel-time proxy:

- min: `6630.2` seconds
- mean: `12123.45` seconds
- median: `11843.8` seconds
- max: `14939.7` seconds

Actual-to-greedy ratio:

- min: `0.8926`
- mean: `0.9673`
- median: `0.9728`
- max: `1.0277`
- actual better than or equal to greedy: `9 / 13`
- actual worse than greedy: `4 / 13`

Interpretation:

- Actual driver sequences are generally travel-time efficient relative to a
  simple greedy fastest-like proxy.
- The result supports the plausibility of shortest/fastest-like route pressure
  under last-mile routing.
- The greedy proxy is not a global optimum, so this does not prove shortest-route
  optimality.

## Per-Route Summary

| Route | Station | Stops | Packages | Windowed | Tight | Actual/greedy | Mean asym | Max asym | Invalid score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `15baae2d` | `DCH4` | `193` | `287` | `1` | `0` | `0.950` | `0.085` | `1.065` | `1.022` |
| `3f166f0e` | `DLA7` | `166` | `238` | `1` | `0` | `0.901` | `0.080` | `0.936` | `0.604` |
| `5486294a` | `DLA7` | `151` | `228` | `3` | `0` | `1.013` | `0.069` | `2.000` | `0.939` |
| `693060a6` | `DSE5` | `168` | `237` | `10` | `0` | `0.984` | `0.057` | `1.243` | `1.081` |
| `7f5d87f0` | `DBO2` | `182` | `253` | `17` | `0` | `0.917` | `0.092` | `1.826` | `0.912` |
| `9475872b` | `DSE4` | `187` | `283` | `10` | `0` | `0.893` | `0.080` | `1.765` | `1.218` |
| `a8f0009d` | `DLA8` | `140` | `205` | `21` | `0` | `0.973` | `0.114` | `1.973` | `0.827` |
| `bcc07fea` | `DLA7` | `140` | `212` | `34` | `0` | `0.987` | `0.111` | `1.727` | `0.893` |
| `d1a8c3dd` | `DBO2` | `154` | `250` | `12` | `0` | `1.028` | `0.144` | `1.857` | `1.182` |
| `e6687a05` | `DSE4` | `105` | `229` | `20` | `0` | `0.938` | `0.164` | `1.899` | `0.440` |
| `2b8df66d` | `DLA9` | `133` | `248` | `65` | `0` | `1.027` | `0.117` | `1.729` | `1.062` |
| `f3261fad` | `DLA7` | `158` | `231` | `14` | `0` | `1.007` | `0.108` | `1.807` | `0.675` |
| `fffd257c` | `DCH4` | `174` | `228` | `2` | `0` | `0.958` | `0.051` | `1.409` | `0.639` |

## Action 24 Route-Side Plausibility

Question:

Can this small sample test action 24's shortest-like route behavior?

Answer:

- Partially yes.
- The sample includes actual driver sequences and directed travel-time matrices.
- Actual routes are generally efficient relative to a simple greedy fastest-like
  route proxy: mean actual/greedy ratio `0.9673`, with `9 / 13` actual routes
  no worse than greedy.
- This supports the idea that shortest/fastest-like route pressure is plausible
  in real last-mile routing.

Limitations:

- Amazon locations are obfuscated, so this is not a literal road-distance
  shortest-path test.
- The greedy proxy is not an exact ATSP optimum.
- The sample has no tight two-hour windows, so urgent stress evidence is weak.
- This validates only action 24's route component, not `secondary_fleet` or
  `none` reorder.

## Action 32 Route-Side Plausibility

Question:

Can this small sample test action 32's low-congestion/reliability behavior?

Answer:

- Partially yes.
- The directed travel-time matrices show meaningful asymmetry.
- Mean pair asymmetry varies from `0.0513` to `0.1637`; max pair asymmetry
  reaches `2.0`.
- This supports using travel-time asymmetry as a reliability / disruption proxy,
  which is directionally aligned with low-congestion route preference.

Limitations:

- The sample does not expose live congestion labels.
- It does not label actual driver route decisions as `low_congestion`.
- It cannot prove action 32 is optimal, only that reliability-sensitive routing
  has a plausible public-data proxy.

## Route Concentration Under Stress Buckets

Question:

Can this sample say anything about route concentration under stress buckets?

Answer:

- Only weakly.
- It can form prototype stress buckets using:
  - travel-time asymmetry,
  - windowed package share,
  - route size / stop count.
- It cannot strongly validate concentration under urgent time-window stress
  because no packages in this small sample have windows at or below two hours.
- It cannot map public routes to simulator route labels (`shortest`,
  `low_congestion`, `high_resilience`) without additional proxy labeling.

Conclusion:

- The small sample is adequate for schema and first-pass route-proxy readiness.
- It is not sufficient for statistical concentration validation.

## What Remains Untestable Without Company Data

- secondary fleet cost and availability
- primary versus secondary carrier reliability
- carrier acceptance/cancellation behavior
- reorder, stockout, backlog, and holding-cost economics
- company SLA penalty curves
- dispatch lock / already-assigned semantics
- no-vehicle semantics in the real dispatch platform

## Decision

`AMAZON_SMALL_SAMPLE_ROUTE_PROXY_READY`

Rationale:

- selective download stayed under 10 MiB and used only approved keys
- schema probe succeeded
- route proxy metrics computed for all `13` routes
- action 24 and action 32 route components can be partially tested with public
  route proxies
- no training or project offline eval was run
- protected production/registry/baseline/DB/checkpoint areas were not mutated

## Independent Review

Final independent read-only reviewer verdict:

`AMAZON_SMALL_SAMPLE_ROUTE_PROXY_READY`

Review history:

- First review: `AMAZON_SMALL_SAMPLE_SCHEMA_NEEDS_FIXES`
- Fix: added invalid sequence score loading to route summaries.
- Second review: `AMAZON_SMALL_SAMPLE_SCHEMA_NEEDS_FIXES`
- Fix: added `new_invalid_sequence_scores.json` to the schema probe and a
  focused schema-probe test.
- Final review: `AMAZON_SMALL_SAMPLE_ROUTE_PROXY_READY`
