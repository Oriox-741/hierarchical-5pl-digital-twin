# Real-World Route Proxy Calibration Scripts

This folder contains lightweight helpers for public route-choice validation.

The scripts are safe by construction:

- no model training
- no offline scenario evaluation
- no registry, production, baseline, DB, checkpoint, or existing eval-output writes
- no dataset download
- no filesystem scan at import time

## First Target Dataset

Use the Amazon Last Mile Routing Research Challenge dataset first. It is the
best public fit for validating the route-choice side of action 24 and action 32
because it includes real last-mile route, stop, package, time-window, travel-time
matrix, and actual-driver-sequence data.

It cannot validate `secondary_fleet` or `reorder` economics.

## Expected Amazon Fields

Useful files:

- `route_data.json`
- `actual_sequences.json`
- `package_data.json`
- `travel_times.json`
- `invalid_sequence_scores.json`
- `new_route_data.json`
- `new_actual_sequences.json`
- `new_package_data.json`
- `new_travel_times.json`

Useful fields:

- route score: `route_data[route_id]["route_score"]`
- stops and zones: `route_data[route_id]["stops"]`
- actual sequence rank: `actual_sequences[route_id]["actual"]`
- package time windows: `package_data[route_id][stop_id][package_id]["time_window"]`
- package service time: `planned_service_time_seconds`
- directed travel time matrix: `travel_times[route_id][from_stop][to_stop]`

## Commands

Probe a local dataset root:

```powershell
python scripts\real_world_calibration\amazon_last_mile_schema_probe.py C:\path\to\almrrc2021
```

Summarize route proxy metrics:

```powershell
python scripts\real_world_calibration\route_proxy_metrics.py C:\path\to\almrrc2021 --route-limit 25
```

## Download Guidance

Do not download the dataset from this task without explicit approval.

The public AWS Open Data command documented by the Waterloo Amazon Challenge
page is:

```powershell
aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/ .\data\public\almrrc2021\
```

Before running it, estimate size with:

```powershell
aws s3 ls --summarize --human-readable --recursive --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/
```

Then ask for approval if the size is acceptable.

## Proxy Metrics

The pure helper functions currently support:

- directed sequence travel time
- actual-sequence extraction from rank maps
- normalized travel-time asymmetry
- route/action concentration
- tight time-window stress
- deterministic bootstrap mean intervals

These are designed to support route-side validation only.
