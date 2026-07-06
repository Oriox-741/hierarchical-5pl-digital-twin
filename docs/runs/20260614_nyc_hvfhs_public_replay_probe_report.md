# NYC HVFHS Public Replay Probe Report - 2026-06-14

## Classification

`PUBLIC_REPLAY_PROBE_READY_WITH_CAUSAL_LIMITATIONS`

## Source and Sample

- Source: `<PROJECT_ROOT>\data\public\nyc_hvfhs_20260614\fhvhv_tripdata_2023-01.parquet`
- Sample rows: `10000`
- Production policy loaded read-only: `True`
- No-policy dry-run row count: `10000`

## Replay Coverage

- Mean missingness: `0.3013698630136986`
- Confidence distribution: `{'high': 10000}`
- Aggregate feature status counts: `{'observed': 230000, 'imputed': 270000, 'unavailable': 220000, 'neutral': 10000}`

## Model Action Tendencies

- Prediction count: `10000`
- Dispatch rate: `0.0466`
- Hold rate: `0.9534`
- Action 24 rate: `0.002`
- Action 32 rate: `0.0`
- Top actions: `[('1', 7441), ('0', 1958), ('25', 217), ('41', 148), ('2', 135), ('29', 68), ('24', 20), ('40', 7)]`
- Route distribution: `{'shortest': 9839, 'high_resilience': 161}`
- Mode distribution: `{'secondary_fleet': 9929, 'primary_fleet': 71}`
- Reorder distribution: `{'none': 1985, 'conservative': 7877, 'aggressive': 135, 'emergency': 3}`

## Proxy Timing / Correlation

- Service proxy summary: `{'count': 10000, 'mean': 0.5264913340935006, 'min': 0.0, 'max': 1.0}`
- Service proxy correlation: `{'action_24': -0.08521060321654683, 'action_32': None, 'dispatch': -0.43604495805209065}`

## Mapping Notes

- timing: request_datetime -> pickup_datetime wait proxy; pickup -> dropoff service proxy
- pressure: pickup-zone-hour and dispatching-base-hour counts normalized within deterministic first batch
- route: trip_miles/trip_time speed proxy for congestion and wait proxy for disruption

## Interpretation Limits

This is descriptive public replay. It does not contain behavior-policy propensities, unchosen alternatives, full 5PL state, or company rewards/costs. Therefore it does not prove counterfactual superiority, secondary-fleet economics, reorder safety, or company-data validation.
