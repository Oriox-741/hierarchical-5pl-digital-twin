# LaDe Public Replay Probe Report - 2026-06-14

## Classification

`PUBLIC_REPLAY_PROBE_READY_WITH_CAUSAL_LIMITATIONS`

## Source and Sample

- Source: `<PROJECT_ROOT>\data\public\lade_20260614\delivery_jl.parquet`
- Sample rows: `5000`
- Production policy loaded read-only: `True`
- No-policy dry-run row count: `5000`

## Replay Coverage

- Mean missingness: `0.30445205479452053`
- Confidence distribution: `{'high': 5000}`
- Aggregate feature status counts: `{'observed': 114750, 'imputed': 134125, 'unavailable': 111125, 'neutral': 5000}`

## Model Action Tendencies

- Prediction count: `5000`
- Dispatch rate: `0.003`
- Hold rate: `0.997`
- Action 24 rate: `0.0`
- Action 32 rate: `0.0`
- Top actions: `[('1', 4451), ('0', 534), ('45', 8), ('41', 7)]`
- Route distribution: `{'shortest': 4985, 'high_resilience': 15}`
- Mode distribution: `{'secondary_fleet': 4992, 'primary_fleet': 8}`
- Reorder distribution: `{'conservative': 4466, 'none': 534}`

## Proxy Timing / Correlation

- Service proxy summary: `{'count': 5000, 'mean': 0.54017304964539, 'min': 0.0, 'max': 0.9976359338061466}`
- Service proxy correlation: `{'action_24': None, 'action_32': None, 'dispatch': -0.11175930352424639}`

## Mapping Notes

- timing: accept_time -> delivery_time service proxy; accept_gps_time used as wait proxy when present
- pressure: AOI-hour demand and courier-day workload normalized within sample
- route: GPS accept-to-delivery haversine distance and high-latency proxy

## Interpretation Limits

This is descriptive public replay. It does not contain behavior-policy propensities, unchosen alternatives, full 5PL state, or company rewards/costs. Therefore it does not prove counterfactual superiority, secondary-fleet economics, reorder safety, or company-data validation.
