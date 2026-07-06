# NYC HVFHS Large Public Replay Report - 2026-06-14

## Classification

`PUBLIC_LARGE_REPLAY_READY_WITH_CAUSAL_LIMITATIONS`

## Source and Sample

- Source: `<PROJECT_ROOT>\data\public\nyc_hvfhs_20260614\fhvhv_tripdata_2023-01.parquet`
- Target rows: `100000`
- Actual replay rows: `100000`
- Rows skipped: `{}`
- Runtime seconds: `117.339`
- Production policy loaded read-only: `True`

## Coverage and Missingness

- Mean missingness: `0.3013698630136986`
- Confidence distribution: `{'high': 100000}`
- Feature status counts: `{'observed': 2300000, 'imputed': 2700000, 'unavailable': 2200000, 'neutral': 100000}`

## Action Tendencies

- Prediction count: `100000`
- Dispatch rate: `0.04092`
- Hold rate: `0.95908`
- Action 24 rate: `0.00051`
- Action 32 rate: `2e-05`
- Top actions: `[('1', 79220), ('0', 13929), ('2', 2754), ('41', 2510), ('25', 1007), ('29', 281), ('45', 157), ('40', 55), ('24', 51), ('43', 13)]`
- Route distribution: `{'shortest': 97260, 'high_resilience': 2738, 'low_congestion': 2}`
- Mode distribution: `{'secondary_fleet': 99550, 'primary_fleet': 450}`
- Reorder distribution: `{'none': 14038, 'conservative': 83175, 'aggressive': 2756, 'emergency': 31}`

## Service Proxy

- Service proxy summary: `{'count': 100000, 'mean': 0.5536019509981851, 'min': 0.0, 'max': 1.0}`
- Service proxy correlation: `{'action_24': -0.03525765583751783, 'action_32': -0.007891244936826736, 'dispatch': -0.4610341719753878}`

## Interpretation

This large replay is descriptive public proxy evidence. It does not include behavior propensities, unchosen alternatives, full 5PL trajectories, or company reward/cost definitions; it therefore does not prove counterfactual superiority or causal OPE.
