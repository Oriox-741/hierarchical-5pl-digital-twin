# Olist Large Public Replay Report - 2026-06-14

## Classification

`PUBLIC_LARGE_REPLAY_READY_WITH_CAUSAL_LIMITATIONS`

## Source and Sample

- Source: `<PROJECT_ROOT>\data\public\olist_20260614`
- Target rows: `all_feasible_delivered_rows`
- Actual replay rows: `96476`
- Rows skipped: `{'not_feasible_delivered_rows': 2965}`
- Runtime seconds: `105.839`
- Production policy loaded read-only: `True`

## Coverage and Missingness

- Mean missingness: `0.5479511690607132`
- Confidence distribution: `{'medium': 96476}`
- Feature status counts: `{'observed': 1640078, 'imputed': 1447112, 'unavailable': 3859082, 'neutral': 96476}`

## Action Tendencies

- Prediction count: `96476`
- Dispatch rate: `0.03643393175504789`
- Hold rate: `0.9635660682449522`
- Action 24 rate: `0.002062689166217505`
- Action 32 rate: `0.0`
- Top actions: `[('1', 52201), ('0', 38699), ('43', 1661), ('3', 1190), ('2', 871), ('27', 643), ('45', 578), ('41', 201), ('24', 199), ('28', 103)]`
- Route distribution: `{'shortest': 93952, 'high_resilience': 2523, 'low_congestion': 1}`
- Mode distribution: `{'secondary_fleet': 95686, 'primary_fleet': 790}`
- Reorder distribution: `{'none': 39001, 'conservative': 53017, 'emergency': 3586, 'aggressive': 872}`

## Service Proxy

- Service proxy summary: `{'count': 96476, 'mean': 0.9365337555409797, 'min': 0.0, 'max': 1.0}`
- Service proxy correlation: `{'action_24': -0.1734050827586187, 'action_32': None, 'dispatch': -0.6505362052667408}`

## Interpretation

This large replay is descriptive public proxy evidence. It does not include behavior propensities, unchosen alternatives, full 5PL trajectories, or company reward/cost definitions; it therefore does not prove counterfactual superiority or causal OPE.
