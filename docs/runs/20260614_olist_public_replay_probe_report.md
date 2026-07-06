# Olist Public Replay Probe Report - 2026-06-14

## Classification

`PUBLIC_REPLAY_PROBE_READY_WITH_CAUSAL_LIMITATIONS`

## Source and Sample

- Source: `<PROJECT_ROOT>\data\public\olist_20260614`
- Sample rows: `5000`
- Production policy loaded read-only: `True`
- No-policy dry-run row count: `5000`

## Replay Coverage

- Mean missingness: `0.5497534246575342`
- Confidence distribution: `{'medium': 5000}`
- Aggregate feature status counts: `{'observed': 84746, 'imputed': 74594, 'unavailable': 200660, 'neutral': 5000}`

## Model Action Tendencies

- Prediction count: `5000`
- Dispatch rate: `0.038`
- Hold rate: `0.962`
- Action 24 rate: `0.0018`
- Action 32 rate: `0.0`
- Top actions: `[('1', 2748), ('0', 1956), ('43', 87), ('3', 66), ('2', 40), ('45', 28), ('27', 25), ('29', 15)]`
- Route distribution: `{'shortest': 4866, 'high_resilience': 130, 'low_congestion': 4}`
- Mode distribution: `{'secondary_fleet': 4940, 'primary_fleet': 60}`
- Reorder distribution: `{'none': 1971, 'conservative': 2800, 'emergency': 188, 'aggressive': 41}`

## Proxy Timing / Correlation

- Service proxy summary: `{'count': 5000, 'mean': 0.9395497932223309, 'min': 0.0, 'max': 1.0}`
- Service proxy correlation: `{'action_24': -0.16634947290063917, 'action_32': None, 'dispatch': -0.5977567699325136}`

## Mapping Notes

- timing: purchase -> delivered service proxy; purchase -> approved wait proxy; delivered -> estimate lateness proxy
- pressure: purchase-day order count normalized within sample
- route: seller/customer state only; route distance/congestion unavailable and not forced

## Interpretation Limits

This is descriptive public replay. It does not contain behavior-policy propensities, unchosen alternatives, full 5PL state, or company rewards/costs. Therefore it does not prove counterfactual superiority, secondary-fleet economics, reorder safety, or company-data validation.
