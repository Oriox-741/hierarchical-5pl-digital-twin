# LaDe Large Public Replay Report - 2026-06-14

## Classification

`PUBLIC_LARGE_REPLAY_READY_WITH_CAUSAL_LIMITATIONS`

## Source and Sample

- Source: `<PROJECT_ROOT>\data\public\lade_20260614\delivery_jl.parquet`
- Target rows: `all_available`
- Actual replay rows: `31415`
- Rows skipped: `{}`
- Runtime seconds: `44.347`
- Production policy loaded read-only: `True`

## Coverage and Missingness

- Mean missingness: `0.3051177454274308`
- Confidence distribution: `{'high': 31415}`
- Feature status counts: `{'observed': 720635, 'imputed': 841520, 'unavailable': 699725, 'neutral': 31415}`

## Action Tendencies

- Prediction count: `31415`
- Dispatch rate: `0.004106318637593506`
- Hold rate: `0.9958936813624065`
- Action 24 rate: `0.0`
- Action 32 rate: `0.0`
- Top actions: `[('1', 24687), ('0', 6589), ('41', 82), ('45', 47), ('2', 10)]`
- Route distribution: `{'shortest': 31286, 'high_resilience': 129}`
- Mode distribution: `{'secondary_fleet': 31368, 'primary_fleet': 47}`
- Reorder distribution: `{'conservative': 24816, 'none': 6589, 'aggressive': 10}`

## Service Proxy

- Service proxy summary: `{'count': 31415, 'mean': 0.5789574670473376, 'min': 0.0, 'max': 1.0}`
- Service proxy correlation: `{'action_24': None, 'action_32': None, 'dispatch': -0.14103583516622648}`

## Interpretation

This large replay is descriptive public proxy evidence. It does not include behavior propensities, unchosen alternatives, full 5PL trajectories, or company reward/cost definitions; it therefore does not prove counterfactual superiority or causal OPE.
