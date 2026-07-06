# Historical Replay Results and Scorecard Impact - 2026-06-14

## Classification

`HISTORICAL_REPLAY_ADAPTER_PARTIALLY_READY_WITH_LIMITATIONS`

## What Ran

- Historical replay/OPE tooling research.
- Public replay adapter spec and implementation.
- LaDe bounded replay probe: `5000` rows.
- NYC HVFHS bounded replay probe: `10000` rows.
- Olist bounded replay probe: `5000` rows.
- Production policy read-only load: successful for all three model probes.

## Probe Summary

```json
{
  "LaDe": {
    "row_count": 5000,
    "prediction_count": 5000,
    "mean_missingness_rate": 0.30445205479452053,
    "confidence_distribution": {
      "high": 5000
    },
    "action_distribution_top": [
      [
        "1",
        4451
      ],
      [
        "0",
        534
      ],
      [
        "45",
        8
      ],
      [
        "41",
        7
      ]
    ],
    "action_24_rate": 0.0,
    "action_32_rate": 0.0,
    "dispatch_rate": 0.003,
    "hold_rate": 0.997,
    "route_distribution": {
      "high_resilience": 15,
      "shortest": 4985
    },
    "mode_distribution": {
      "primary_fleet": 8,
      "secondary_fleet": 4992
    },
    "reorder_distribution": {
      "conservative": 4466,
      "none": 534
    },
    "service_proxy_correlation": {
      "action_24": null,
      "action_32": null,
      "dispatch": -0.11175930352424639
    }
  },
  "NYC_HVFHS": {
    "row_count": 10000,
    "prediction_count": 10000,
    "mean_missingness_rate": 0.3013698630136986,
    "confidence_distribution": {
      "high": 10000
    },
    "action_distribution_top": [
      [
        "1",
        7441
      ],
      [
        "0",
        1958
      ],
      [
        "25",
        217
      ],
      [
        "41",
        148
      ],
      [
        "2",
        135
      ],
      [
        "29",
        68
      ],
      [
        "24",
        20
      ],
      [
        "40",
        7
      ],
      [
        "43",
        3
      ],
      [
        "45",
        3
      ]
    ],
    "action_24_rate": 0.002,
    "action_32_rate": 0.0,
    "dispatch_rate": 0.0466,
    "hold_rate": 0.9534,
    "route_distribution": {
      "high_resilience": 161,
      "shortest": 9839
    },
    "mode_distribution": {
      "primary_fleet": 71,
      "secondary_fleet": 9929
    },
    "reorder_distribution": {
      "aggressive": 135,
      "conservative": 7877,
      "emergency": 3,
      "none": 1985
    },
    "service_proxy_correlation": {
      "action_24": -0.08521060321654683,
      "action_32": null,
      "dispatch": -0.43604495805209065
    }
  },
  "Olist": {
    "row_count": 5000,
    "prediction_count": 5000,
    "mean_missingness_rate": 0.5497534246575342,
    "confidence_distribution": {
      "medium": 5000
    },
    "action_distribution_top": [
      [
        "1",
        2748
      ],
      [
        "0",
        1956
      ],
      [
        "43",
        87
      ],
      [
        "3",
        66
      ],
      [
        "2",
        40
      ],
      [
        "45",
        28
      ],
      [
        "27",
        25
      ],
      [
        "29",
        15
      ],
      [
        "24",
        9
      ],
      [
        "41",
        8
      ]
    ],
    "action_24_rate": 0.0018,
    "action_32_rate": 0.0,
    "dispatch_rate": 0.038,
    "hold_rate": 0.962,
    "route_distribution": {
      "high_resilience": 130,
      "low_congestion": 4,
      "shortest": 4866
    },
    "mode_distribution": {
      "primary_fleet": 60,
      "secondary_fleet": 4940
    },
    "reorder_distribution": {
      "aggressive": 41,
      "conservative": 2800,
      "emergency": 188,
      "none": 1971
    },
    "service_proxy_correlation": {
      "action_24": -0.16634947290063917,
      "action_32": null,
      "dispatch": -0.5977567699325136
    }
  }
}
```

## Interpretation

The adapter successfully turns public historical logistics rows into CODEX-like 73-dimensional proxy states with explicit missingness and confidence. The production model can be run read-only on those states. The observed action distributions are descriptive action tendencies, not counterfactual proof.

The public probes do not reproduce the exact internal stress contexts where action 24/32 residual watches were seen. Therefore low action 24/32 rates in public replay should be treated as a coverage and domain-gap finding, not as evidence that the internal residual watches are wrong.

## Causal Limitations

Current public data lacks:

- logged CODEX-compatible behavior actions
- behavior propensities / action probabilities
- unchosen alternatives
- full state transitions and terminal flags
- company reward/cost definitions
- secondary fleet and reorder economics

## Score Impact Proposal

| Dimension | Prior | Proposed | Reason |
|---|---:|---:|---|
| Company-data validation readiness | 4.2 | 4.35 | Adapter and bridge define the executable path, but no company data was ingested |
| SOTA pathway maturity | 4.3 | 4.4 | Public model-vs-proxy replay layer improves benchmark structure |
| Thesis/advisor defensibility | 4.5 | 4.6 | Honest replay/OPE distinction strengthens the story |
| Real-world deployment readiness | unchanged | unchanged | No live telemetry, company data, or causal OPE |

## No-Overclaim Warning

This package does not support claims of global SOTA, real-world optimality, action 24/32 counterfactual superiority, secondary-fleet economic optimality, no-reorder safety, or company-data validation.

Machine-readable synthesis:

`reports/benchmarks/historical_replay_20260614/final_synthesis/historical_replay_summary.json`
