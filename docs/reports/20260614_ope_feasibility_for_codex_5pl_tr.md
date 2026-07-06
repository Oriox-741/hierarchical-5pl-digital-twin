# OPE Feasibility for CODEX-5PL - 2026-06-14

## Classification

`OPE_FEASIBILITY_PARTIAL_PUBLIC_REPLAY_ONLY`

## Executive Summary

Current public datasets support descriptive historical replay, not defensible OPE. The production policy can be loaded read-only and evaluated on CODEX-like public proxy states, but public rows do not contain behavior-policy propensities, unchosen alternatives, full 5PL state transitions, or reward/cost trajectories.

## Tool Feasibility

| Tool | Feasibility Now | Future Company-Data Use |
|---|---|---|
| OBP | Not valid for current public rows | Contextual bandit OPE for one-step dispatch/route/fleet slices if pscore/reward logged |
| SCOPE-RL | Not valid for current public rows | Sequential trajectory OPE/OPS if `s,a,r,s_next,done` and coverage exist |
| d3rlpy FQE | Not valid as public-data value estimate | Offline policy selection for DQN-style candidates if trajectories and rewards exist |
| D4RL-style | Format/protocol only | Useful for versioned CODEX-5PL replay dataset packaging |

## Public Probe Evidence

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

## Required Fields for Future OPE

- behavior-policy action probability / pscore
- evaluation policy action distribution
- logged action id and action alternatives
- reward/cost at each decision
- next-state and terminal flags
- route/fleet/inventory/cost state at decision time
- adequate support for action 24/32 and close alternatives

## Conclusion

Public historical replay is ready as a descriptive diagnostic layer. OPE remains blocked on approved company data with propensities or reconstructable behavior-policy support.
