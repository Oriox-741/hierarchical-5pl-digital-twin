# Lisans Tezi Benchmark Bolumu - 2026-06-14

## Classification

`FINAL_THESIS_BENCHMARK_SECTION_READY`

## Benchmark Families

- Internal 5PL simulator gates and equal-budget production comparison.
- Rule-based and continuous-aware simulator baselines.
- Route references: PyVRP, OR-Tools, Amazon route proxy, SVRP geometry/dynamic appear times.
- Inventory references: gym-invmgmt and MABIM/ReplenishmentEnv.
- Fleet/dispatch references: HVFHV, FleetPy no-Gurobi RPP/IRS, SF taxi, NYC TLC.
- Historical replay v0: public rows converted to CODEX-like proxy states.

```json
{
  "LaDe": {
    "row_count": 31415,
    "prediction_count": 31415,
    "mean_missingness_rate": 0.3051177454274308,
    "confidence_distribution": {
      "high": 31415
    },
    "action_24_rate": 0.0,
    "action_32_rate": 0.0,
    "dispatch_rate": 0.004106318637593506,
    "hold_rate": 0.9958936813624065,
    "top_actions": [
      [
        "1",
        24687
      ],
      [
        "0",
        6589
      ],
      [
        "41",
        82
      ],
      [
        "45",
        47
      ],
      [
        "2",
        10
      ]
    ],
    "route_distribution": {
      "high_resilience": 129,
      "shortest": 31286
    },
    "mode_distribution": {
      "primary_fleet": 47,
      "secondary_fleet": 31368
    },
    "reorder_distribution": {
      "aggressive": 10,
      "conservative": 24816,
      "none": 6589
    },
    "service_proxy_correlation": {
      "action_24": null,
      "action_32": null,
      "dispatch": -0.14103583516622648
    }
  },
  "NYC_HVFHS": {
    "row_count": 100000,
    "prediction_count": 100000,
    "mean_missingness_rate": 0.3013698630136986,
    "confidence_distribution": {
      "high": 100000
    },
    "action_24_rate": 0.00051,
    "action_32_rate": 2e-05,
    "dispatch_rate": 0.04092,
    "hold_rate": 0.95908,
    "top_actions": [
      [
        "1",
        79220
      ],
      [
        "0",
        13929
      ],
      [
        "2",
        2754
      ],
      [
        "41",
        2510
      ],
      [
        "25",
        1007
      ],
      [
        "29",
        281
      ],
      [
        "45",
        157
      ],
      [
        "40",
        55
      ]
    ],
    "route_distribution": {
      "high_resilience": 2738,
      "low_congestion": 2,
      "shortest": 97260
    },
    "mode_distribution": {
      "primary_fleet": 450,
      "secondary_fleet": 99550
    },
    "reorder_distribution": {
      "aggressive": 2756,
      "conservative": 83175,
      "emergency": 31,
      "none": 14038
    },
    "service_proxy_correlation": {
      "action_24": -0.03525765583751783,
      "action_32": -0.007891244936826736,
      "dispatch": -0.4610341719753878
    }
  },
  "Olist": {
    "row_count": 96476,
    "prediction_count": 96476,
    "mean_missingness_rate": 0.5479511690607132,
    "confidence_distribution": {
      "medium": 96476
    },
    "action_24_rate": 0.002062689166217505,
    "action_32_rate": 0.0,
    "dispatch_rate": 0.03643393175504789,
    "hold_rate": 0.9635660682449522,
    "top_actions": [
      [
        "1",
        52201
      ],
      [
        "0",
        38699
      ],
      [
        "43",
        1661
      ],
      [
        "3",
        1190
      ],
      [
        "2",
        871
      ],
      [
        "27",
        643
      ],
      [
        "45",
        578
      ],
      [
        "41",
        201
      ]
    ],
    "route_distribution": {
      "high_resilience": 2523,
      "low_congestion": 1,
      "shortest": 93952
    },
    "mode_distribution": {
      "primary_fleet": 790,
      "secondary_fleet": 95686
    },
    "reorder_distribution": {
      "aggressive": 872,
      "conservative": 53017,
      "emergency": 3586,
      "none": 39001
    },
    "service_proxy_correlation": {
      "action_24": -0.1734050827586187,
      "action_32": null,
      "dispatch": -0.6505362052667408
    }
  }
}
```

Simulator evidence tests the actual CODEX policy in the digital twin. Public evidence increases external plausibility and benchmark breadth. Company data is required for causal action-level validation.
