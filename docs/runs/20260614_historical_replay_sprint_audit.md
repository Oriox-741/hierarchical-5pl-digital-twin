# Historical Replay Sprint Audit - 2026-06-14

## Classification Before Final Review

`HISTORICAL_REPLAY_ADAPTER_PARTIALLY_READY_WITH_LIMITATIONS`

## Scope

This sprint researched historical replay/OPE tooling, implemented a public replay adapter, ran bounded public replay probes over existing public data, and wrote a company replay bridge and OPE feasibility report. It did not train, run project offline eval, run long-run gates, register, activate, promote, update baselines, mutate DB, mutate checkpoints, or overwrite existing eval outputs.

## Main Outputs

- `scripts/public_historical_replay_adapter.py`
- `tests/benchmarks/test_public_historical_replay_adapter.py`
- `docs/reports/20260614_historical_replay_and_ope_tooling_research_tr.md`
- `docs/reports/20260614_public_historical_replay_adapter_design_tr.md`
- `docs/runs/20260614_lade_public_replay_probe_report.md`
- `docs/runs/20260614_nyc_hvfhs_public_replay_probe_report.md`
- `docs/runs/20260614_olist_public_replay_probe_report.md`
- `docs/runbooks/20260614_company_replay_bridge_to_model_policy_spec.md`
- `docs/reports/20260614_ope_feasibility_for_codex_5pl_tr.md`
- `docs/reports/20260614_historical_replay_results_and_scorecard_impact_tr.md`
- `reports/benchmarks/historical_replay_20260614/final_synthesis/historical_replay_summary.json`

## Public Replay Results

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

## Protected No-Mutation Preflight

- Preflight protected snapshot: `reports/benchmarks/historical_replay_20260614/preflight_protected_state.json`
- Preflight ops bundle: `reports/ops/20260614_historical_replay_preflight_ops_bundle_report.json`
- Preflight ops bundle result: `OPS_BUNDLE_CLEAN`

## Verification

Focused tests:

```powershell
python -m unittest tests.benchmarks.test_public_historical_replay_adapter -v
```

Result: `OK` (7 tests).

Relevant benchmark/reporting tests:

```powershell
python -m unittest tests.benchmarks.test_public_historical_replay_adapter tests.benchmarks.test_company_data_replay_validator tests.benchmarks.test_public_data_expansion_benchmarks tests.benchmarks.test_fleet_dispatch_upgrade_benchmarks tests.orchestration.test_production_artifact_health_report tests.orchestration.test_monitoring_report_generator -v
```

Result: `OK` (46 tests).

Compile check:

```powershell
python -m py_compile scripts\public_historical_replay_adapter.py tests\benchmarks\test_public_historical_replay_adapter.py
```

Result: passed.

Final ops bundle:

```powershell
python scripts\run_production_ops_bundle.py --output reports\ops\20260614_historical_replay_final_ops_bundle_report.json
```

Result: `OPS_BUNDLE_CLEAN`.

Protected final audit:

- `reports/benchmarks/historical_replay_20260614/final_protected_state_audit.json`
- Result: `PROTECTED_STATE_CLEAN`

Independent review is still required before the final classification.
