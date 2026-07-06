# Scorecard Benchmark Reproducibility Guide

## Decision

`SCORECARD_BENCHMARK_REPRODUCIBILITY_GUIDE_READY`

## Scope

This guide reproduces the scorecard-upgrade evidence under:

- `reports/benchmarks/scorecard_upgrade_20260613/`
- `reports/ops/20260613_scorecard_upgrade_preflight_ops_bundle_report.json`

It does not train, run project offline eval, run a long-run gate, mutate registry, mutate production, mutate baselines, mutate DB, or mutate checkpoints.

## Runtime Fingerprint

- Python: `3.12.9`
- PyVRP: `0.13.4`
- `gym-invmgmt`: `0.2.1`
- FleetPy: `0.2.3`

Full compact manifest:

- `reports/benchmarks/scorecard_upgrade_20260613/reproducibility_manifest.json`

## Reproduction Commands

Preflight protected-state bundle:

```powershell
python scripts/run_production_ops_bundle.py --output reports/ops/20260613_scorecard_upgrade_preflight_ops_bundle_report.json
```

Extended PyVRP route benchmark:

```powershell
python scripts/pyvrp_route_reference_benchmark.py --data-dir data/public/route_benchmarks_20260613 --output-dir reports/benchmarks/scorecard_upgrade_20260613/route_extended --time-limit-seconds 2 --max-cvrp 10 --max-vrptw 6
```

Focused route tests:

```powershell
python -m unittest tests.benchmarks.test_pyvrp_route_reference_benchmark -v
```

## Evidence Files

| Track | Evidence |
| --- | --- |
| Rule continuous diagnostics | `reports/benchmarks/scorecard_upgrade_20260613/rule_based_continuous_diagnostics/rule_continuous_diagnostics.json` |
| Inventory/reorder extended | `reports/benchmarks/scorecard_upgrade_20260613/inventory_reorder_extended/inventory_reorder_extended_report.json` |
| Fleet/dispatch extension | `reports/benchmarks/scorecard_upgrade_20260613/fleet_dispatch_extended/fleet_dispatch_extended_report.json` |
| Route extended | `reports/benchmarks/scorecard_upgrade_20260613/route_extended/pyvrp_benchmark_report.json` |
| Route gap caveat | `reports/benchmarks/scorecard_upgrade_20260613/route_extended/route_gap_summary.json` |

## Guardrails

Do not overwrite existing benchmark output directories. Use a new dated output root for reruns. Do not use these route/inventory/fleet analog benchmarks as proof of global SOTA or real-world deployment optimality.

