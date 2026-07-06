# MABIM Inventory Benchmark Report

Date: 2026-06-13

## Decision

`MABIM_INVENTORY_BENCHMARK_READY`

## Scope

Downloaded public source:

- `data/public/mabim_or_replenishment_20260613/source/ReplenishmentEnv-main`

Benchmark task:

- `sku50.single_store.standard`
- Wrapper: `OracleWrapper`
- Policies: 4
- Horizon observed: 60 steps

## Outputs

- JSON: `reports/benchmarks/sota_pathway_20260613/inventory_mabim/mabim_benchmark_report.json`
- CSV: `reports/benchmarks/sota_pathway_20260613/inventory_mabim/mabim_benchmark_summary.csv`

## Results

| Policy | Steps | Final balance | Mean balance | Min balance | Mean action |
| --- | ---: | ---: | ---: | ---: | ---: |
| `zero_reorder_stress_floor` | 60 | -6253.29 | 7908.73 | -6253.29 | 0.000 |
| `conservative_ss_builtin_style` | 60 | 112111.35 | 65783.24 | 4425.18 | 0.475 |
| `aggressive_ss_builtin_style` | 60 | 175450.25 | 97325.44 | 4415.18 | 0.665 |
| `project_inspired_continuous_reorder` | 60 | 126513.67 | 72307.96 | 4425.18 | 0.510 |

## Interpretation

MABIM ran successfully as a second public inventory benchmark. The no-reorder stress floor lost money by final balance, while built-in-style `(s,S)` policies and the project-inspired continuous reorder policy stayed positive.

This benchmark supports inventory/reorder plausibility only. It does not validate routing, dispatch, fleet mode, or full 5PL economics.

## Classification

`MABIM_INVENTORY_BENCHMARK_READY`
