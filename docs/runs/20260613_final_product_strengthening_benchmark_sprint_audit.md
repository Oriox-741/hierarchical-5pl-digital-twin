# Final Product Strengthening and Benchmark Sprint Audit

Date: 2026-06-13

Final classification: `FINAL_PRODUCT_STRENGTHENING_BENCHMARK_SPRINT_READY`

## Scope

This sprint strengthened the already-promoted hierarchical v1 1M production model without training, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, or editing existing eval outputs.

Production identity preserved:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- production checkpoint: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- runtime: `torch_joint`
- DQN architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation/action: `73 / 48`

## Files Changed

Implementation and tests:

- `src/eval/rule_based_baseline_arena.py`
- `tests/eval/test_rule_based_baseline_arena.py`
- `scripts/benchmark_runtime_latency.py`
- `tests/orchestration/test_runtime_latency_benchmark.py`
- `scripts/ortools_vrptw_smoke_benchmark.py`
- `tests/benchmarks/__init__.py`
- `tests/benchmarks/test_ortools_vrptw_smoke_benchmark.py`

Reports and docs:

- `reports/ops/20260613_pre_benchmark_ops_bundle_report.json`
- `reports/ops/20260613_post_benchmark_ops_bundle_report.json`
- `reports/ops/20260613_post_reviewfix_ops_bundle_report.json`
- `reports/ops/20260613_final_ops_bundle_report.json`
- `reports/benchmarks/rule_based_baseline_20260613_bounded_3ep/`
- `reports/benchmarks/runtime_latency_hierarchical_v1_20260613.json`
- `reports/benchmarks/amazon_route_proxy_20260613/`
- `reports/benchmarks/ortools_vrptw_smoke_20260613.json`
- `docs/reports/20260613_benchmark_results_and_strengthening_summary_tr.md`
- `docs/reports/20260613_final_advisor_showcase_pack_tr.md`
- `docs/reports/20260613_sunum_slayt_icerik_taslagi_tr.md`
- `docs/reports/20260613_demo_script_and_qna_tr.md`
- `docs/reports/20260613_benchmark_status_and_next_steps_tr.md`
- `docs/thesis/20260613_lisans_tezi_ilk_taslak_tr.md`
- `docs/00_PROJECT_DASHBOARD.md`
- `docs/runs/20260613_final_product_strengthening_benchmark_sprint_audit.md`

## Dependency Note

OR-Tools was not installed initially. The sprint installed:

- `ortools==9.15.6755`
- `absl-py==2.4.0`
- `immutabledict==4.3.1`
- `protobuf==6.33.6`

The install replaced the pre-existing global Python `protobuf 7.34.1` package with `protobuf 6.33.6` as required by OR-Tools. No repository protected artifact was mutated by this dependency install.

## Benchmark Results

### Rule-Based Baseline

Output:

`reports/benchmarks/rule_based_baseline_20260613_bounded_3ep/`

Decision: `RULE_BASED_BASELINE_BENCHMARK_READY`

Kapsam:

- 8 scenarios.
- 8 deterministic baselines.
- 3 episodes per scenario.
- 64 baseline-scenario summaries.
- fatal hard-blocker totals: `0`.

The first attempted full run, 20 episodes x 8 scenarios x 8 baselines, exceeded the 30-minute command ceiling before writing output. The empty fresh directory was removed after verifying it was empty. The completed sprint result is explicitly bounded, not a full equal-budget rule baseline benchmark.

### Runtime Latency

Output:

`reports/benchmarks/runtime_latency_hierarchical_v1_20260613.json`

Decision: `RUNTIME_LATENCY_BENCHMARK_READY`

CPU deterministic prediction stats over 1000 iterations:

- min: `0.7321 ms`
- mean: `1.1090 ms`
- p50: `1.0288 ms`
- p95: `1.6751 ms`
- p99: `1.9568 ms`
- max: `5.0587 ms`

All predictions validated:

- continuous action length `5`
- continuous action finite and bounded in `[-1, 1]`
- discrete action in `0..47`

### Amazon Route Proxy

Output:

`reports/benchmarks/amazon_route_proxy_20260613/`

Decision: `AMAZON_ROUTE_PROXY_BOUNDED_SAMPLE_READY`

Data scope:

- Existing approved local sample: `data/public/almrrc2021_small/`
- No new public data download in this sprint.
- AWS S3 listing only was run.
- Full public listing remains 42 objects / 3.1 GiB.
- Full training travel-times is 1.7 GiB and should use a streaming parser before any full analysis.

Bounded sample metrics:

- route count: `13`
- actual/greedy travel-time ratio mean: `0.9673`
- actual/greedy travel-time ratio median: `0.9728`
- directed travel-time asymmetry mean: `0.0979`
- directed travel-time asymmetry median: `0.0920`
- 2-hour tight-window share: `0.0`

Interpretation:

- Supports route-side plausibility for action 24 shortest/fastest-like routing.
- Supports route-side plausibility for action 32 low-congestion/reliability proxy through directed travel-time asymmetry.
- Does not validate secondary fleet, reorder none, company costs, carrier economics, or dispatch failure semantics.

### OR-Tools VRPTW Smoke

Output:

`reports/benchmarks/ortools_vrptw_smoke_20260613.json`

Decision: `ORTOOLS_VRPTW_SMOKE_BENCHMARK_READY`

Smoke result:

- OR-Tools version: `9.15.6755`
- status: `FEASIBLE`
- objective: `24`
- route: `[0, 5, 4, 3, 2, 1, 0]`
- served nodes: `[1, 2, 3, 4, 5]`

This is a classical optimizer smoke test only. It is not a full production benchmark for the 5PL digital twin.

## Ops Bundle Results

- Pre-benchmark ops bundle: `OPS_BUNDLE_CLEAN`
- Post-benchmark ops bundle: `OPS_BUNDLE_CLEAN`
- Post-review-fix ops bundle: `OPS_BUNDLE_CLEAN`
- Final ops bundle after dashboard/audit links: `OPS_BUNDLE_CLEAN`

Latest output:

`reports/ops/20260613_final_ops_bundle_report.json`

## Review Trail

Independent read-only reviewer: `gpt-5.5`

Initial verdict: `FINAL_PRODUCT_STRENGTHENING_REVIEW_NEEDS_FIXES`

Findings fixed:

- absolute protected paths could bypass relative prefix guards.
- direct rule benchmark writer could reuse an existing output directory and overwrite report files.

Fix:

- all new benchmark output guards now resolve candidate paths against repo-root protected directories.
- `write_rule_benchmark_outputs()` now owns fresh output directory creation and refuses existing dirs.
- regression tests were added for absolute protected paths and existing rule output dirs.

Final reviewer verdict: `FINAL_PRODUCT_STRENGTHENING_REVIEW_APPROVED`

## Verification

Focused benchmark/public-route tests:

```powershell
python -m unittest tests.eval.test_rule_based_baselines tests.eval.test_rule_based_baseline_arena tests.orchestration.test_runtime_latency_benchmark tests.benchmarks.test_ortools_vrptw_smoke_benchmark tests.real_world_calibration.test_route_proxy_metrics -v
```

Result: `38 tests OK`

Ops/tooling tests:

```powershell
python -m unittest tests.orchestration.test_production_artifact_health_report tests.orchestration.test_monitoring_report_generator tests.orchestration.test_company_data_intake_validator tests.orchestration.test_production_ops_bundle -v
```

Result: `47 tests OK`

Py compile:

```powershell
python -m py_compile src/eval/rule_based_baseline_arena.py scripts/benchmark_runtime_latency.py scripts/ortools_vrptw_smoke_benchmark.py tests/eval/test_rule_based_baseline_arena.py tests/orchestration/test_runtime_latency_benchmark.py tests/benchmarks/test_ortools_vrptw_smoke_benchmark.py
```

Result: PASS

## Protected No-Mutation Proof

File hashes match preflight:

| Path | SHA256 |
| --- | --- |
| `models/registry/active_models.json` | `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A` |
| `models/registry/models.jsonl` | `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` |

Directory profiles match preflight:

| Path | Files | Bytes |
| --- | ---: | ---: |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/production` | `8` | `328265276` |
| `models/eval` | `128` | `388267659` |

Process scan:

- no train process found.
- no offline eval process found.
- no long-run gate process found.
- no AWS/download process found at final scan.

Generated Python cache directories under `src`, `scripts`, and `tests` were removed after verification.

## Final Notes

- No training was run.
- No offline eval or long-run gate was run.
- No candidate registration, active activation, production promotion, baseline update, DB migration, or checkpoint mutation was run.
- Existing `models/eval` outputs were not edited.
- Existing production artifacts remain byte-identical.

Final classification: `FINAL_PRODUCT_STRENGTHENING_BENCHMARK_SPRINT_READY`
