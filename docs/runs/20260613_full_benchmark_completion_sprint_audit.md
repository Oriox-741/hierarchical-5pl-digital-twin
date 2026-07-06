# Full Benchmark Completion Sprint Audit

Final classification: `FULL_BENCHMARK_COMPLETION_READY`

## Scope

Completed the full benchmark sprint without model training, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, or existing eval-output overwrite. Fresh outputs were written under reports/data/docs/scripts/tests as allowed.

## Completion Matrix

| Task | Required target | Result | Output |
| --- | --- | --- | --- |
| Rule-based full 20ep | 8 baselines x 8 scenarios x 20 episodes | `RULE_BASED_FULL_BENCHMARK_READY`, 1280 rows | `reports/benchmarks/full_completion_20260613/rule_based_full_20ep/` |
| Amazon full route proxy | model_build training inputs, all available routes | `AMAZON_FULL_ROUTE_PROXY_READY`, 6112 routes, 1457175 packages | `reports/benchmarks/full_completion_20260613/amazon_full_route_proxy/` |
| OR-Tools route suite | >=6 Solomon, >=3 Homberger, >=5 CVRPLIB | `OR_TOOLS_ROUTE_BENCHMARK_READY`, {'cvrplib_cvrp': 5, 'homberger_200_vrptw': 3, 'solomon_vrptw': 6} | `reports/benchmarks/full_completion_20260613/ortools_route_benchmarks/` |
| Multi-seed robustness | seeds 42-46, 20 ep/scenario | `MULTISEED_ROBUSTNESS_READY`, 800 rows | `reports/benchmarks/full_completion_20260613/multiseed_robustness/` |
| Runtime latency repeated | 5 x 1000 CPU predictions | `RUNTIME_LATENCY_REPEATED_BENCHMARK_READY`, p99 mean 1.5655 ms | `reports/benchmarks/full_completion_20260613/runtime_latency_repeated/` |
| Historical ablation synthesis | existing artifacts only | `HISTORICAL_ABLATION_SYNTHESIS_READY`, 10 branches | `reports/benchmarks/full_completion_20260613/historical_ablation_synthesis/` |
| Statistical/advisor synthesis | final JSON + Turkish reports | `FULL_BENCHMARK_STATISTICAL_SYNTHESIS_READY` | `reports/benchmarks/full_completion_20260613/statistical_synthesis/` |

No benchmark section is blocked or downgraded to smoke/sample.

## Key Results

- Rule-based full: all eight rule policies completed full 20ep coverage; hard blockers total `0`. Caveat: neutral continuous PPO vector makes this a tactical comparator, and aggregate outputs were identical across rule families.
- Amazon full: actual/greedy ratio mean `0.9852`, mean-pair asymmetry `0.1000`, route scores `{'High': 2718, 'Low': 102, 'Medium': 3292}`.
- OR-Tools: success counts `{'cvrplib_cvrp': 5, 'homberger_200_vrptw': 2, 'solomon_vrptw': 4}` within 5s; several larger VRPTW instances timed out but the required suite was executed and reported.
- Multi-seed: hard blocker total `0`; route service mean `0.9003`; mixed service mean `0.9423`.
- Latency: mean prediction latency mean `0.8726` ms; p95 mean `1.2435` ms; p99 mean `1.5655` ms.

## Downloads And Dependencies

- Installed dependency: `ijson==3.5.0` for streaming Amazon JSON parsing.
- Reused dependency: `ortools==9.15.6755`.
- Amazon S3 full model-build inputs downloaded to `data/public/almrrc2021_full/`:
  - `actual_sequences`: 9665078 bytes, SHA256 `3de44067242e8fa841d4b852db8809afaa23fe852e452ee4a2f54f0f0fb77e3c`
  - `invalid_sequence_scores`: 414742 bytes, SHA256 `265c8603ea5e658bf733b5d32a6037415324f7edb8331499e7c51e0f8ababefa`
  - `package_data`: 375437806 bytes, SHA256 `9ac858358e9f43d34cb65198c03601c814328a06dec9b71ee88b40aaec7f0966`
  - `route_data`: 78972162 bytes, SHA256 `da3a5d4e73b683d756111f0d9e6d3f20eb8e82adf7c89ee6ee0cb2239abd2f38`
  - `travel_times`: 1817146363 bytes, SHA256 `0023d7517380f15b611d77bc0f8a41a1b89459c7c12c37e88981ea16a2b99e2d`
- OR-Tools public route benchmark data downloaded to `data/public/route_benchmarks_20260613/` from SINTEF and CVRPLIB/PUC-Rio sources.

## Commands Run

- `python scripts/run_production_ops_bundle.py --output reports/ops/20260613_full_benchmark_preflight_ops_bundle_report.json` -> `OPS_BUNDLE_CLEAN`
- `python scripts/full_completion_rule_based_benchmark.py ... --max-chunks N`, repeated until 64/64 chunks complete; then `--aggregate-only` -> `RULE_BASED_FULL_BENCHMARK_READY`
- `aws s3 cp --no-sign-request --only-show-errors ...` for required Amazon model-build files
- `python scripts/real_world_calibration/amazon_route_streaming_sampler.py --data-root data/public/almrrc2021_full --output-dir reports/benchmarks/full_completion_20260613/amazon_full_route_proxy` -> `AMAZON_FULL_ROUTE_PROXY_READY`
- `python scripts/ortools_route_benchmark_suite.py --output-dir reports/benchmarks/full_completion_20260613/ortools_route_benchmarks --data-dir data/public/route_benchmarks_20260613 --time-limit-seconds 5` -> `OR_TOOLS_ROUTE_BENCHMARK_READY`
- `python -m src.eval.evaluate_real_world_scenarios ... --seed 42/43/44/45/46 --episodes 20 --deterministic` -> all five fresh seed dirs written
- `python scripts/runtime_latency_repeated_benchmark.py --runs 5 --iterations 1000 ...` -> `RUNTIME_LATENCY_REPEATED_BENCHMARK_READY`
- `python scripts/run_production_ops_bundle.py --output reports/ops/20260613_full_benchmark_final_ops_bundle_report.json` -> `OPS_BUNDLE_CLEAN`

## Verification

- Focused unittest suite: 85 tests, PASS.
- Py compile touched Python files: PASS.
- Final ops bundle: `OPS_BUNDLE_CLEAN`.
- Process scan: no train/eval/gate/AWS process running after completion.
- Transient `__pycache__` dirs removed after verification.

## Independent Review

Reviewer verdict:

```text
FULL_BENCHMARK_COMPLETION_APPROVED
```

Reviewer scope was read-only and covered the full benchmark outputs, tests,
reports, final ops bundle evidence, and protected-state proof.

## Protected No-Mutation Proof

Postflight comparison against `reports/benchmarks/full_completion_20260613/preflight_protected_state.json`:

| Protected item | Match |
| --- | --- |
| `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt` | `True` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `True` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt` | `True` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `True` |
| `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json` | `True` |
| `models/registry/active_models.json` | `True` |
| `models/registry/models.jsonl` | `True` |
| `db` profile | `True` |
| `models/baselines` profile | `True` |
| `models/checkpoints` profile | `True` |
| `models/eval` profile | `True` |
| `models/production` profile | `True` |

## Output Index

- `docs/runs/20260613_rule_based_full_20ep_benchmark_report.md`
- `docs/runs/20260613_amazon_full_route_proxy_benchmark_report.md`
- `docs/runs/20260613_ortools_route_benchmark_suite_report.md`
- `docs/runs/20260613_multiseed_robustness_benchmark_report.md`
- `docs/runs/20260613_runtime_latency_repeated_benchmark_report.md`
- `docs/runs/20260613_historical_ablation_synthesis_report.md`
- `docs/reports/20260613_full_benchmark_statistical_synthesis_tr.md`
- `docs/reports/20260613_full_benchmark_results_for_advisor_tr.md`
- `reports/ops/20260613_full_benchmark_final_ops_bundle_report.json`

## Final Note

The evidence strengthens the simulator-production case for hierarchical v1 1M. It does not prove live TMS/WMS/ERP deployment, secondary_fleet economics, reorder_none economics, or universal real-world optimality without company data.
