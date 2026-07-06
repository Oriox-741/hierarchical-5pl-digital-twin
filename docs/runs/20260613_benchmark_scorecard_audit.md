# 2026-06-13 Benchmark Scorecard Audit

## Scope

Objective: create the final benchmark interpretation and scorecard for CODEX PROJE without rerunning benchmarks, training, eval, gates, downloads, or mutating protected artifacts.

Written outputs:

- `docs/reports/20260613_benchmark_scorecard_for_advisor_tr.md`
- `docs/reports/20260613_benchmark_sonuclari_basit_yorum_tr.md`
- `docs/runs/20260613_benchmark_scorecard_audit.md`

Dashboard was intentionally not updated because the requested write scope was limited to the three scorecard/audit files.

## Evidence Read

- `docs/runs/20260613_full_benchmark_completion_sprint_audit.md`
- `docs/reports/20260613_full_benchmark_statistical_synthesis_tr.md`
- `docs/reports/20260613_full_benchmark_results_for_advisor_tr.md`
- `reports/benchmarks/full_completion_20260613/statistical_synthesis/final_benchmark_statistical_synthesis.json`
- `reports/benchmarks/full_completion_20260613/rule_based_full_20ep/rule_based_full_report.json`
- `reports/benchmarks/full_completion_20260613/amazon_full_route_proxy/amazon_full_route_proxy_report.json`
- `reports/benchmarks/full_completion_20260613/ortools_route_benchmarks/ortools_route_benchmark_report.json`
- `reports/benchmarks/full_completion_20260613/multiseed_robustness/multiseed_summary.json`
- `reports/benchmarks/full_completion_20260613/runtime_latency_repeated/runtime_latency_repeated_report.json`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`

## Key Evidence Used

Full sprint status:

- Final sprint classification: `FULL_BENCHMARK_COMPLETION_READY`.
- No blocked benchmark in statistical synthesis.
- Protected no-mutation proof in the sprint audit covered registry, production, baselines, DB, checkpoints, eval outputs, and source production checkpoint.

Equal-budget old production comparison:

- Old production and hierarchical production both evaluated with 8 scenarios, 20 episodes per scenario, seed 42, deterministic CPU.
- Both had 8/8 PASS and hard blockers zero.
- Hiyerarsik model improved service in 7/8 scenarios.
- Route disruption service: 0.904 -> 0.901, delta -0.004, sign-flip p=0.760, bootstrap 95% CI [-0.0271, 0.0183].
- Mixed stress service: 0.940 -> 0.942, delta +0.003, bootstrap 95% CI [-0.0022, 0.0077].

Rule-based full benchmark:

- Decision: `RULE_BASED_FULL_BENCHMARK_READY`.
- 8 baselines x 8 scenarios x 20 episodes = 1280 episode rows.
- Hard blocker total: 0.
- Mean service across baseline rollups: 0.9878660922751054.
- Worst service: 0.9463453904708932.
- Mean dispatch success per attempt: 0.9995717262720939.
- Caveat: neutral continuous control; tactical comparator only.

Amazon full route proxy:

- Decision: `AMAZON_FULL_ROUTE_PROXY_READY`.
- 6112 routes, 1,457,175 packages.
- Actual-to-greedy travel-time ratio mean: 0.9852256233910107, sd: 0.06951873141505499.
- Mean-pair travel-time asymmetry mean: 0.09997917186593447.
- Route score distribution: High 2718, Medium 3292, Low 102.
- Tight <=2h window package count: 0.
- Caveat: route-side proxy only.

OR-Tools route benchmark:

- Decision: `OR_TOOLS_ROUTE_BENCHMARK_READY`.
- 14 route-only instances: CVRPLIB 5, Homberger 3, Solomon 6.
- Success counts: CVRPLIB 5/5, Homberger 2/3, Solomon 4/6.
- 5 second time limit per instance.
- Caveat: route-only classical optimization, not full 5PL digital twin.

Multi-seed robustness:

- Decision: `MULTISEED_ROBUSTNESS_READY`.
- Seeds [42, 43, 44, 45, 46], 800 episode rows.
- Hard blocker total: 0.
- Seed42 consistency max_abs_service_delta: 0.0.
- Route disruption service mean: 0.9002641517969256.
- Mixed stress service mean: 0.9422679371457201.
- Mixed action24 total: 16704, action24 no-current 0.
- Route action32 total: 13806, action32 no-current 0.

Runtime latency:

- Decision: `RUNTIME_LATENCY_REPEATED_BENCHMARK_READY`.
- 5 x 1000 CPU predictions.
- Mean prediction latency mean: 0.8725571602350101 ms.
- p95 mean: 1.2434809911064801 ms.
- p99 mean: 1.565499596763402 ms.
- Max observed: 16.204999992623925 ms.

Historical ablation synthesis:

- Decision: `HISTORICAL_ABLATION_SYNTHESIS_READY`.
- 10 historical branches/axes.
- Main interpretation: exact resume fixed a real continuation bug; decisive product-quality improvement came from hierarchical_v1 action factorization plus flat-teacher distillation and exact-resume laddering.

## Scorecard Requirements Checklist

- Overall simulator-production readiness score included: yes, 92/100.
- Old-production comparison score included: yes, 88/100.
- Multi-seed robustness score included: yes, 91/100.
- Runtime latency score included: yes, 96/100.
- Amazon route-side external plausibility score included: yes, 82/100.
- OR-Tools route benchmark coverage score included: yes, 72/100.
- Rule-based baseline benchmark score included: yes, 74/100.
- Real-world deployment readiness score included: yes, 48/100.
- Thesis-defense readiness score included: yes, 86/100.
- Scores identified as subjective interpretation, not official metrics: yes.

Per-benchmark required fields:

- What was tested: yes.
- Sample size: yes.
- Key numeric results: yes.
- What result means: yes.
- What it proves: yes.
- What it does not prove: yes.
- 100-point interpretation: yes.
- Advisor-safe sentence: yes.
- Overclaim warning: yes.

Required Turkish interpretation sections:

- `Çok iyi olduğumuz alanlar`: yes.
- `Orta olduğumuz alanlar`: yes.
- `Hâlâ zayıf/kanıtlanmamış alanlar`: yes.
- `Diğer araştırmalarla neden birebir karşılaştırılamaz?`: yes.
- `Bu proje klasik VRP benchmark projesi mi, yoksa daha geniş 5PL decision-control sistemi mi?`: yes.
- `Danismana tek cumleyle benchmark sonucu nasil anlatilir?`: yes.

## No-Rerun / No-Mutation Proof

Actions performed for this scorecard:

- Read existing benchmark reports and summaries.
- Created three documentation files listed above.
- Did not train.
- Did not run offline eval.
- Did not run long-run gate.
- Did not rerun benchmarks.
- Did not download datasets.
- Did not install dependencies.
- Did not mutate registry, production, baselines, DB, checkpoints, or existing eval outputs.
- Did not overwrite benchmark outputs.

## Verification

Verification results:

- File existence: all three requested files exist.
  - `docs/reports/20260613_benchmark_scorecard_for_advisor_tr.md`
  - `docs/reports/20260613_benchmark_sonuclari_basit_yorum_tr.md`
  - `docs/runs/20260613_benchmark_scorecard_audit.md`
- Required score labels present: yes.
- Required Turkish interpretation headings present using encoding-safe Unicode escape check: yes.
- No train/eval/gate/AWS process found by process scan: yes.
- `git status` verification was attempted but `git` is not installed in this environment (`git : The term 'git' is not recognized...`).
- Protected artifact hash/read-only spot check completed:
  - `models/registry/active_models.json`: `b7c6e79749044b92639b8a27ef38483022fc21d9f0ee08743af88725e42aa60a`
  - `models/registry/models.jsonl`: `935d8bf34adc8dd956a68fb57581ec860b02d74ec26434dd6c084e63055c38e1`
  - `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: `fdaf8de495d90bdfd69bcc2ee3772eb6a09661e104a5127ea3cb01c6d02cf99a`
  - `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: `c1e756bdf7cd6b824ca134dbe6fb021612367ab18b12f2a24e13bf2167e51cde`
  - `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7a6e4eaac7cb9cdbf5926a544cf1cab7812940586c26e4c01cc757f3fd81b52e`
- Independent read-only reviewer: pending.

## Independent Reviewer

Reviewer verdict: `BENCHMARK_SCORECARD_APPROVED`.

Allowed verdicts:

- `BENCHMARK_SCORECARD_APPROVED`
- `BENCHMARK_SCORECARD_NEEDS_FIXES`
- `BENCHMARK_SCORECARD_BLOCKED`

## Final Classification

`BENCHMARK_SCORECARD_READY`
