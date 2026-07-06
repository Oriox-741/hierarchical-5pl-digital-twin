# Full Benchmark Statistical Synthesis TR

Siniflandirma: `FULL_BENCHMARK_STATISTICAL_SYNTHESIS_READY`

## Tamamlanan Benchmarklar

| Benchmark | Kapsam | Karar |
| --- | --- | --- |
| `rule_based_full_20ep` | 8 baselines x 8 scenarios x 20 episodes = 1280 rows | `RULE_BASED_FULL_BENCHMARK_READY` |
| `amazon_full_route_proxy` | 6112 routes, 1457175 packages | `AMAZON_FULL_ROUTE_PROXY_READY` |
| `ortools_route_benchmark_suite` | {'cvrplib_cvrp': 5, 'homberger_200_vrptw': 3, 'solomon_vrptw': 6} | `OR_TOOLS_ROUTE_BENCHMARK_READY` |
| `multiseed_robustness` | 800 episode rows, seeds [42, 43, 44, 45, 46] | `MULTISEED_ROBUSTNESS_READY` |
| `runtime_latency_repeated` | 5 x 1000 predictions | `RUNTIME_LATENCY_REPEATED_BENCHMARK_READY` |
| `historical_ablation_synthesis` | 10 historical branches/axes | `HISTORICAL_ABLATION_SYNTHESIS_READY` |

Engellenen benchmark yok.

## Ana Istatistikler

- Rule-based full: `1280` episode row, hard blocker toplam `0`.
- Amazon full route proxy: `6112` route, `1457175` package, actual/greedy ratio mean `0.9852`.
- Amazon travel-time asymmetry mean-pair mean `0.1000`.
- OR-Tools: `{'cvrplib_cvrp': 5, 'homberger_200_vrptw': 3, 'solomon_vrptw': 6}` instance; 5 saniye limitte basarili cozum `{'cvrplib_cvrp': 5, 'homberger_200_vrptw': 2, 'solomon_vrptw': 4}`.
- Multi-seed: `800` episode row, hard blocker toplam `0`, seed42 onceki equal-budget ile service delta max `0.0`.
- Runtime latency: mean `0.8726 ms`, p95 `1.2435 ms`, p99 `1.5655 ms`.

## Genel Yorum

The hierarchical_v1 1M production model remains simulator-production-validated under the project gate suite, with new full benchmark support from rule baselines, multi-seed robustness, full Amazon route-side proxy analysis, expanded OR-Tools route-only benchmarks, repeated latency, and historical ablation synthesis. Real-world fleet/reorder economics and live integration remain unproven without company data.

## Sinirlar

- Rule-based baseline neutral continuous control kullandigi icin tam PPO+DQN alternatifi degildir.
- Amazon ve OR-Tools route tarafini destekler; secondary_fleet ve reorder_none ekonomisini kanitlamaz.
- Sirket verisi olmadan live TMS/WMS/ERP entegrasyonu veya evrensel optimalite iddia edilmemelidir.
