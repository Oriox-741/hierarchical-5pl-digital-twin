# Runtime Latency Repeated Benchmark Report

Final classification: `RUNTIME_LATENCY_REPEATED_BENCHMARK_READY`

Coverage: `5` runs x `1000` predictions per run.

Output:
- `reports/benchmarks/full_completion_20260613/runtime_latency_repeated/runtime_latency_repeated_report.json`

## Aggregate Latency Across Runs

- Mean latency mean/sd/min/max: `0.8726` / `0.0845` / `0.8000` / `1.0026` ms.
- P95 mean/sd/min/max: `1.2435` / `0.1800` / `1.0779` / `1.5375` ms.
- P99 mean/sd/min/max: `1.5655` / `0.3482` / `1.2824` / `2.1416` ms.
- Per-run max latency mean/max: `5.1267` / `16.2050` ms.

Interpretation: CPU inference is sub-millisecond on average with low p99; one max outlier remains visible and should be treated as normal runtime jitter unless repeated under monitored load.
