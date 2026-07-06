# Amazon Full Route-Proxy Benchmark Report

Final classification: `AMAZON_FULL_ROUTE_PROXY_READY`

Routes processed: `6112` of `6112` route-data routes.
Packages processed: `1457175`.
Windowed packages: `113993`; <=2h tight windows: `0`.

Outputs:
- `reports/benchmarks/full_completion_20260613/amazon_full_route_proxy/amazon_full_route_proxy_report.json`
- `reports/benchmarks/full_completion_20260613/amazon_full_route_proxy/amazon_full_route_proxy_summary.csv`
- `reports/benchmarks/full_completion_20260613/amazon_full_route_proxy/per_route_metrics.jsonl`

## Key Metrics

- Actual/greedy travel-time ratio mean/sd/min/max: `0.9852` / `0.0695` / `0.7963` / `1.6533`.
- Mean pair asymmetry mean/sd/min/max: `0.1000` / `0.0505` / `0.0371` / `0.6028`.
- Route-score distribution: `{'High': 2718, 'Low': 102, 'Medium': 3292}`.
- Full vs old sample route count: `6112` vs `13`.

Interpretation: this supports route-side plausibility for shortest-like and reliability/low-congestion preferences. It does not validate secondary-fleet or reorder-none economics.
