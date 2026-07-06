# NYC HVFHV Dispatch Benchmark Report - 2026-06-14

## Classification

`NYC_HVFHS_DISPATCH_PROXY_READY_200K_MONTH_STRIDE_SAMPLE`

JSON report:

- `reports/benchmarks/fleet_dispatch_upgrade_20260614/nyc_hvfhs_dispatch/nyc_hvfhs_dispatch_report.json`

Source:

- `data/public/nyc_hvfhs_20260614/fhvhv_tripdata_2023-01.parquet`
- Source rows: 18,479,031
- Source size: 473,816,636 bytes
- Sample: 200,000 rows, deterministic stride every 92 rows

## Key Metrics

- Dispatch bases in sample: 11
- HVFHS licenses in sample: 2
- Request-to-pickup wait mean: 267.55 seconds
- Request-to-on-scene wait mean: 189.88 seconds over 147,481 rows
- Peak sampled rows: 73,831
- Off-peak sampled rows: 126,169
- Zone-hour buckets: 93,476
- Max sampled zone-hour demand pressure: 25
- Mean sampled zone-hour demand pressure: 2.14
- Mean driver pay: 16.79
- Mean base passenger fare: 21.55

The raw public data includes negative wait/fare/pay minima in the sample, so the report keeps them visible as public-data quality/anomaly evidence rather than silently cleaning them away.

## Capacity Proxy

| Variant | Vehicles per zone | Served proxy rate | Unserved proxy |
|---|---:|---:|---:|
| Low capacity | 1 | 0.0290 | 194,194 |
| Balanced capacity | 2 | 0.0573 | 188,540 |
| High capacity | 4 | 0.1153 | 176,939 |

This proxy is intentionally simple and pessimistic: it is a zone-local immediate capacity abstraction, not a real platform assignment policy.

## Interpretation

HVFHV materially improves fleet/dispatch external evidence because it is large, official, recent, and contains request-to-service timing fields that the previous 5,000-row FHV proxy lacked. It still cannot validate 5PL company economics, declined dispatches, carrier acceptance, or exact action 24/32 optimality.

