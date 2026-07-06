# 2026-06-13 Final Urun Guclendirme ve Benchmark Ozeti

## Sonuc

Siniflandirma: `FINAL_PRODUCT_STRENGTHENING_BENCHMARK_SPRINT_READY`

Bu sprint, mevcut `hierarchical_v1` 1M uretim modelini yeniden egitmeden veya uretim durumunu degistirmeden dort guclendirme ekseninde kanit uretti:

1. Deterministik rule-based baseline benchmark.
2. CPU runtime latency benchmark.
3. Amazon Last Mile public route-proxy benchmark.
4. OR-Tools VRPTW smoke benchmark.

Korunan alanlar: registry, production, baselines, DB, checkpoints ve mevcut eval ciktilari degistirilmedi.

## Uretim Gercegi

- Logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Uretim checkpoint: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- Runtime: `torch_joint`
- DQN mimarisi: `hierarchical_v1`
- Baslatma yontemi: `flat_teacher_distillation_v1`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation/action: `73 / 48`
- Surec: 250k PASS, 500k PASS, 1M PASS
- Equal-budget residual-watch gate: PASS

## Benchmark Ciktilari

| Benchmark | Cikti | Karar |
| --- | --- | --- |
| Rule-based baseline | `reports/benchmarks/rule_based_baseline_20260613_bounded_3ep/` | `RULE_BASED_BASELINE_BENCHMARK_READY` |
| Runtime latency | `reports/benchmarks/runtime_latency_hierarchical_v1_20260613.json` | `RUNTIME_LATENCY_BENCHMARK_READY` |
| Amazon route proxy | `reports/benchmarks/amazon_route_proxy_20260613/` | `AMAZON_ROUTE_PROXY_BOUNDED_SAMPLE_READY` |
| OR-Tools VRPTW smoke | `reports/benchmarks/ortools_vrptw_smoke_20260613.json` | `ORTOOLS_VRPTW_SMOKE_BENCHMARK_READY` |

## Rule-Based Baseline Benchmark

Kapsam:

- 8 mevcut scenario.
- 8 deterministik baseline.
- Her scenario icin 3 episode.
- Toplam 64 baseline-scenario ozeti.
- Cikti `reports/benchmarks` altina yazildi, `models/eval` degistirilmedi.

Not: 20 episode x 8 scenario x 8 baseline tam kosu 30 dakikalik calisma limitini asti. Bu nedenle sprint icinde bounded 3-episode benchmark kullanildi ve bu durum raporda acikca isaretlendi.

En iyi bounded baseline gozlemleri:

| Scenario | En iyi bounded baseline | Service | Dispatch success | Route failure |
| --- | --- | ---: | ---: | ---: |
| baseline_normal | conservative_stock_threshold_reorder | 1.0000 | 0.9972 | 1 |
| demand_spike_volatility | conservative_stock_threshold_reorder | 0.9570 | 1.0000 | 3 |
| high_holding_cost | conservative_stock_threshold_reorder | 1.0000 | 1.0000 | 1 |
| lead_time_volatility | conservative_stock_threshold_reorder | 1.0000 | 1.0000 | 0 |
| mixed_stress | conservative_stock_threshold_reorder | 0.9528 | 1.0000 | 1 |
| premium_sla_pressure | conservative_stock_threshold_reorder | 1.0000 | 0.9968 | 1 |
| route_disruption_congestion | conservative_stock_threshold_reorder | 1.0000 | 1.0000 | 1 |
| vehicle_scarcity_capacity_shock | conservative_stock_threshold_reorder | 1.0000 | 1.0000 | 1 |

Fatal hard-blocker toplamlarinin tamami `0`.

## Runtime Latency

Komut:

```powershell
python scripts/benchmark_runtime_latency.py --checkpoint models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt --output reports/benchmarks/runtime_latency_hierarchical_v1_20260613.json --iterations 1000 --seed 42 --device cpu
```

Sonuc:

- 1000 deterministik CPU prediction.
- p50: `1.0288 ms`
- p95: `1.6751 ms`
- p99: `1.9568 ms`
- max: `5.0587 ms`
- Tum continuous action ciktilari finite ve `[-1, 1]` icinde.
- Tum discrete action ciktilari `0..47` araliginda.

## Amazon Last Mile Route Proxy

Bu sprintte yeni buyuk public data indirimi yapilmadi. Mevcut onayli kucuk sample kullanildi:

- Root: `data/public/almrrc2021_small/`
- Route sayisi: `13`
- Route data, package data, directed travel times, actual sequences ve invalid sequence scores mevcut.

Oz metrikler:

- Actual / greedy travel-time ratio mean: `0.9673`
- Actual / greedy ratio median: `0.9728`
- Directed travel-time asymmetry mean: `0.0979`
- Directed travel-time asymmetry max mean-route: `0.1637`
- 2 saat veya daha dar time-window share: `0.0`

Yorum:

- Action 24 route bileseni, yani shortest/fastest-like route tercihi, public route verisinde yon olarak makul.
- Action 32 route bileseni, yani low-congestion/reliability tercihi, directed travel-time asymmetry nedeniyle yon olarak makul.
- Bu public sample secondary fleet, reorder none, sirket maliyetleri veya dispatch failure ekonomisini kanitlamaz.

Full Amazon listing `42` object ve `3.1 GiB`. En buyuk training travel-times dosyasi `1.7 GiB`. Mevcut analiz scripti JSON'u eager load ettigi icin full dataset icin once streaming parser planlanmalidir.

## OR-Tools VRPTW Smoke

Komut:

```powershell
python scripts/ortools_vrptw_smoke_benchmark.py --output reports/benchmarks/ortools_vrptw_smoke_20260613.json --time-limit-seconds 1
```

Sonuc:

- OR-Tools version: `9.15.6755`
- Tiny VRPTW instance feasible.
- Objective: `24`
- Route: `[0, 5, 4, 3, 2, 1, 0]`
- Served nodes: `[1, 2, 3, 4, 5]`

Yorum: Bu sadece klasik optimizer yolunun calistigini gosteren smoke benchmarktir. 5PL uretim modelinin tam alternatif benchmarki degildir.

## Tez ve Danisman Icin En Guclu Cikarim

Proje artik yalnizca "bir model egittim" noktasinda degil. Mevcut uretim modeli:

- dijital ikiz ortaminda gated ladder ile secildi,
- registry ve production promotion sureciyle guvenli sekilde paketlendi,
- residual watch'lari equal-budget benchmark ile izlenebilir seviyeye indirildi,
- rule-based, public-route, classical-optimizer smoke ve runtime latency eksenlerinde ek kanitla desteklendi.

## Sinirlar

- Sirket verisi olmadan secondary_fleet ve reorder_none ekonomisi kanitlanamaz.
- Amazon sample route tarafinda yardimci kanittir, operasyonel gerceklik garantisi degildir.
- Full public route dataset icin streaming parser gerekir.
- Blind 3M/5M/10M egitim onerilmez.

## Guvenli Sonraki Adim

1. Uretim monitoring runbook'u ile action 24/32, failed-noop, route failure ve service watch'larini takip et.
2. Sirket verisi gelirse once schema validator ve monitoring report generator ile kalite kontrol yap.
3. Daha buyuk public dataset gerekiyorsa once streaming Amazon parser gelistir.
4. Model arastirmasi gerekiyorsa 1.5M/2M gated extension planindan basla, blind 3M baslatma.
