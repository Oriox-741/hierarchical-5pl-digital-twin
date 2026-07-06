# Updated Benchmark Scorecard After Continuous and Non-Route Expansion

Tarih: 2026-06-13

## Summary

Bu skorlar iddia değil, kanıt olgunluğu değerlendirmesidir. Global SOTA iddiası yapılmaz.

| Area | Score | Rationale |
| --- | ---: | --- |
| Rule-based baseline score | 4.0 / 5 | Full continuous-aware 5120-row benchmark tamamlandı; heuristic continuous ayrıştırması sınırlı. |
| Inventory/reorder external evidence | 4.0 / 5 | `gym-invmgmt` ve MABIM iki ayrı public inventory kanıtı sağladı; şirket maliyeti yok. |
| Fleet/dispatch external evidence | 3.0 / 5 | FleetPy public example çalıştı; tek küçük senaryo, tam 5PL değil. |
| Route benchmark score | 3.5 / 5 | PyVRP CVRPLIB success, OR-Tools/Amazon önceki kanıtlar var; SVRPBench ve VRPTW format blocked. |
| SOTA pathway maturity | 3.5 / 5 | Benchmark suite tanımlı ve birçok track çalışıyor; stochastic/congestion blocker ve company-data eksik. |
| Thesis-defense readiness | 4.0 / 5 | Danışmana gösterilebilir güçlü paket var; overclaim sınırları açık. |

## What Improved

- Continuous-aware rule benchmark artık tam 5120 episode row ile tamam.
- Inventory tarafında iki ayrı public ortam sonucu var.
- FleetPy ile public fleet-dispatch analog koşuldu.
- PyVRP route reference eklendi.
- CODEX-5PL benchmark suite formalize edildi.

## What Is Still Weak

- SVRPBench kaynak/paket erişimi çözülemedi.
- CityFlow/LibSignal source-build analog işi deferred.
- PyVRP VRPTW dosyaları format nedeniyle blocked.
- Şirket verisi olmadan secondary fleet ve reorder ekonomisi kesin doğrulanamaz.

## Safe Claim

"CODEX PROJE, 5PL dijital ikizi için production-promoted hierarchical v1 modelini; rule-based, route, inventory ve fleet-dispatch public benchmark kanıtlarıyla destekleyen olgun bir benchmark yoluna sahiptir."

## Unsafe Claim

"CODEX PROJE global SOTA'dır" veya "gerçek dünyada optimaldir" denmemelidir.

## Classification

`UPDATED_BENCHMARK_SCORECARD_READY_WITH_EXTERNAL_BLOCKERS`
