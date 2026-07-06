# Demo Script ve Soru-Cevap

## Demo Script

### 1. Dashboard

"Once mevcut production state'i gosteriyorum. Aktif model `joint_torch_v5_prod_hierarchical_v1_1m_20260611`; runtime `torch_joint`; DQN mimarisi `hierarchical_v1`."

Dosya:

`docs/00_PROJECT_DASHBOARD.md`

### 2. Production Handoff

"Bu dosya uretime alinan artifact'leri, hash'leri, registry ID'lerini ve rollback notlarini kaydediyor."

Dosya:

`docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`

### 3. Model Mimarisi

"Flat DQN yerine hiyerarsik DQN kullaniyoruz. 48 action sabit kaliyor, ama model bu action'i dispatch, route, fleet mode ve reorder alt kararlarindan uretiyor."

Vurgular:

- obs 73.
- action 48.
- action 24: `dispatch + shortest + secondary_fleet + none`.
- action 32: `dispatch + low_congestion + secondary_fleet + none`.

### 4. Runtime Latency

"Uretim checkpoint'i CPU'da yuklenip 1000 prediction ile olculdu. p95 yaklasik 1.68 ms."

Dosya:

`reports/benchmarks/runtime_latency_hierarchical_v1_20260613.json`

### 5. Rule-Based Benchmark

"Dijital ikiz scenario'larinda deterministik baseline ailelerini calistiran benchmark harness eklendi. Bu sprintte bounded 3-episode kosu yapildi."

Dosya:

`reports/benchmarks/rule_based_baseline_20260613_bounded_3ep/rule_based_baseline_report.json`

### 6. Public Route Proxy

"Amazon Last Mile kucuk sample'inda actual route sequence'lari greedy travel-time proxy ile karsilastirdik. Route tarafinda shortest ve low-congestion tercihleri yon olarak makul."

Dosya:

`reports/benchmarks/amazon_route_proxy_20260613/amazon_route_proxy_summary.json`

### 7. OR-Tools Smoke

"Klasik optimizer entegrasyon yolunu gostermek icin tiny VRPTW instance'i OR-Tools ile cozdurduk."

Dosya:

`reports/benchmarks/ortools_vrptw_smoke_20260613.json`

## Q&A

### Bu model su anda gercek operasyonda calisiyor mu?

Bu repository icinde production artifact olarak aktif ve promoted durumda. Gercek TMS/WMS/ERP entegrasyonu iddia edilmiyor.

### Model neden guvenilir?

Guvenilirlik iddiasi dort katmandan geliyor: gated digital-twin eval, protected registry/production lifecycle, runtime smoke/latency testleri ve monitoring runbook'u.

### En onemli teknik katkisi ne?

Flat DQN'deki action kalitesi problemini hiyerarsik action factorization ve teacher distillation ile cozmesi.

### Sirket verisi olmadan tez savunulabilir mi?

Evet, eger iddia dogru sinirlanirsa. Bu tez dijital ikiz ve public proxy ile desteklenen bir karar destek mimarisi sunar. Sirket verisiyle maliyet/fleet/reorder kalibrasyonu gelecek is olarak belirtilir.

### Action 24 ve 32 neden sorun degil?

Equal-budget residual-watch gate PASS. Amazon route sample route tarafinin yon olarak makul oldugunu gosteriyor. Ancak secondary fleet ve reorder none bilesenleri hala monitoring ve company-data validation gerektirir.

### Neden OR-Tools tam benchmark degil?

OR-Tools smoke sadece klasik VRPTW cozum yolunun mevcut oldugunu gosterir. 5PL ortamindaki envanter, dispatch failure, fleet tier ve reorder dinamiklerini tam temsil etmez.

### Bu sprintte production degisti mi?

Hayir. Registry, production, baselines, DB, checkpoints ve mevcut eval outputs korunmustur. Yeni yazilanlar benchmark/report/script/test/doc artifact'leridir.

### Son SOTA pathway sprintinde ne degisti?

Continuous-aware rule benchmark tam butcede 5120 episode row ile tamamlandi. Inventory tarafi `gym-invmgmt` ve MABIM ile, fleet/dispatch tarafi FleetPy ile, route reference tarafi PyVRP ile genisledi. SVRPBench ve CityFlow ise gercek external blocker/deferred olarak raporlandi.

### Bu artik SOTA demek mi?

Hayir. Bu, SOTA iddiasina giden benchmark pathway'in guclendigini gosterir. Global SOTA veya gercek dunya optimum iddiasi icin stochastic routing, congestion analog ve sirket verisi eksikleri kapanmalidir.

## 2026-06-14 Final Truth Addendum

This document is retained for chronology, but final advisor/thesis claims must use the 2026-06-14 evidence package. Superseded wording includes bounded 3-episode rule-baseline framing, 13-route Amazon sample as current route truth, OR-Tools smoke as current route truth, FleetPy single-scenario as complete fleet truth, SVRPBench as simply fully blocked, and lack of a historical replay adapter.

Current truth:

- Full rule-based and continuous-aware benchmark evidence is available.
- Amazon/OR-Tools/PyVRP/SVRP geometry and route-reference evidence supersede smoke-only wording.
- Fleet/dispatch evidence now includes HVFHV, FleetPy no-Gurobi RPP/IRS, SF taxi, and NYC TLC proxies, with remaining blockers documented.
- Public historical replay adapter exists and large replay ran on LaDe 31,415 rows, NYC HVFHS 100,000 rows, and Olist 96,476 feasible delivered rows.
- Public replay is descriptive proxy evidence only; no causal OPE, global SOTA, secondary-fleet economics, no-reorder optimality, or company-data validation is claimed.

Use these final sources instead:

- `docs/reports/20260614_final_truth_source_index_tr.md`
- `docs/reports/20260614_final_integrated_benchmark_scorecard_tr.md`
- `docs/reports/20260614_final_advisor_master_package_tr.md`
- `docs/thesis/20260614_lisans_tezi_guncel_taslak_tr.md`

