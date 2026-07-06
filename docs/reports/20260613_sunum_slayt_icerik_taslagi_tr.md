# Sunum Slayt Icerik Taslagi

## Slayt 1 - Baslik

5PL Dijital Ikiz Uzerinde Hiyerarsik Pekistirmeli Ogrenme ile Operasyonel Karar Destegi

Alt baslik: Production-ready model lifecycle, benchmark ve monitoring hattiyla guclendirilmis tez projesi.

## Slayt 2 - Problem

- 5PL operasyonlarinda dispatch, route, fleet ve reorder kararlarini birlikte almak gerekir.
- Klasik tekil metrik optimizasyonu, servis seviyesi, gecikme, kapasite ve envanter riskini ayni anda tasimakta zorlanir.
- Sirket verisi yoksa once dijital ikiz, public proxy ve guvenli monitoring altyapisi gerekir.

## Slayt 3 - Sistem Mimarisi

- `env_5pl`: dijital ikiz ortam.
- `observation_builder`: 73 boyutlu gozlem.
- `action_projector`: 5 boyutlu continuous kontrol.
- `discrete_action_mapper`: 48 ayrik action sozlesmesi.
- `hierarchical_v1`: dispatch, route, mode, reorder heads.
- `PolicyService`: active registry uzerinden runtime karar servisi.

## Slayt 4 - Neden Hiyerarsik DQN?

- Flat DQN action id'leri davranissal olarak kirilgan cepler olusturdu.
- Hiyerarsik DQN action'i alt kararlara ayirir.
- HOLD action'larda anlamsiz route/fleet bilesenleri maskelenir.
- External API degismez: hala 48 action.

## Slayt 5 - Training ve Gate Ozeti

- Production parent'tan hierarchical init.
- `flat_teacher_distillation_v1` ile warm-start.
- 250k PASS.
- 500k PASS.
- 1M PASS.
- 8/8 scenario PASS.
- Hard blockers zero.

## Slayt 6 - Uretime Alma

- Candidate registration.
- Active registry activation.
- Copy-only production promotion.
- Baseline, DB ve checkpoint korumasi.
- Production manifest ve final handoff.

## Slayt 7 - Residual Watch'lar

- Route action 32 concentration.
- Mixed action 24 concentration.
- Top-action concentration warnings.
- Mixed-success route-failure warnings.
- Equal-budget residual-watch gate PASS.
- Sonuc: monitoring item, yeni egitim tetigi degil.

## Slayt 8 - Benchmark Sprint

| Benchmark | Sonuc |
| --- | --- |
| Rule-based baseline bounded run | READY |
| Runtime latency | p95 `1.6751 ms` |
| Amazon route proxy | bounded sample READY |
| OR-Tools VRPTW smoke | feasible, objective `24` |

## Slayt 9 - Amazon Route Proxy

- 13 route public sample.
- Actual sequence ve directed travel-time matrix mevcut.
- Actual/greedy travel-time ratio mean `0.9673`.
- Directed asymmetry mean `0.0979`.
- Route-side plausibility var.
- Fleet/reorder ekonomisi icin company data gerekir.

## Slayt 10 - Runtime

- `load_torch_joint_policy` production checkpoint'i CPU'da yukler.
- `PolicyService.predict_joint` algorithm=`torch_joint` doner.
- Continuous action len `5`, finite ve bounded.
- Discrete action `0..47`.
- SB3 loader path'ine dusmez.

## Slayt 11 - Sinirlar ve Durluk

- Gercek sirket verisi yok.
- Public route data sadece route tarafini destekler.
- No universal optimality claim.
- No blind 3M.
- Company-data validation gelmeden fleet/reorder ekonomisi iddia edilmez.

## Slayt 12 - Sonraki Adim

- Production monitoring.
- Company-data intake validation.
- Equal-budget veya streaming public benchmark genisletmesi.
- Gerekirse 1.5M/2M gated research plan.

## Slayt 13 - SOTA Pathway Sprint Update

| Track | Sonuc |
| --- | --- |
| Continuous-aware rule benchmark | 5120 episode row READY |
| Inventory public analog | `gym-invmgmt` + MABIM READY |
| Fleet/dispatch analog | FleetPy example READY |
| Route reference | PyVRP 5 CVRPLIB success |
| Stochastic routing | SVRPBench BLOCKED |
| Congestion analog | CityFlow source-build DEFERRED |

Konusma notu: "Bu paket SOTA iddiasini degil, SOTA'ya giden kanit yolunu guclendiriyor. Calisan public analoglari ve blocker'lari ayri ayri gosteriyoruz."

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

