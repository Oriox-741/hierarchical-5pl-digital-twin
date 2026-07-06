# Lisans Tezi Ilk Taslak

## Baslik

5PL Dijital Ikiz Ortaminda Hiyerarsik Pekistirmeli Ogrenme ile Operasyonel Karar Destegi

## Ozet

Bu calisma, besinci parti lojistik operasyonlarinda dispatch, rota secimi, filo modu ve yeniden siparis kararlarini birlikte ele alan bir karar destek sistemi tasarlar. Sistem, fiziksel gerceklik kisitlarini iceren bir 5PL dijital ikiz ortaminda egitilen Torch tabanli ortak PPO+DQN politikasina dayanir. Ayrik karar uzayi 48 action'dan olusur; nihai sistemde flat DQN yerine `hierarchical_v1` mimarisi kullanilarak action, dispatch, route, mode ve reorder alt kararlarina ayrilir. Bu sayede anlamsiz action cepleri azaltilir ve route/fleet/reorder davranisi daha yorumlanabilir hale getirilir. Nihai model 250k, 500k ve 1M gated degerlendirme asamalarini gecmis, production artifact olarak paketlenmis ve monitoring/benchmark altyapisi ile desteklenmistir.

## 1. Giris

Lojistik operasyonlarinda kararlar yalnizca bir araci bir siparise atamaktan ibaret degildir. Servis seviyesi, gecikme riski, kapasite soklari, rota bozulmalari, envanter riski ve maliyet baskisi ayni anda ortaya cikar. Bu nedenle tekil sezgisel kurallar bazen yeterli olurken, karisik stres altinda cok amacli karar verme ihtiyaci dogar.

Bu tezde amac, gercek sirket verisi yokken once guvenli bir dijital ikiz ve model lifecycle altyapisi kurmak, ardindan model davranisini scenario gate'leri, rule-based baseline'lar, public route proxy'leri ve runtime benchmark'lari ile desteklemektir.

## 2. Problem Tanimi

Problem, 5PL operasyonlarinda asagidaki kararlarin birlikte alinmasidir:

- dispatch veya hold,
- shortest, low_congestion veya high_resilience route tercihi,
- primary veya secondary fleet kullanimi,
- none, conservative, aggressive veya emergency reorder modu.

Bu alt kararlar mevcut sistemde 48 ayrik action'a encode edilir. Observation boyutu 73'tur ve continuous kontrol boyutu 5'tir.

## 3. Dijital Ikiz Ortami

`env_5pl` fiziksel ve operasyonel kisitlari simule eder:

- siparis ve arac durumlari,
- kapasite ve dispatch uygunlugu,
- rota bozulmasi ve congestion baskisi,
- envanter ve reorder sinyalleri,
- servis seviyesi ve lateness odakli reward/metric yapisi.

Degerlendirme scenario'lari `configs/eval_scenarios` altinda tutulur ve `real_world_scenario_arena` ile kosulur.

## 4. Model Mimarisi

Sistem ortak Torch PPO+DQN politikasidir:

- PPO continuous kontrol uretir.
- DQN tactical dispatch action uretir.
- Runtime `PolicyService` tarafindan `torch_joint` algoritmasi olarak sunulur.

Flat DQN denemelerinde action id'leri arasinda davranissal kirilganlik gorulmustur. Bu nedenle `hierarchical_v1` tasarlanmistir. Bu mimari action'i su alt basliklara boler:

- dispatch head,
- route head,
- mode head,
- reorder head.

Sonra bu alt kararlar tekrar dis sozlesmedeki 48 action'a compose edilir. Dis API degismez.

## 5. Initialization

Flat production checkpoint, hierarchical mimariyle exact resume uyumlu degildir. Bu nedenle random baslatma yerine `flat_teacher_distillation_v1` secilmistir. Flat teacher read-only yuklenir, sampled observation bank uzerinden hierarchical student 48-Q ciktilarini teacher'a yaklastirir. Bu islem training environment rollout'u degil, supervised warm-start niteligindedir.

## 6. Egitim ve Degerlendirme

Nihai ladder:

- 250k PASS.
- 500k PASS.
- 1M PASS.

1M model:

- 8/8 scenario PASS.
- 160 episode row.
- Hard blockers zero.
- Long-run gate PASS.
- Equal-budget residual-watch gate PASS.

Residual watch'lar:

- route action 32 concentration,
- mixed action 24 concentration,
- top-action concentration warnings,
- mixed-success route-failure warnings.

Bu watch'lar monitoring konusu olarak kabul edilmistir; yeni training tetigi degildir.

## 7. Production Lifecycle

Model once candidate olarak registry'ye eklendi, sonra active registry'ye gecirildi ve en son copy-only production promotion ile `models/production` altina kopyalandi. Promotion sirasinda:

- baseline degismedi,
- DB degismedi,
- source checkpoint degismedi,
- registry yalnizca onayli adimlarda degisti,
- production manifest yazildi.

## 8. Benchmark Guclendirmesi

### Rule-Based Baseline

Deterministik baseline aileleri:

- FIFO shortest primary none,
- earliest-due shortest,
- premium-first,
- low-congestion under disruption,
- conservative stock-threshold reorder,
- emergency stockout prevention,
- vehicle scarcity fallback,
- high-holding no-overstock.

Bu sprintte bounded 3-episode benchmark kosuldu ve `RULE_BASED_BASELINE_BENCHMARK_READY` sonucu alindi.

### Runtime Latency

Production checkpoint CPU uzerinde 1000 kez predict edildi:

- p50 `1.0288 ms`,
- p95 `1.6751 ms`,
- p99 `1.9568 ms`.

### Amazon Last Mile Public Proxy

13 route'luk public sample analiz edildi:

- actual/greedy travel-time ratio mean `0.9673`,
- directed travel-time asymmetry mean `0.0979`.

Bu sonuc action 24 ve action 32'nin route bilesenlerini yon olarak destekler, ancak secondary fleet ve reorder ekonomisini kanitlamaz.

### OR-Tools VRPTW Smoke

Tiny VRPTW instance OR-Tools ile feasible cozuldu:

- objective `24`,
- route `[0, 5, 4, 3, 2, 1, 0]`.

## 9. Sinirlar

Bu tez asagidaki iddialari yapmaz:

- gercek sirket verisinde evrensel optimum,
- live TMS/WMS/ERP deployment,
- secondary fleet ekonomisinin public route data ile kanitlandigi,
- reorder none kararinin maliyet olarak kesin dogru oldugu.

Bu alanlar company data gerektirir.

## 10. Sonuc

Calisma, dijital ikiz tabanli bir 5PL karar destek modelini egitimden uretim artifact'ine kadar tasiyan, hiyerarsik action mimarisiyle action kalitesini iyilestiren ve benchmark/monitoring katmanlariyla savunulabilir hale getiren bir sistem sunar. En guclu katkisi, model basarimindan cok model lifecycle'in tamamini bilimsel olarak izlenebilir ve guvenli hale getirmesidir.

## 11. Gelecek Calisma

- Sirket verisi ile schema ve join-key validation.
- Fleet, dispatch failure, inventory ve cost calibration.
- Amazon full dataset icin streaming route parser.
- Full-budget rule-based benchmark.
- Gerekirse 1.5M/2M gated extension.

## 2026-06-13 Full Benchmark Sprint Addendum

Full completion benchmark sprinti sonrasi ek kanitlar:

- Rule-based full 20ep benchmark: 1280 episode row.
- Amazon full route proxy: 6112 public route ve 1.46M package route-side analiz.
- OR-Tools route suite: Solomon/Homberger/CVRPLIB instance benchmarklari.
- Multi-seed robustness: 42-46 seed araliginda 800 episode row.
- Runtime latency repeated: 5 x 1000 CPU prediction.
- Historical ablation synthesis: flat DQN, exact-resume, teacher-retention ve hierarchical_v1 karsilastirmasi.

Bu ekler tezin iddiasini guclendirir, ancak sirket verisi olmadan secondary_fleet/reorder_none ekonomisi ve live entegrasyon iddia edilmemelidir. Ayrintilar: `docs/reports/20260613_full_benchmark_statistical_synthesis_tr.md`.

## 2026-06-13 SOTA Pathway Execution Addendum

Son sprintte benchmark kanitlari public analoglarla genisletildi:

- Continuous-aware rule baseline: 8 baseline x 8 scenario x 20 episode x 4 mode = 5120 episode row.
- Inventory/reorder: `gym-invmgmt` serial inventory benchmark ve MABIM/ReplenishmentEnv `sku50.single_store.standard` benchmark.
- Fleet/dispatch: FleetPy `example_pool_irsonly_sc_1` public scenario.
- Route reference: PyVRP ile 5 CVRPLIB instance success; 3 VRPTW dosyasi format uyumsuzlugu nedeniyle blocked.
- Stochastic routing: SVRPBench package/source path bulunamadigi icin blocked.
- Congestion analog: CityFlow PyPI install olmadigi ve source-build analog is oldugu icin deferred.

Bu ek kanitlar tez savunmasinda "sistematik benchmark yolu" olarak sunulabilir. Ancak bu kanitlar global SOTA veya gercek sirket operasyonunda optimalite iddiasi yapmaz. Sirket verisi halen fleet cost, dispatch failure, secondary_fleet ve reorder_none kararlarinin ekonomik kalibrasyonu icin gereklidir.

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

