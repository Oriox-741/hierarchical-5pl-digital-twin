# Tez Danismani Tek Sayfa Proje El Notu

Tarih: 2026-06-13

Durum: Danismana dogrudan verilebilir tek sayfa el notu. Bu dokuman yalnizca aciklama ve planlama amaclidir; egitim, offline eval, long-run gate, veri indirme, registry/production/baseline/DB/checkpoint degisikligi veya sirket verisi talebi/ingest islemi yapmaz.

## Proje Adi

CODEX PROJE: 5PL Digital Twin Icin Hierarchical RL Karar Kontrol Prototipi

## Tek Cumlelik Aciklama

Bu proje, 5PL lojistik operasyonlarini simule eden bir digital twin icinde siparis, rota, filo ve reorder kararlarini PPO+DQN tabanli hierarchical reinforcement learning ile birlikte veren ve production-like MLOps sureciyle yoneten bir arastirma prototipidir.

## Problem

5PL lojistikte kararlar birbirinden bagimsiz degildir. Bir siparisi dispatch etmek, hangi rotanin secilecegi, primary/secondary fleet kullanimi ve reorder davranisi ayni anda service, lateness, kapasite, stok ve maliyet baskilarini etkiler. Bu nedenle proje sadece rota optimizasyonu degil, cok katmanli lojistik karar koordinasyonu problemidir.

## Cozum

Proje, `env_5pl` digital twin icinde PPO+DQN joint policy kullanir:

- PPO: 5 continuous strategic/physical control uretir.
- DQN: 48 discrete tactical action arasindan karar secer.
- `hierarchical_v1`: DQN action kararini dispatch, route, fleet ve reorder basliklarina ayirir.
- Runtime: `torch_joint`, `PolicyService`, production manifest, monitoring ve ops bundle ile yonetilir.

## Sistem Ne Yapiyor?

Her karar dongusunde:

1. Simulator mevcut operasyon durumunu uretir.
2. `observation_builder` bunu 73 boyutlu observation'a cevirir.
3. PPO 5 continuous control uretir.
4. DQN 0..47 arasi tactical action secer.
5. `DiscreteActionMapper` action'i dispatch/hold, route, fleet ve reorder kararina acar.
6. Safety/feasibility katmani karari sinirlar.
7. `env_5pl` karari simule eder ve service/lateness/dispatch/route/no-work metriklerini toplar.

48 DQN action su carpimdan gelir:

`2 dispatch x 3 route x 2 fleet x 4 reorder = 48`

Ornek residual-watch action'lar:

- Action 24: `dispatch + shortest + secondary_fleet + none`
- Action 32: `dispatch + low_congestion + secondary_fleet + none`

## Current Model

- Logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Runtime: `torch_joint`
- Architecture: `hierarchical_v1`
- Init method: `flat_teacher_distillation_v1`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`
- PPO continuous action dimension: `5`
- DQN external discrete action count: `48`
- Training step: `1,000,000`
- Status: active registry + copy-only production promoted

## Kanit

Mevcut model proje simulator/eval contract'i icinde su kanitlara sahiptir:

- 250k PASS
- 500k PASS
- 1M PASS
- 8/8 scenario PASS
- hard blockers zero
- long-run gate PASS
- equal-budget residual-watch gate PASS
- production ops bundle: `OPS_BUNDLE_CLEAN`

Bu sonuc, modelin mevcut 5PL simulator sozlesmesi icinde guclu oldugunu gosterir. Gercek dunya optimumlugu iddiasi degildir.

## Ne Degildir?

Bu proje:

- live TMS/WMS/ERP entegrasyonu degildir;
- gercek sirkette otomatik dispatch veya reorder yapmaz;
- private company data kullanmaz;
- real-world optimality kaniti degildir;
- secondary_fleet ve reorder-none ekonomisini sirket verisi olmadan kanitlamaz;
- daha uzun blind 3M/5M/10M egitimi otomatik olarak hakli cikarmaz.

## Akademik Deger

Projenin akademik degeri tek bir model skorundan cok entegre sistem mimarisindedir:

- 5PL digital twin;
- PPO+DQN joint reinforcement learning;
- hierarchical/factorized DQN action design;
- simulator stress scenarios ve gate sistemi;
- exact-resume stability;
- production-like registry, manifest, monitoring, artifact-health ve ops governance;
- simulation-to-real calibration icin acik veri ve benchmark planlari.

## Siradaki Arastirma

Bu task sirket verisi talep etmez. Siradaki safe-now arastirma adimi benchmark protokoludur:

1. once advisor'a anlatilabilir rule-based baseline design/spec;
2. sonra approval varsa OR-Tools feasibility;
3. download approval varsa Amazon full route-proxy expansion;
4. eval approval varsa multi-seed robustness;
5. training approval ve bilimsel gerekce varsa ablation.

Guvenli danisman cumlesi:

> Sistem 5PL digital twin icinde otonom karar loop'u yurutur ve proje sozlesmesi icinde production-like olarak onaylanmistir. Ancak gercek dunya deployment iddiasi icin benchmarklar, ablationlar, sirket verisi kalibrasyonu ve entegrasyon validasyonu gerekir.

