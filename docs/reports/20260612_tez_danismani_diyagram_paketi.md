# Tez Danismani Diyagram Paketi

Tarih: 2026-06-13

Durum: Tez danismani anlatimi icin cizim paketi. Bu dokuman diyagram tarifidir; yeni eval, training, gate, veri indirme, registry/production/baseline/DB/checkpoint degisikligi veya sirket verisi talebi yapmaz.

## 1. Overall Architecture: Sense -> Think -> Act -> Learn -> Govern

Title: 5PL Digital Twin Karar Kontrol Mimarisi

What boxes to draw:

- Sense: `env_5pl`, orders, inventory, fleet, route, stress state
- Think: observation 73, PPO, DQN `hierarchical_v1`
- Act: action projector, discrete action mapper, safety/feasibility, env step
- Learn: curriculum, replay, reward, exact resume, gates
- Govern: registry, active model, production manifest, artifact health, monitoring, ops bundle

Arrows:

- Sense -> Think: operational state becomes 73-dimensional observation
- Think -> Act: PPO continuous controls + DQN discrete action
- Act -> Learn: reward and scenario metrics
- Learn -> Govern: candidate/gate evidence
- Govern -> Sense/Think: approved production model and monitoring feedback

Advisor explanation:

Bu diyagram projenin yalnizca model egitimi olmadigini, simulator, policy, eval ve governance katmanlarini birlestirdigini gosterir.

Caveat:

Governance proje icinde production-like'tir; real company deployment veya live TMS/WMS/ERP entegrasyonu degildir.

## 2. One Operation Flow

Title: Tek Operasyon Karar Dongusu

What boxes to draw:

- env state
- observation 73
- PPO 5 continuous control
- DQN 48 tactical action
- action mapper
- safety/feasibility
- env step
- reward/metrics
- monitoring watches

Arrows:

- env state -> observation 73
- observation 73 -> PPO and DQN
- PPO -> continuous physical control
- DQN -> action id 0..47
- action id -> action mapper -> dispatch/route/fleet/reorder
- mapped action -> safety/feasibility -> env step
- env step -> metrics -> monitoring

Advisor explanation:

Her step'te model mevcut operasyon durumunu sayisal olarak gorur, iki karar katmani uretir, karar lojistik komuta cevrilir, safety'den gecer ve digital twin icinde uygulanir.

Caveat:

Walkthrough ornekleri temsilidir; mevcut eval artefaktlari step-level PPO vector veya DQN raw Q/logit trace saklamaz.

## 3. PPO vs DQN

Title: Surekli Stratejik Kontrol ve Ayrik Taktik Karar Ayrimi

What boxes to draw:

- PPO box: 5 continuous strategic/physical levers
- DQN box: 48 tactical actions
- Shared input: observation 73
- Shared output: joint action to simulator

Arrows:

- observation 73 -> PPO
- observation 73 -> DQN
- PPO output -> action projector
- DQN output -> discrete action mapper
- both -> joint env step

Advisor explanation:

PPO operasyonun genel tonunu ayarlar; DQN "hangi taktik lojistik hamlesi?" sorusunu cevaplar.

Caveat:

PPO'nun 5 degeri tek basina dispatch action degildir; joint modda dispatch/hold semantigi DQN tarafindan gelir.

## 4. DQN Action Decomposition

Title: 48 Action'in Lojistik Bilesenlere Ayrilmasi

What boxes to draw:

- Dispatch head: hold / dispatch
- Route head: shortest / low_congestion / high_resilience
- Fleet head: secondary_fleet / primary_fleet
- Reorder head: none / conservative / aggressive / emergency
- External action space: 0..47

Arrows:

- four heads -> composer -> external 48 Q values
- external action id -> mapper -> structured logistics command

Advisor explanation:

`hierarchical_v1`, 48 action'i tek duz liste olarak ezberlemek yerine dispatch, route, fleet ve reorder alt kararlarina ayirir. Bu daha okunabilir ve action-pocket drift riskine karsi daha kontrolludur.

Caveat:

Disariya hala 48 action verilir; runtime contract degismemistir.

## 5. Action 24

Title: Action 24 Decode

What boxes to draw:

- Action id: 24
- Dispatch: dispatch
- Route: shortest
- Fleet: secondary_fleet
- Reorder: none
- Monitoring watch: mixed_stress action concentration

Arrows:

- 24 -> decode formula
- decode formula -> `dispatch + shortest + secondary_fleet + none`
- action -> mixed_stress monitoring watch

Advisor explanation:

Action 24, mixed stress benzeri durumlarda sik gorulen bir karardir: dispatch yap, shortest route kullan, secondary fleet sec, DQN tarafindan ek reorder override yapma.

Caveat:

Bu action gate'i gecmis ve hard blocker uretmemistir; fakat secondary_fleet ve no-reorder ekonomisi sirket verisi olmadan kanitlanmis sayilmaz.

## 6. Action 32

Title: Action 32 Decode

What boxes to draw:

- Action id: 32
- Dispatch: dispatch
- Route: low_congestion
- Fleet: secondary_fleet
- Reorder: none
- Monitoring watch: route_disruption action concentration

Arrows:

- 32 -> decode formula
- decode formula -> `dispatch + low_congestion + secondary_fleet + none`
- action -> route_disruption monitoring watch

Advisor explanation:

Action 32, route disruption/congestion durumunda low-congestion route ailesini tercih eder. En kisa rota congestion altinda her zaman en iyi operasyonel tercih olmayabilir.

Caveat:

Amazon small sample route-side plausibility saglar; fleet/reorder/cost kismi public route-only data ile kanitlanmaz.

## 7. Evaluation And Gating

Title: Scenario Suite -> Gate -> Residual Watch -> Production

What boxes to draw:

- 8 eval scenarios
- scenario summaries and episode metrics
- hard blocker checks
- long-run gate
- equal-budget residual-watch eval
- production promotion decision

Arrows:

- scenario configs -> closed-loop eval
- eval outputs -> threshold and hard blocker checks
- checks -> long-run gate
- gate -> residual-watch assurance
- assurance -> copy-only production promotion

Advisor explanation:

Model sadece aggregate reward ile kabul edilmemistir. Sekiz senaryo, hard blockers, service/lateness/dispatch checks ve equal-budget residual-watch assurance birlikte kullanilmistir.

Caveat:

Bu gate'ler simulator sozlesmesi icindir; real-world deployment kaniti yerine gecmez.

## 8. Production Governance

Title: Registry -> Active -> Manifest -> Health -> Ops Bundle

What boxes to draw:

- candidate registration
- active registry
- production directory
- production manifest and SHA256 hashes
- artifact-health report
- monitoring report
- company schema validator synthetic mode
- ops bundle

Arrows:

- candidate -> active registry
- active registry -> production artifact identity
- production files -> manifest/hashes
- manifest + registry -> artifact health
- artifact health + monitoring + schema validator -> ops bundle

Advisor explanation:

Proje modeli sadece checkpoint dosyasi olarak birakmaz; kimlik, hash, registry, manifest ve monitoring raporlariyla izlenebilir hale getirir.

Caveat:

Ops bundle clean olmak modelin real-world optimum oldugunu kanitlamaz; protected-state ve synthetic monitoring sagligini gosterir.

## 9. Real-World Boundary

Title: Simulator Autonomy vs Real Company Deployment

What boxes to draw:

- Left side: simulator autonomy
  - env_5pl
  - scenario arena
  - PolicyService
  - gates
  - monitoring
- Right side: real company deployment, future only
  - live TMS/WMS/ERP
  - real order stream
  - real fleet/carrier data
  - real inventory/reorder/cost data
  - human override and audit

Arrows:

- simulator evidence -> benchmark protocol -> future data/integration validation
- no direct arrow from simulator to live autonomous deployment

Advisor explanation:

Proje simulatorde otonom karar verir ve proje icinde production-like yonetilir. Gercek sirkette otomatik operasyon icin veri, entegrasyon, shadow mode ve insan onayi gerekir.

Caveat:

Bu task sirket veri talebi hazirlamaz ve gercek veriyi okumaz; sadece benchmark ve advisor handoff paketidir.

