# Tez Danismani Icin Detayli Tek Operasyon Walkthrough

Tarih: 2026-06-12

Durum: Tez danismani icin ayrintili operasyon anlatimi. Bu dokuman egitim, offline eval, long-run gate, registry degisikligi, production degisikligi, baseline degisikligi, DB degisikligi, checkpoint degisikligi, mevcut eval ciktisi degisikligi, veri seti indirme veya ozel sirket verisi ingest etmeden hazirlanmistir.

## 0. Bu Dokuman Neyi Gosteriyor?

Bu dokuman, mevcut hierarchical v1 production modelinin simule 5PL digital twin icinde bir operasyon kararini nasil yurutebilecegini adim adim anlatir. Amac danismana su zinciri gostermektir:

`operasyon durumu -> observation 73 -> PPO 5 continuous control -> DQN 48 discrete action -> action mapper -> safety/feasibility -> env_5pl uygulamasi -> reward/metrics -> monitoring`

Ana ornek action 24'tur:

`dispatch + shortest route + secondary_fleet + no reorder`

Ikinci kisa karsilastirma action 32'dir:

`dispatch + low_congestion route + secondary_fleet + no reorder`

Bu dokuman gercek production sistemine bagli canli bir operasyon logu degildir. Simulasyon kodunun nasil calistigini, mevcut aggregate eval/monitoring kanitlariyla birlikte danismanin anlayacagi operasyon diline cevirir.

## 1. Izlenebilirlik Notu

Bu dokuman iki kanit katmanini ayirir:

1. **Gercek kod ve artefakt kaniti:** `env_5pl`, `observation_builder`, `action_projector`, `discrete_action_mapper`, `safety_projector`, `joint_policies`, eval summary ve monitoring raporlari.
2. **Kodla uyumlu temsili operasyon ornegi:** Action 24 ve action 32 icin asagidaki tek-step anlatimlar, mevcut eval artefaktlarinda ham step trace/logit/PPO sayisal vektoru bulunmadigi icin birebir loglanmis bir operasyon adimi degildir. Bunlar, kodun nasil calistigini ve aggregate metriklerin neyi destekledigini gosteren temsili rekonstruksiyonlardir.

Bu nedenle bu raporda kesin olarak iddia edilenler:

- observation boyutu `73`;
- PPO continuous action boyutu `5`;
- DQN external discrete action sayisi `48`;
- action 24 decode sonucu `dispatch + shortest + secondary_fleet + none`;
- action 32 decode sonucu `dispatch + low_congestion + secondary_fleet + none`;
- current production model `joint_torch_v5_prod_hierarchical_v1_1m_20260611`;
- runtime `torch_joint`;
- DQN mimarisi `hierarchical_v1`;
- init method `flat_teacher_distillation_v1`;
- equal-budget residual-watch eval sonucu 8/8 PASS ve hard blocker zero.

Kesin olarak iddia edilmeyenler:

- belirli bir step icin ham PPO vektor degerleri;
- belirli bir step icin DQN raw Q/logit degerleri;
- belirli bir order id, vehicle id veya path'in gercek step trace'i;
- action 24/32'nin gercek sirket verisinde optimal oldugu.

## 2. Mevcut Production Model

Mevcut promoted production model:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- production checkpoint: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- architecture: `hierarchical_v1`
- initialization: `flat_teacher_distillation_v1`
- runtime: `torch_joint`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: `73`
- PPO continuous action dimension: `5`
- DQN external discrete action count: `48`
- ladder: 250k PASS, 500k PASS, 1M PASS
- 1M eval: 8/8 scenario PASS, 160 episode rows, hard blockers zero, long-run gate PASS
- equal-budget residual-watch result: PASS
- ops bundle result: `OPS_BUNDLE_CLEAN`

## 3. Tek Operasyon Dongusu

Bir operasyon adimi su sirayla ilerler:

1. `env_5pl` simule lojistik dunyanin mevcut durumunu tutar.
2. `observation_builder` bu durumu 73 boyutlu observation vektorune cevirir.
3. PPO policy 5 boyutlu continuous control uretir.
4. DQN policy 0 ile 47 arasinda bir discrete action secer.
5. `ActionProjector` PPO cikisini fiziksel/surekli kontrollere projekte eder.
6. `DiscreteActionMapper` DQN action id'sini dispatch, route, fleet mode ve reorder kararlarina acar.
7. `SafetyProjector` continuous ve discrete komutlari guvenlik sinirlarindan gecirir.
8. `env_5pl` komutu simule eder: stok, kapasite, arac hizi, dispatch, rota, vehicle secimi, teslimat sureci ve metrikler guncellenir.
9. Reward ve episode metrikleri toplanir.
10. Monitoring raporu bu karar ailelerini action-level operational watch olarak izler.

## 4. Operation Pre-State: Mixed Stress Benzeri Durum

Asagidaki operasyon ornegi **kodla uyumlu temsili bir action 24 operasyonudur**. Gercek eval artefaktlari step-level trace saklamadigi icin bu bir birebir log degildir.

Mixed stress senaryosu su baskilari birlestirir:

- demand spike: rolling demand probability ve volume artar;
- urgent order probability artar;
- holding cost artar;
- vehicle availability azalir;
- fleet capacity azalir;
- capacity shock vardir;
- route disruption probability artar;
- congestion artar;
- stockout ve inventory shortfall cezasi artar.

Bu tur pre-state'te kontrol kulesi soyle bir tablo gorebilir:

- bekleyen siparis sayisi yuksek;
- bazi siparisler SLA acisindan yakin deadline'a sahiptir;
- stok seviyesi ve safety stock arasi fark izlenir;
- primary fleet kisitli veya premium islerde daha pahali olabilir;
- secondary fleet gecici kapasite saglayabilir;
- rota adaylari arasinda shortest, low_congestion ve high_resilience skor farklari gorunur;
- operasyon hem teslimat gecikmesi hem de fazla stok maliyeti baskisi altindadir.

## 5. Observation 73: Modelin Gordugu Ekran

Sistem gercek dunyayi direkt metin gibi okumaz. `observation_builder` mevcut durumu 73 sayilik normalize bir vektore cevirir. Bu 73 ozellik gruplara ayrilir:

| Grup | Adet | Ne anlatir? |
| --- | ---: | --- |
| Legacy operational features | 32 | zaman, pending/delivered/late/failed, service, cost, inventory, demand, vehicle, route, DB ve bias sinyalleri |
| Reality features | 12 | inventory coverage, stockout, safety-stock gap, demand volatility, lead-time, backlog age, urgent ratio, speed/cost, premium fleet, route disruption, capacity slack |
| Dispatch feasibility features | 5 | available vehicles, dispatchable orders, feasible dispatch opportunity/ratio, vehicle availability pressure |
| Real-world stress features | 8 | holding/stockout/capacity/route/premium/supplier/lead-time stressleri |
| Route candidate features | 16 | shortest/low_congestion/high_resilience candidate skor ve gap'leri, route pressure, secondary fleet feasibility, useful dispatch |

Bu vektor, danisman icin "kontrol kulesinin ekrani" gibi dusunulebilir. Model bu ekrana bakar, fakat ekrani 73 sayilik sabit bir sayisal temsil olarak gorur.

## 6. PPO Rolu: 5 Continuous Control

PPO, taktik menu secimi yapmaz. PPO'nun rolu genel operasyon tonunu ayarlamaktir. `ActionProjector` PPO'nun 5 boyutlu cikisini su fiziksel alanlara map eder:

1. `reorder_fraction`
2. `dispatch_intensity`
3. `speed_multiplier`
4. `safety_stock_multiplier`
5. `capacity_buffer_fraction`

Joint action modunda kritik ayrim sudur:

- PPO continuous action stok, kapasite ve hiz gibi ayarlari etkiler.
- PPO'nun `dispatch_intensity` degeri dispatch icin makro butce/release sinyali verir.
- Joint modda PPO tek basina dogrudan dispatch yapmaz. `env_5pl`, continuous action'i `allow_dispatch=False` ile uygular.
- Asil dispatch/hold ve route/fleet/reorder taktik karari DQN tarafindan gelir.

Mixed stress benzeri durumda PPO'nun etkileri soyle okunur:

- kapasite buffer restore edilebilir;
- planned replenishment hesaplanabilir;
- arac nominal hizlari `speed_multiplier` ile ayarlanabilir;
- dispatch intensity, DQN'in dispatch yapmak istemesi halinde kac siparise kadar hareket edilecegini sinirlayan makro dispatch budget'a katkida bulunabilir.

## 7. DQN Rolu: Hierarchical_v1 Taktik Karar

DQN 0 ile 47 arasinda bir action secer. `hierarchical_v1` mimarisi bu 48 action'i tek duz liste gibi ezberlemek yerine dort baslikta dusunur:

- dispatch head: hold mu dispatch mi?
- route head: shortest, low_congestion, high_resilience?
- mode head: primary_fleet mi secondary_fleet mi?
- reorder head: none, conservative, aggressive, emergency?

Bu basliklar sonra tekrar external 48 Q degerine compose edilir. HOLD action'larda route ve mode bilesenlerinin anlamsiz etkisi maskelenir. Bu, eski flat action uzayindaki anlamsiz kombinasyon ceplerini azaltmak icin onemlidir.

## 8. Action 24 Decode: Matematiksel Acilim

`DiscreteActionMapper` 48 action'i su carpimdan olusturur:

`2 dispatch x 3 route x 2 fleet mode x 4 reorder = 48`

Kodun decode sirasi:

1. `reorder = action % 4`
2. `mode = (action // 4) % 2`
3. `route = (action // 8) % 3`
4. `dispatch = action // 24`

Action 24 icin:

- `24 % 4 = 0` -> reorder `none`
- `(24 // 4) % 2 = 6 % 2 = 0` -> mode `secondary_fleet`
- `(24 // 8) % 3 = 3 % 3 = 0` -> route `shortest`
- `24 // 24 = 1` -> dispatch `dispatch`

Bu yuzden action 24:

`dispatch + shortest route + secondary_fleet + no reorder`

Operasyon dilinde:

> "Simdi dispatch yap, en kisa rota ailesini kullan, secondary fleet tercih et, DQN tarafindan ek reorder override yapma."

## 9. Safety ve Feasibility Katmani

Action secildikten sonra sistem komutu hemen korumasiz uygulamaz.

Continuous safety:

- `speed_multiplier` min/max sinirlara clamp edilir;
- safety potential yuksekse dispatch intensity sinirlanir;
- capacity utilization cok yuksekse dispatch bloke edilebilir;
- backlog pressure yuksekse reorder fraction yukari cekilebilir.

Discrete safety:

- safety potential cok yuksekse dispatch action HOLD'a cevrilebilir;
- capacity utilization cok yuksekse dispatch HOLD'a cevrilebilir;
- backlog pressure yuksekse reorder NONE, AGGRESSIVE'a zorlanabilir.

Bu katmanin amaci modelin o an "agresif" bir karar secmesi halinde bile fiziksel/operasyonel kontrati korumaktir.

## 10. Action 24 Komutunun Simulatorde Uygulanmasi

Action 24 safety'den gecip degismeden kalirsa `env_5pl` tarafinda su mantik calisir:

1. PPO continuous action once uygulanir:
   - capacity state restore edilir;
   - planned replenishment hesaplanir;
   - arac nominal speed metadata'si guncellenir;
   - joint modda PPO dogrudan dispatch yapmaz.
2. DQN action decode edilir:
   - dispatch = DISPATCH;
   - route = SHORTEST;
   - mode = SECONDARY_FLEET;
   - reorder = NONE.
3. Hub discrete action:
   - reorder NONE oldugu icin DQN emergency/aggressive override siparisi yaratmaz.
4. Dispatch feasibility hesaplanir:
   - dispatchable order var mi?
   - preferred tier olan secondary vehicle var mi?
   - feasible dispatch ratio nedir?
   - macro dispatch budget PPO dispatch intensity ve feasibility ile uyumlu mu?
5. Vehicle discrete action:
   - dispatch pending orders cagrilir;
   - secondary fleet tercih edilir;
   - order icin shortest path secilir;
   - `simulation.evaluate_order_route` route feasibility kontrolu yapar;
   - feasible ise dispatch event log'a yazilir ve siparis dispatch edilir;
   - infeasible ise route failure, no vehicle, no unassigned veya already assigned gibi counter'lar artabilir.

## 11. Action 24 Aggregate Evidence

Equal-budget hierarchical eval'de `mixed_stress` icin action 24 su sekilde goruldu:

| Metrik | Deger |
| --- | ---: |
| Scenario service | `0.9423472160631124` |
| Scenario true lateness pressure | `0.00017783554585983237` |
| Scenario dispatch rate | `0.9866319444444445` |
| Scenario dispatch success per attempt | `0.9992932328922539` |
| Action 24 attempts | `3342` |
| Action 24 percentage | `0.5802083333333333` |
| Action 24 failed_noop | `0` |
| Action 24 no_current_work | `0` |
| Action 24 no_unassigned | `0` |
| Action 24 no_vehicle | `121` |
| Action 24 route_failure | `10` |
| Action 24 already_assigned | `2010` |
| Action 24 dispatch success ratio | `1.0` |

Bu tablo su yorumu destekler:

- action 24 mixed stress'te yogun kullaniliyor;
- bu yogunluk tek basina production blocker sayilmadi;
- cunku hard blocker zero, scenario PASS ve no-current/no-unassigned/failed-noop patlamasi yok;
- yine de no_vehicle, already_assigned ve route_failure counter'lari monitoring altinda tutulmali.

## 12. Reward ve Metrik Yorumlama

`env_5pl` step sonrasi reward ve info uretir. Eval tarafinda step-level raw trace saklanmadigi icin final artefaktlar daha cok episode ve scenario aggregation seviyesindedir.

Izlenen kritik metrikler:

- service level;
- true lateness pressure;
- dispatch rate;
- dispatch success per attempt;
- route failures;
- no vehicle;
- already assigned;
- no current work;
- no unassigned orders;
- failed noop dispatch;
- top action concentration;
- action 24/32 concentration.

Danisman icin onemli nokta:

> Modelin "dispatch yaptim" demesi tek basina yeterli degildir. Dispatch gercekten ise yaradi mi, rota feasible miydi, arac bulundu mu, siparis zaten atanmis miydi, no-work/fake-credit olustu mu gibi operational quality counter'lari ayrica kontrol edilir.

## 13. Monitoring: Action 24 Watch

Monitoring katmani action 24 icin su riskleri izler:

- action 24 concentration mevcut baseline'in uzerine materyal sekilde cikiyor mu?
- action 24 ile service dusuyor mu?
- action 24'te no-current-work gorunuyor mu?
- action 24'te no-unassigned gorunuyor mu?
- action 24'te failed-noop gorunuyor mu?
- route failures veya no_vehicle counter'lari artiyor mu?
- secondary_fleet ve reorder none davranisi gercek maliyet verisiyle celisiyor mu?

Accepted watch yorumu:

- equal-budget gate PASS;
- hard blockers zero;
- mixed stress service old production'a gore regresyon degil;
- action 24 route-side davranis public Amazon small sample ile sadece kismi olarak plausible;
- secondary_fleet ve no-reorder economics sirket verisi olmadan dogrulanmis sayilmaz.

## 14. Action 32 Route Disruption Karsilastirmasi

Action 32 icin decode:

- `32 % 4 = 0` -> reorder `none`
- `(32 // 4) % 2 = 8 % 2 = 0` -> mode `secondary_fleet`
- `(32 // 8) % 3 = 4 % 3 = 1` -> route `low_congestion`
- `32 // 24 = 1` -> dispatch `dispatch`

Action 32:

`dispatch + low_congestion route + secondary_fleet + no reorder`

Route disruption/congestion senaryosunda bu tercih operasyonel olarak mantiklidir:

- senaryoda route disruption probability ve congestion multiplier artar;
- shortest route artik en iyi operasyonel rota olmayabilir;
- `LOW_CONGESTION` route cost, distance uzerine congestion ve delay multiplier cezasi ekler;
- secondary fleet, primary fleet kisitli veya pahali oldugunda kapasite alternatifi olabilir;
- no reorder, route-only bir karar adiminda gereksiz inventory override yapmama anlamina gelebilir.

## 15. Action 32 Aggregate Evidence

Equal-budget hierarchical eval'de `route_disruption_congestion` icin action 32 su sekilde goruldu:

| Metrik | Deger |
| --- | ---: |
| Scenario service | `0.9007785656472469` |
| Scenario true lateness pressure | `0.008255604811047559` |
| Scenario dispatch rate | `0.49861111111111106` |
| Scenario dispatch success per attempt | `0.9993198529411765` |
| Scenario route failures | `8` |
| Scenario no_vehicle | `0` |
| Action 32 attempts | `2737` |
| Action 32 failed_noop | `0` |
| Action 32 no_current_work | `0` |
| Action 32 no_unassigned | `0` |
| Action 32 no_vehicle | `0` |
| Action 32 route_failure | `5` |
| Action 32 already_assigned | `1444` |
| Action 32 dispatch success ratio | `1.0` |

Old production comparator ayni equal-budget route-disruption senaryosunda:

- service `0.9044717143119803`;
- dispatch success per attempt `0.8615574751263317`;
- action 32 attempts `39`;
- action 32 dispatch success ratio `0.9743589743589743`.

Bu tablo su yorumu destekler:

- hierarchical model route disruption'da action 32'yi cok daha fazla kullaniyor;
- service old production'dan biraz dusuk, fakat gate toleransi icinde ve senaryo PASS;
- dispatch success per attempt ciddi sekilde daha yuksek;
- action 32'de no-current/no-unassigned/failed-noop/no_vehicle yok;
- route failure 5 ve already_assigned 1444 monitoring watch olarak kalmali.

## 16. Action 32 Risk Kosullari

Action 32 kabul edilebilir watch olmaktan cikip risk olurdu, eger:

- route disruption service senaryo threshold altina duserse;
- action 32 no-current-work uretirse;
- action 32 no-unassigned orders uretirse;
- action 32 failed-noop uretirse;
- low_congestion route secimi route failure'i materyal sekilde artirirsa;
- secondary fleet gercek maliyet/SLA verisiyle uyumsuz cikarsa;
- no reorder davranisi gercek stockout veya inventory cost verisiyle celisirse.

## 17. Simulator ile Gercek Dunya Siniri

Bu projenin simulator tarafi gucludur:

- autonomous closed loop vardir;
- observation/action contract sabittir;
- scenario suite ve gate sistemi vardir;
- production-like registry, manifest, monitoring ve ops bundle vardir.

Fakat gercek dunya siniri aciktir:

- gercek order stream yok;
- gercek TMS/WMS/ERP entegrasyonu yok;
- gercek carrier/fleet availability yok;
- gercek cost, inventory ve dispatch failure taxonomy yok;
- secondary_fleet ve reorder-none economics sirket verisiyle dogrulanmadi.

Guvenli thesis cumlesi:

> Sistem, 5PL digital twin icinde otonom karar verme ve production-like model yonetimi gosterir. Action 24 ve 32 davranislari simule stres senaryolarinda gate'lerden gecmis ve operasyonel blocker uretmemistir; ancak gercek sirket verisi olmadan bu aksiyonlarin ekonomik optimum oldugu iddia edilmez.

## 18. Danisman Icin Kisa Teknik Yorum

Bu tek operasyon ornegi, projenin sadece "bir model egittim" olmadigini gosterir. Modelin verdigi bir action:

- 73 boyutlu operasyonel ekrandan turetilir;
- PPO tarafinda 5 continuous control ile operasyon tonu ayarlanir;
- DQN tarafinda hierarchical 48 action uzayindan taktik karar secilir;
- mapper ile lojistik komuta donusur;
- safety/projector katmanindan gecer;
- digital twin'de stok, kapasite, dispatch, route, vehicle ve teslimat etkileriyle uygulanir;
- reward ve operational counter'larla denetlenir;
- monitoring tarafinda action-level watch olarak takip edilir.

Bu, tezde savunulabilecek ana mimari degerdir.

## 19. Hocama Nasil Anlatirim?

Danismana su sekilde anlatmak guvenlidir:

> Bu sistem bir lojistik kontrol kulesi gibi calisiyor. Her karar aninda simulator mevcut siparis, stok, filo, rota ve stres durumunu 73 sayilik bir observation olarak modele veriyor. PPO bes tane surekli ayarla operasyonun genel tonunu belirliyor. DQN ise 48 olasi taktik karar arasindan birini seciyor. Ornegin action 24 secilirse bu, "dispatch yap, shortest route kullan, secondary fleet kullan, reorder yapma" anlamina geliyor. Bu karar safety ve feasibility kontrollerinden geciyor, sonra env_5pl digital twin icinde uygulanıyor. Sonucta service, lateness, dispatch success, route failure, no vehicle ve no-work counter'lariyla karar denetleniyor. Bu yuzden model simulasyon icinde otonom karar veriyor, fakat gercek dunyada tam deployment icin sirket verisi ve TMS/WMS/ERP entegrasyonu gerekiyor.

Action 24 icin tek cumle:

> Mixed stress altinda model, gecikmeyi ve kapasite baskisini azaltmak icin secondary fleet ile shortest route dispatch kararini sik kullanmis; eval'de bu davranis no-current, no-unassigned veya failed-noop uretmedigi ve gate'i gectigi icin bug degil, izlenen residual watch olarak kabul edilmistir.

Action 32 icin tek cumle:

> Route disruption altinda model low_congestion route ailesini yogun kullanmis; bu route-side olarak mantiklidir, cunku en kisa rota congestion veya disruption yuzunden her zaman en iyi operasyonel tercih olmayabilir.

## 20. Overclaim Etmeme

Soylenmemesi gereken iddialar:

- "Bu action 24/32 gercek sirket operasyonunda optimumdur."
- "Model gercek TMS/WMS/ERP sisteminde otomatik dispatch yapiyor."
- "Bu rapordaki tek operasyon birebir loglanmis bir production step'idir."
- "PPO'nun ham numeric output'lari ve DQN logits bu dokumanda gosterildi."
- "Secondary fleet ve no-reorder economics gercek maliyet verisiyle kanitlandi."

Soylenebilecek guvenli iddialar:

- "Bu action semantigi source code ile dogrulanmistir."
- "Eval artefaktlari step-level raw trace degil, episode/scenario aggregate metrikleri saklar."
- "Action 24 ve 32 ornekleri kodla uyumlu temsili operasyon ornegidir."
- "Equal-budget residual-watch eval 8/8 PASS ve hard blockers zero sonucunu vermistir."
- "Sistem simulasyon icinde otonom karar loop'u yurutur."
- "Gercek dunyada deployment icin company data ve entegrasyon validasyonu gerekir."

## 21. Mini Glossary

| Terim | Sade anlam |
| --- | --- |
| `env_5pl` | Siparis, stok, rota, arac ve teslimatlari simule eden lojistik dunya |
| Observation 73 | Modelin karar aninda gordugu 73 sayilik operasyon ozeti |
| PPO | Bes surekli ayar ureten policy: stok, dispatch intensity, hiz, safety stock, capacity buffer |
| DQN | 48 taktik action arasindan birini secen policy |
| `hierarchical_v1` | DQN action kararini dispatch, route, fleet mode ve reorder basliklarina ayiran mimari |
| Action mapper | DQN'in sectigi action id'yi lojistik komuta ceviren kod |
| Action 24 | Dispatch + shortest route + secondary_fleet + no reorder |
| Action 32 | Dispatch + low_congestion route + secondary_fleet + no reorder |
| Safety projector | Tehlikeli veya fiziksel olarak uygun olmayan komutu sinirlayan/bloke eden katman |
| Feasibility | Dispatch icin is, arac, rota ve kapasite uygun mu kontrolu |
| Hard blocker | Fake dispatch credit, route failure credit, customer revisit, NaN/Inf gibi kabul edilemez hata siniflari |
| Residual watch | Gate'i engellemeyen ama production monitoring'de izlenecek davranis |
| Equal-budget eval | Old production ve hierarchical production'i ayni episode/seed/scenario butcesiyle karsilastiran adil eval |
| Simulation-to-real boundary | Simulator kanitinin gercek dunya deployment kaniti yerine gecmedigi sinir |
