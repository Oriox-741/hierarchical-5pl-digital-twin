# 5PL Decision-Control Icin SOTA Pathway Raporu

Tarih: 2026-06-13

Durum: Danisman/tez yorumu icin planlama raporu. Bu rapor SOTA iddiasi yapmaz; SOTA iddiasi icin ne gerektigini tanimlar.

## Kisa Sonuc

Mevcut proje guclu bir 5PL digital-twin decision-control sistemidir, fakat klasik anlamda route SOTA projesi degildir. SOTA demek, belirli bir benchmark, belirli veri seti, belirli metrik ve belirli protokol altinda bilinen en iyi veya en ust seviye sonuc demektir. Bu yuzden "biz SOTA'yiz" demek su anda dogru degildir.

Daha guvenli iddia:

> Bu proje, 5PL dijital ikizinde hiyerarsik PPO+DQN karar kontrol sisteminin simulator-production seviyesinde guvenilir hale getirilebilecegini gostermistir; SOTA benzeri iddia icin acik benchmark protokolu, guclu baseline ailesi ve kamuya tekrarlanabilir karsilastirma gerekir.

## SOTA Ne Demektir?

SOTA, "state of the art" demektir ama genel bir ovgu sozu degildir. Teknik olarak:

- hangi benchmark?
- hangi veri seti?
- hangi metrik?
- hangi butce?
- hangi rakip baseline'lar?
- hangi istatistiksel guven araligi?
- hangi tekrar edilebilir protokol?

sorulari cevaplanmadan SOTA iddiasi kurulamaz.

Ornek:

- Solomon VRPTW distance objective icin SOTA ayri bir iddiadir.
- CVRPLIB total distance/vehicle objective icin SOTA ayri bir iddiadir.
- Inventory benchmark cost/fill-rate SOTA ayri bir iddiadir.
- 5PL digital-twin service/dispatch/fleet/reorder decision-control SOTA ise baska ve daha genis bir iddiadir.

## Neden Mevcut Proje Route SOTA Degil?

Mevcut proje route solver olarak tasarlanmadi. OR-Tools benchmarklari ve Amazon route proxy analizleri route tarafini destekliyor, fakat modelin ana hedefi:

- yalnizca mesafe minimize etmek degil;
- yalnizca VRPTW instance objective'i optimize etmek degil;
- yalnizca route plan uretmek degil.

Ana hedef 5PL karar kontrolu:

- dispatch/hold;
- route family secimi;
- primary/secondary fleet secimi;
- reorder posture;
- PPO continuous stok/kapasite/hiz ayarlari;
- service, lateness, dispatch success, no-work, no-vehicle ve route-failure guard'lari.

Bu nedenle "OR-Tools'tan iyi" veya "VRP SOTA" demek overclaim olur.

## Mevcut Projenin Guclu Pozisyonu

Mevcut kanitlar sunlari destekler:

- simulator-production readiness guclu;
- hierarchical_v1 1M model eski production'a gore esit-butce gate'te basarili;
- multi-seed robustness temiz;
- runtime latency cok iyi;
- Amazon public data route-side plausibility sagliyor;
- OR-Tools route suite klasik route literaturune bag kuruyor;
- rule-based baselines var, ama continuous-aware hale getirilmeli;
- monitoring/ops/audit zinciri guclu.

Bu pozisyon SOTA degil, ama tez ve prototip icin degerli:

> "Yeni bir 5PL decision-control benchmark ailesi ve guvenli model lifecycle'i icin temel altyapi."

## 5PL Decision-Control Benchmark Konumlandirmasi

Proje SOTA iddiasi yerine su sekilde konumlandirilabilir:

1. Klasik VRP benchmark projesi degil.
2. 5PL operasyonel karar kontrol sistemi.
3. Rota, dispatch, fleet, inventory/reorder ve governance kararlarini birlikte simule ediyor.
4. Hiyerarsik DQN action factorization, flat-teacher distillation ve exact-resume training discipline ile gelistirildi.
5. Benchmark paketi route-only, inventory/fleet analoglari ve simulator-production gate'leriyle genisletilebilir.

Bu daha savunulabilir bir akademik cumledir:

> Bu calisma, route-only benchmarklarin otesinde, 5PL digital twin uzerinde cok bilesenli bir karar kontrol problemini tanimlayip olcen bir sistem sunar.

## Legitimate SOTA-Like Iddia Icin Gerekenler

### 1. Acik benchmark protokolu

Gerekli:

- sabit scenario seti;
- sabit seed listesi;
- sabit episode budget;
- sabit obs/action kontrati;
- sabit output schema;
- raw episode metrics;
- denominator semantics;
- hard blocker tanimlari;
- statistical confidence method.

### 2. Guclu baseline ailesi

Gerekli baseline'lar:

- neutral continuous rule baseline;
- heuristic continuous rule baseline;
- PPO-assisted rule tactical baseline;
- inventory/reorder heuristics;
- fleet dispatch heuristics;
- OR route solvers;
- PyVRP/OR-Tools route references;
- possibly external inventory/fleet analog benchmarks.

### 3. Esit ve adil karsilastirma

Her karsilastirmada:

- ayni data;
- ayni senaryo;
- ayni seed;
- ayni butce;
- ayni metric;
- ayni output reporting;
- raw count degil normalize rate;
- istatistiksel test veya bootstrap/permutation.

### 4. Public reproducibility

SOTA benzeri iddia icin:

- kod;
- benchmark config;
- data veya public synthetic generator;
- exact commands;
- dependency versions;
- artifacts/hashes;
- independent reviewer evidence.

### 5. Sirket verisi ile ayri business validation

5PL icin SOTA benzeri simulator iddiasi bile gercek business validation'dan ayridir. Sirket verisi olmadan:

- secondary_fleet cost;
- carrier acceptance;
- reorder economics;
- holding/stockout finance;
- dispatch failure taxonomy;
- live TMS/WMS/ERP constraints

tam dogrulanamaz.

## SOTA Pathway Roadmap

### Faz A - Native simulator fairness

- Continuous-aware rule baseline skeleton.
- Heuristic continuous modes.
- PPO-assisted diagnostic mode.
- Fresh benchmark only after explicit approval.

Amac: mevcut 74/100 rule-based benchmark caveat'ini azaltmak.

### Faz B - Non-route external evidence

- OR-Gym or MABIM inventory/reorder preflight.
- FleetPy or OpenMines dispatch/fleet preflight.
- No install/download without approval.

Amac: rota disi karar ailelerini dis benchmarklarla tartismak.

### Faz C - Route uncertainty strengthening

- SVRPBench preflight for stochastic route uncertainty.
- PyVRP stronger route reference.

Amac: action 32 low-congestion/reliability tarafini daha guclu route benchmarklarla desteklemek.

### Faz D - 5PL benchmark publication package

- Public scenario definitions.
- Baseline suite.
- Scoring card.
- Statistical protocol.
- Reproducible scripts.
- Human-readable thesis report.

Amac: "5PL decision-control benchmark" diye yeni ve acik bir benchmark ailesi onerilebilir hale gelmek.

## Danismana Guvenli Cumleler

- "Bu proje route SOTA iddiasi yapmiyor; 5PL digital twin decision-control sistemi olarak degerlendiriliyor."
- "Mevcut benchmarklar simulator-production hazirligini guclendiriyor, gercek dunya validasyonu icin sirket verisi gerekiyor."
- "SOTA benzeri iddia icin benchmark-specific protokol ve dis baseline ailesi gerekir."
- "Continuous-aware rule baselines, mevcut rule benchmark caveat'ini kapatmak icin en dogru sonraki adimdir."

## Overclaim Uyarilari

Soylenmemeli:

- "Bu model VRP SOTA."
- "OR-Tools'tan daha iyi."
- "Amazon verisi tum sistemi dogruladi."
- "Company data olmadan real-world deployment tamamen kanitlandi."
- "Rule-based benchmark full PPO+DQN-equivalent comparator'dir."

Soylenmeli:

- "Mevcut benchmark paketi simulator-production kanitini guclendiriyor."
- "Route-side public evidence var, ama fleet/reorder/cost icin company data veya non-route benchmarklar gerekiyor."
- "SOTA benzeri iddia icin daha acik, benchmark-specific ve tekrarlanabilir protokol kurulacak."

## Son Karar

Mevcut proje icin en saglam akademik pozisyon:

> 5PL decision-control icin guclu bir simulator-production model ve benchmark altyapisi kuruldu; bundan sonraki bilimsel guclendirme, continuous-aware baselines ve rota disi benchmark aileleriyle adil karsilastirma kapsamlarini genisletmektir.

Final rapor siniflandirmasi: `SOTA_PATHWAY_DEFINED_NO_SOTA_CLAIM`
