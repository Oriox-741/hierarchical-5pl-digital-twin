# Tez Danışmanı İçin Türkçe Proje Kabiliyet Raporu

Tarih: 2026-06-12

Durum: Tez danışmanına sunulabilecek açıklama paketi. Bu doküman yalnızca dokümantasyon çalışmasıdır. Bu çalışma sırasında eğitim, offline eval, long-run gate, veri seti indirme, registry değişikliği, production değişikliği, baseline değişikliği, DB değişikliği, checkpoint değişikliği, mevcut eval çıktısı değişikliği, özel şirket verisi okuma/yazma veya training config oluşturma yapılmamıştır.

## 1. Yönetici Özeti

CODEX PROJE, 5PL lojistik için geliştirilmiş bir **digital twin tabanlı otonom karar kontrol prototipidir**. Proje gerçek bir şirket sistemine bağlı değildir; kararlarını simülasyon ortamında verir. Amaç, sipariş, stok, filo, rota ve yeniden sipariş gibi birbirine bağlı lojistik kararlarını tek tek değil, birlikte düşünen bir yapay zeka kontrol kulesi araştırmaktır.

Mevcut proje production modeli:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- runtime: `torch_joint`
- DQN architecture: `hierarchical_v1`
- initialization: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: `73`
- PPO continuous action dimension: `5`
- DQN external discrete action count: `48`
- gate status: 250k PASS, 500k PASS, 1M PASS
- equal-budget residual-watch gate: PASS
- ops bundle: `OPS_BUNDLE_CLEAN`

Tez danışmanına güvenli şekilde söylenebilecek ana iddia şudur:

> Bu proje, 5PL lojistik için simülasyon içinde çalışan otonom bir karar kontrol mimarisi gösterir. Sistem simüle edilmiş operasyonel durumu gözlemler, sürekli stratejik kontroller ve ayrık taktik lojistik kararları üretir, kendini farklı stres senaryolarında test eder ve onaylı modeli production-like bir MLOps süreciyle yönetir. Proje kendi simülasyon sözleşmesi içinde production-ready durumdadır; ancak gerçek şirket verisi, canlı TMS/WMS/ERP entegrasyonu ve gerçek filo/reorder ekonomisi doğrulanmadığı için henüz gerçek dünyada full autonomous deployment değildir.

## 2. Bu Proje Nedir?

Bu proje, bir **5PL lojistik kontrol kulesi prototipidir**. 5PL, tek bir taşıyıcıyı veya tek bir depoyu yönetmekten daha geniş bir seviyeyi ifade eder: siparişler, stoklar, taşıyıcılar, rotalar, kapasite, teslimat performansı ve maliyet kararları birlikte düşünülür.

Proje üç şeyi birleştirir:

1. **Digital twin simülasyonu:** Gerçek operasyonun sadeleştirilmiş ama kontrollü bir kopyası.
2. **Reinforcement learning modeli:** PPO ve DQN ile karar veren öğrenen ajan.
3. **Production-like yönetişim:** Registry, production manifest, monitoring, artifact health, rollback ve audit dokümantasyonu.

Bu nedenle proje yalnızca "rota optimizasyonu" değildir. Rota seçimi sistemdeki kararlardan sadece biridir. Model aynı anda dispatch/hold, route family, primary/secondary fleet ve reorder mode gibi kararları da ele alır.

## 3. Hangi Problemi Çözüyor?

Projenin araştırma problemi şudur:

> Bir reinforcement-learning kontrol kulesi, 5PL digital twin içinde stok, dispatch, rota, filo ve reorder kararlarını operasyonel güvenlik sınırları içinde koordine edebilir mi?

Simülasyon içinde sistem şu sorulara cevap arar:

- Sipariş hemen dispatch edilmeli mi, yoksa bekletilmeli mi?
- En kısa rota mı, düşük yoğunluklu rota mı, yoksa dayanıklılığı yüksek rota mı seçilmeli?
- Primary fleet mi, secondary fleet mi kullanılmalı?
- Reorder yapılmayacak mı, konservatif mi, agresif mi, emergency reorder mı seçilecek?
- Stok, kapasite ve hız gibi sürekli stratejik kontroller nasıl ayarlanmalı?
- Demand spike, yüksek holding cost, lead-time volatility, premium SLA pressure, route disruption, vehicle scarcity ve mixed stress altında model güvenli kalabiliyor mu?

## 4. Sistem Nasıl Çalışır?

Sistem bir döngü şeklinde çalışır:

1. Simülasyon ortamı mevcut lojistik durumunu oluşturur.
2. Bu durum 73 boyutlu bir observation vektörüne çevrilir.
3. PPO modeli 5 boyutlu sürekli stratejik kontrol üretir.
4. DQN modeli 0 ile 47 arasında bir taktik action seçer.
5. Action mapper bu sayıyı gerçek anlamlı lojistik kararlara açar.
6. Simülasyon bu kararı uygular: dispatch, rota, filo, reorder, stok ve teslimat sonuçları güncellenir.
7. Sistem service, lateness, dispatch success, route failure, no-work gibi metrikleri toplar.
8. Bölüm bitene kadar bu döngü tekrar eder.

Bu kapalı döngü, projenin simülasyon içindeki otonomluk temelidir.

## 5. Sense / Think / Act / Learn / Govern Açıklaması

Proje şu beş katmanla anlatılabilir:

### Sense

Sistem operasyonel dünyayı "hisseder". Sipariş durumu, stok, rota baskısı, araç/fleet kapasitesi, servis seviyesi, gecikme ve senaryo stresleri observation haline getirilir. Bu projenin sabit observation boyutu `73`'tür.

### Think

Model karar verir. PPO sürekli kontrolleri, DQN ise ayrık taktik kararları üretir. `hierarchical_v1` DQN mimarisi, 48 action'ı tek düz liste gibi ezberlemek yerine dispatch, route, fleet ve reorder bileşenlerine ayırarak düşünür.

### Act

Seçilen karar digital twin ortamında uygulanır. Yani simülasyon dispatch yapar veya bekletir, rota seçer, primary/secondary fleet kullanır, reorder modunu uygular ve sonuçları hesaplar.

### Learn

Eğitim sürecinde model senaryolardan geri bildirim alır. Bu projede eğitim basit bir sıfırdan öğrenme değildir; final model production parent, flat-teacher distillation, hierarchical architecture ve exact resume kullanarak kademeli şekilde güçlendirilmiştir.

### Govern

Model sadece eğitilip bırakılmaz. Candidate registration, active registry, production promotion, manifest, artifact health, monitoring report, company-data schema validator ve ops bundle ile yönetilir. Bu katman, araştırma prototipini production-like bir yapıya yaklaştırır.

## 6. Digital Twin Nedir ve Bu Projede Nasıl Kullanılır?

Digital twin, gerçek bir operasyonun simüle edilmiş temsilidir. Bu projede digital twin, 5PL lojistik ağını kontrollü bir ortamda temsil eder. Gerçek şirket sistemine bağlanmadan, "eğer model şu kararı verirse ne olur?" sorusunu güvenli şekilde test etmeye yarar.

Bu projedeki digital twin şunları simüle eder:

- stok ve safety stock dinamikleri,
- bekleyen, teslim edilen ve geç kalan siparişler,
- talep dalgalanması,
- premium ve urgent demand,
- primary ve secondary araçlar,
- dispatch feasibility,
- rota, congestion, disruption ve resilience,
- lead-time volatility,
- high holding cost pressure,
- vehicle scarcity,
- mixed stress.

Önemli sınır: Bu digital twin gerçek dünyaya benzer karar problemleri kurar, fakat gerçek şirket verisi olmadan gerçek operasyonun birebir ispatı değildir.

## 7. Otonomluk Seviyesi

### Simülasyonda otonom

Model simülasyon içinde kapalı döngü çalışır. Her step'te observation alır, action seçer, ortam bu action'ı uygular ve sonraki duruma geçer. İnsan her karar adımında müdahale etmez.

### Gerçek dünyada henüz full autonomous değil

Model şu anda gerçek TMS, WMS, ERP, carrier platform veya canlı operasyon sistemine bağlı değildir. Gerçek dispatch, gerçek reorder, gerçek araç ataması veya gerçek satın alma işlemi yapmaz. Bu yüzden doğru ifade:

- "Simülasyonda otonom karar verir."
- "Proje içinde production-promoted artifact vardır."
- "Gerçek dünyada full autonomous deployment henüz değildir."

## 8. PPO Ne Yapar?

PPO, sistemdeki **sürekli stratejik kontrolleri** üretir. Sürekli kontrol, 0/1 veya sabit sınıf seçimi değildir; belirli bir aralıkta değer alan ayarlardır. Örneğin kapasite, stok veya operasyonel duruş gibi simülatör içi fiziksel/stratejik ayarlar PPO tarafından temsil edilir.

Bu projede PPO continuous action dimension `5`'tir. Yani PPO aynı anda 5 adet sürekli kontrol değeri üretir.

Danışman için basit benzetme:

> PPO, kontrol kulesinin gaz/fren/denge ayarları gibidir. Sistemin genel operasyonel tonunu ayarlar.

## 9. DQN Ne Yapar?

DQN, sistemdeki **ayrık taktik kararı** seçer. Ayrık karar, sınırlı sayıdaki seçeneklerden birini seçmektir. Bu projede DQN 0 ile 47 arasında bir action seçer.

Bu action şunların birleşimidir:

- dispatch veya hold,
- route choice,
- fleet mode,
- reorder mode.

Danışman için basit benzetme:

> DQN, "şimdi hangi operasyonel hamleyi yapalım?" sorusuna cevap verir.

## 10. Neden PPO Action Dim 5 ama DQN Action Count 48?

Çünkü PPO ve DQN farklı tipte karar verir.

PPO'nun 5 boyutlu sürekli action'ı, sayısal ayar gibidir. Örneğin bir mikser panelinde 5 kaydırıcı düşünün.

DQN'in 48 action'ı ise menü seçimi gibidir. Bu 48 action şu çarpımdan gelir:

`2 dispatch choices x 3 route choices x 2 fleet choices x 4 reorder choices = 48`

Yani DQN tek bir sayı seçer, ama o sayı aslında dört parçalı bir lojistik karar kombinasyonunu temsil eder.

## 11. Action 24 ve Action 32 Ne Demek?

Action 24 ve action 32, residual watch olarak izlenen iki taktik action'dır. Bunlar hata olarak kapatılmamıştır; gate'lerden geçtikleri için kabul edilmiş ama monitoring altında tutulacak davranışlardır.

- **Action 24:** dispatch + shortest route + secondary fleet + no reorder
- **Action 32:** dispatch + low-congestion route + secondary fleet + no reorder

Bu action'ların bazı stres senaryolarında yoğunlaşması gözlenmiştir. Equal-budget residual-watch değerlendirmesinde bu yoğunlaşmalar hard blocker, no-current/no-unassigned veya failed-noop patlamasıyla birleşmediği için production için engel sayılmamıştır. Ancak gerçek şirket verisi olmadan secondary fleet ve no-reorder ekonomisinin doğru olduğu iddia edilmez.

## 12. Current Production Model Nedir?

Mevcut production model:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- production directory: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- runtime: `torch_joint`
- architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- exact resume capable: true
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: `73`
- PPO continuous action dimension: `5`
- DQN external discrete action count: `48`
- training step: `1000000`
- eval verdict: PASS
- hard blocker status: zero
- long-run gate verdict: PASS
- residual watches accepted: true

Aktif registry ID'leri:

- PPO active: `17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- DQN active: `f87e10d6-479f-44fc-99d1-6925bc9cb346`

Candidate registry ID'leri:

- PPO candidate: `3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- DQN candidate: `71af6701-ac3a-40f2-bdd9-cc80868d5a1c`

## 13. 1M Step Neden Yeterli Oldu?

1M step'in yeterli olmasının sebebi modelin her şeyi sıfırdan ve tek başına öğrenmesi değildir. Tam tersine, final model daha önceki başarılı ve başarısız denemelerden çıkan derslerin üzerine kurulmuştur.

1M başarıyı mümkün kılan faktörler:

- Daha önce 8/8 senaryo geçen flat production parent vardı.
- Reward ve hard-blocker semantiği önceki döngülerde düzeltilmişti.
- Observation/action contract sabitti: obs 73, action 48.
- `hierarchical_v1`, 48 action'ı dispatch/route/fleet/reorder bileşenlerine ayırdı.
- `flat_teacher_distillation_v1`, eski flat production davranışını yeni hierarchical modele warm-start olarak aktardı.
- 500k ve 1M continuation exact resume ile yapıldı; replay buffer ve RNG state korunarak devam edildi.
- Curriculum, aynı finite scenario family içinde tekrarlı ve hedefli exposure sağladı.
- Gate'ler önceki kötü çözümleri yakalayacak kadar güçlü hale gelmişti.

Bu yüzden doğru yorum şudur:

> 1M step, bu bounded simulator/eval contract içinde yeterli kanıt üretti. Bu, modelin tüm lojistiği sıfırdan ve evrensel olarak öğrendiği anlamına gelmez.

## 14. Hangi Testlerden Geçti?

Hierarchical v1 ladder:

- 250k: PASS
- 500k: PASS
- 1M: PASS

1M değerlendirme:

- 8/8 scenario PASS
- 160 episode rows
- hard blockers: zero
- long-run gate: PASS
- equal-budget residual-watch gate: PASS
- ops bundle: `OPS_BUNDLE_CLEAN`

Equal-budget comparison sonuçları:

| Scenario | Old production service/success | Hierarchical 1M service/success | Sonuç |
| --- | ---: | ---: | --- |
| `baseline_normal` | 0.945 / 0.998 | 0.990 / 1.000 | PASS |
| `demand_spike_volatility` | 0.840 / 1.000 | 0.859 / 1.000 | PASS |
| `high_holding_cost` | 0.931 / 0.997 | 0.966 / 0.999 | PASS |
| `lead_time_volatility` | 0.914 / 0.998 | 1.000 / 0.999 | PASS |
| `mixed_stress` | 0.940 / 0.997 | 0.942 / 0.999 | PASS |
| `premium_sla_pressure` | 0.982 / 0.759 | 1.000 / 0.999 | PASS |
| `route_disruption_congestion` | 0.904 / 0.862 | 0.901 / 0.999 | PASS |
| `vehicle_scarcity_capacity_shock` | 0.908 / 0.998 | 0.949 / 1.000 | PASS |

## 15. Gerçek Dünyaya Ne Kadar Yakın?

Proje gerçek operasyonlara **yapı olarak yakın**, fakat **deployment olarak gerçek dünyada çalışıyor değildir**.

Yakın olduğu yönler:

- service, lateness, inventory, holding cost, route disruption, vehicle scarcity, premium SLA pressure ve demand spike gibi gerçek lojistik kavramlarını içerir.
- production-like registry, manifest, monitoring, artifact health ve rollback belgeleri vardır.
- Amazon Last Mile small sample ile route-side plausibility için sınırlı public-data desteği vardır.

Eksik kalan yönler:

- gerçek order stream,
- gerçek dispatch attempts,
- delivery outcomes,
- carrier/fleet availability,
- inventory/reorder events,
- gerçek cost bilgisi,
- gerçek failure taxonomy,
- TMS/WMS/ERP entegrasyon semantiği.

Bu yüzden doğru ifade:

> Proje, gerçek dünyaya geçiş için güçlü bir simulator-production prototype'tır. Ancak real-world autonomous logistics claim için company data ve entegrasyon doğrulaması gerekir.

## 16. Ne Yapabiliyor?

Proje şu anda şunları yapabiliyor:

1. 5PL digital twin içinde otonom policy loop çalıştırmak.
2. 73 boyutlu observation üretmek.
3. PPO ile 5 continuous control üretmek.
4. DQN ile 48 taktik action'dan birini seçmek.
5. Action'ı dispatch/route/fleet/reorder bileşenlerine çözmek.
6. Sekiz stres senaryosunda model davranışını ölçmek.
7. Hard blocker, service, lateness, dispatch success ve action-quality kontrolleri yapmak.
8. Candidate registration, active registry ve copy-only production promotion yürütmek.
9. Production manifest ve SHA256 hash'leriyle artifact izlenebilirliği sağlamak.
10. Artifact health, synthetic monitoring, synthetic company-data schema ve ops bundle raporları üretmek.
11. Route-choice tarafında public Amazon Last Mile örneğiyle kısmi plausibility çalışması yapmak.

## 17. Ne Yapamıyor?

Proje şu anda şunları yapmıyor:

1. Canlı TMS/WMS/ERP sistemine bağlanmıyor.
2. Gerçek dispatch, reorder, vehicle assignment veya satın alma yürütmüyor.
3. Private company data ingest etmiyor.
4. Gerçek dünya lojistiğinde evrensel optimum kanıtlamıyor.
5. Secondary fleet economics veya reorder-none economics'i gerçek veriyle doğrulamıyor.
6. Human operational approval yerine geçmiyor.
7. Blind 3M/5M/10M training'i haklı çıkarmıyor.
8. Peer-reviewed SOTA iddiası taşımıyor.

## 18. Akademik Dünyadaki Yeri

Proje şu alanların kesişimindedir:

- 5PL digital twin,
- reinforcement learning for logistics control,
- hierarchical/factorized action-space RL,
- multi-objective reward design,
- simulation-to-real calibration,
- autonomous decision systems için safe MLOps.

Akademik katkı, tek bir algoritma iddiasından çok sistemsel entegrasyondadır: simülasyon ortamı, PPO+DQN joint policy, hierarchical action decomposition, exact resume, scenario-based gates ve production governance aynı repo içinde izlenebilir şekilde birleştirilmiştir.

SOTA iddiası için hâlâ benchmark, ablation, multiple seeds, company-data validation ve external reproducibility gerekir.

## 19. Ürünleşme Açısından Durumu

Proje ürünleşmeye yaklaşan bazı parçalar içerir:

- `PolicyService` ile runtime decision path,
- active registry,
- production artifact directory,
- production manifest,
- rollback dokümantasyonu,
- artifact health report,
- monitoring report generator,
- company-data intake validator,
- ops bundle.

Ancak ürünleşme tamamlanmış değildir. Ürünleşme için ayrıca gerekir:

- API/service wrapper,
- operator dashboard,
- health endpoint,
- live telemetry ingestion,
- privacy/schema approval,
- shadow-mode testing,
- gerçek entegrasyon testleri.

## 20. Riskler ve Sınırlamalar

| Risk | Mevcut durum | Önlem |
| --- | --- | --- |
| Action 24/32 concentration | Accepted residual watch | Action rate ve failure counter monitoring |
| Simulation-to-real gap | Açık | Company data calibration |
| Secondary-fleet economics | Kanıtlanmadı | Carrier/fleet/cost data gerekli |
| Reorder-none economics | Kanıtlanmadı | Inventory/reorder/cost data gerekli |
| No live TMS/WMS integration | Yok | Gelecek productization layer |
| Finite scenario suite | 8 senaryo + equal-budget assurance | Yeni senaryo yalnızca kanıtla eklenmeli |
| Complex simulator/trainer | High complexity | Test, dokümantasyon ve no-blind-training governance |
| Blind longer training | Haklı değil | Trigger-based 1.5M/2M planı dışında önerilmez |

## 21. Gelecek Yol Haritası

Kısa vadede:

- Current hierarchical v1 production state korunmalı.
- Monitoring ve ops bundle düzenli kullanılmalı.
- Action 24/32 residual watches izlenmeli.
- Tez danışmanı için bu Türkçe paket kullanılmalı.

Veriye bağlı adımlar:

- Company data request package ile veri alanları netleştirilmeli.
- Synthetic schema validator gerçek veri gelmeden önce kontratı doğrulamak için kullanılmalı.
- Company KPI'ları simülasyon metriklerine map edilmeli.
- Fleet, route, inventory, dispatch failure ve cost kalibrasyonu yapılmalı.

Araştırmaya bağlı adımlar:

- Sadece gerçek monitoring regression veya company-data mismatch çıkarsa 1.5M/2M exact-resume extension planı düşünülmeli.
- Blind 3M/5M/10M/100M training başlatılmamalı.
- Hierarchical v2 ancak residual watch'lar operasyonel risk haline gelirse gündeme alınmalı.

## 22. Tez Danışmanına Söylenecek Güvenli İddia

> Bu proje, 5PL lojistik için digital twin içinde çalışan, PPO+DQN tabanlı ve hierarchical DQN action mimarisiyle güçlendirilmiş otonom karar kontrol prototipidir. Model, simülasyon içinde sipariş, stok, rota, filo ve reorder kararlarını birlikte değerlendirir; 250k, 500k ve 1M gate'lerinden geçmiştir; 8/8 senaryoda PASS almıştır; hard blocker üretmemiştir; production-like registry, manifest, monitoring ve ops bundle ile yönetilmektedir. Bu sonuç, proje sözleşmesi içinde simulator-production readiness gösterir. Ancak gerçek dünya deployment readiness için company data, canlı TMS/WMS/ERP entegrasyonu ve gerçek fleet/reorder/cost doğrulaması hâlâ gereklidir.

