# Tez Danışmanı İçin Türkçe Terimler Sözlüğü

Tarih: 2026-06-12

Bu sözlük, CODEX PROJE anlatılırken geçen teknik terimleri Türkçe ve danışman-dostu şekilde açıklar.

## 5PL

Fifth-party logistics. Tek bir taşıyıcıyı veya depoyu yönetmekten daha üst seviyede, farklı lojistik taraflarını ve karar katmanlarını koordine eden kontrol kulesi yaklaşımıdır. Bu projede 5PL; sipariş, stok, filo, rota, kapasite ve reorder kararlarının birlikte düşünülmesi anlamına gelir.

## Digital Twin

Gerçek bir operasyonun simüle edilmiş temsilidir. Bu projede digital twin, gerçek şirkete zarar vermeden lojistik kararlarını denemek için kullanılan 5PL simülasyon ortamıdır.

## Control Tower

Operasyonu yukarıdan izleyen ve koordine eden karar merkezi. Bu projede control tower, yapay zekanın sipariş, stok, rota, filo ve reorder kararlarını birlikte değerlendiren üst karar katmanını ifade eder.

## Reinforcement Learning

Bir ajanın, ortamla etkileşerek ödül/ceza sinyallerinden karar vermeyi öğrenmesidir. Bu projede ajan, digital twin içinde lojistik kararları verir ve service, lateness, dispatch success gibi metriklerden geri bildirim alır.

## PPO

Proximal Policy Optimization. Bu projede PPO, sürekli stratejik kontrol değerlerini üretir. PPO'nun action dimension'ı `5`'tir. Bunu operasyonun genel ayarlarını belirleyen bir kontrol paneli gibi düşünebiliriz.

## DQN

Deep Q-Network. Bu projede DQN, ayrık taktik action seçer. DQN 0 ile 47 arasında bir action üretir; bu action dispatch, route, fleet ve reorder kararlarının birleşimidir.

## Hierarchical DQN

DQN'in action'ları tek düz liste olarak değil, bileşenlere ayrılmış şekilde düşünmesidir. Bu projede `hierarchical_v1`, dispatch, route, fleet mode ve reorder heads kullanır; dışarıya yine 48 action uyumlu Q değeri verir.

## Observation

Modelin gördüğü durum vektörüdür. Bu projede observation dimension `73`'tür. Yani model her karar anında 73 özellikten oluşan bir state representation alır.

## Action

Modelin verdiği karardır. PPO için action sürekli değerlerden oluşur; DQN için action 0..47 arasında ayrık bir seçimdir.

## Continuous Action

Sürekli aralıkta değer alan karar. Örneğin -1 ile 1 arasında ayarlanabilen bir kontrol gibi düşünülebilir. Bu projede PPO 5 continuous action üretir.

## Discrete Action

Sınırlı seçeneklerden birini seçme kararıdır. Bu projede DQN 48 discrete action'dan birini seçer.

## Reward

Modelin öğrenmesini yönlendiren geri bildirim sinyalidir. İyi service, düşük lateness, başarılı dispatch ve güvenli operasyon ödülle desteklenir; kötü davranışlar cezalandırılır.

## Curriculum Learning

Modelin eğitim sırasında senaryoları belirli bir sıra veya karışımla görmesidir. Bu projede baseline, demand spike, high holding cost, lead-time volatility, premium SLA pressure, route disruption, vehicle scarcity ve mixed stress gibi senaryolar curriculum içinde kullanılmıştır.

## Exact Resume

Eğitimi kaldığı yerden gerçekten devam ettirme yöntemidir. Sadece model ağırlıkları değil, optimizer state, global step, replay buffer ve RNG state de geri yüklenir. Bu projede 500k ve 1M continuation için exact resume kritik olmuştur.

## Replay Buffer

DQN'in geçmiş deneyimlerini tuttuğu hafızadır. Model daha önce gördüğü state/action/reward/next-state geçişlerinden tekrar öğrenebilir. Replay buffer boş devam ederse continuation davranışı bozulabilir.

## Checkpoint

Modelin belirli bir andaki kaydedilmiş durumudur. Ağırlıklar, metadata ve exact resume için gerekli state bilgilerini içerebilir.

## Long-Run Gate

Bir modelin uzun koşu değerlendirme sonuçlarını production karşılaştırması ve threshold'larla kontrol eden gate mekanizmasıdır. PASS almayan model bir sonraki aşamaya geçmez.

## Hard Blocker

Ortalama performans iyi görünse bile modeli reddettirecek ciddi hata durumudur. Örneğin açıkça faydasız veya fiziksel gerçekliğe aykırı kararlar hard blocker olabilir.

## Residual Watch

Gate'leri geçmesine rağmen izlenmesi gereken davranıştır. Bu projede action 24 ve action 32 concentration residual watch olarak kabul edilmiştir. Yani "şimdilik engel değil, ama monitoring altında kalmalı" demektir.

## Artifact Health

Production artifact'larının sağlıklı ve tutarlı olup olmadığını kontrol eden raporlama yaklaşımıdır. Registry, manifest, hash, dosya varlığı ve process scan gibi kontroller içerir.

## Registry

Hangi modelin candidate veya active olduğunu kaydeden yerel model kayıt sistemidir. Bu projede `models/registry/active_models.json` ve `models/registry/models.jsonl` registry kanıtlarıdır.

## Production Promotion

Onaylı model artifact'larının production dizinine taşınması/kopyalanması ve manifest ile kayıt altına alınmasıdır. Bu projede hierarchical v1 1M model copy-only production promotion ile production dizinine alınmıştır.

## TMS/WMS/ERP

- TMS: Transportation Management System, taşıma operasyonlarını yönetir.
- WMS: Warehouse Management System, depo operasyonlarını yönetir.
- ERP: Enterprise Resource Planning, şirket kaynak planlama sistemidir.

Bu proje henüz bu sistemlere canlı bağlı değildir.

## Simulation-to-Real Gap

Simülasyonda iyi çalışan bir modelin gerçek dünyada aynı şekilde çalışacağının garanti olmaması problemidir. Gerçek veri, entegrasyon, maliyet, operasyonel istisnalar ve insan süreçleri bu gap'i oluşturur.

## OTIF

On Time In Full. Siparişin zamanında ve tam teslim edilmesini ölçen lojistik performans metriğidir. Bu projede doğrudan OTIF etiketi her yerde kullanılmasa da service/lateness/dispatch success metrikleri aynı operasyonel kalite ailesine yakındır.

## Secondary Fleet

Primary fleet dışında kullanılan alternatif/ikincil taşıma kapasitesidir. Esnek kapasite sağlayabilir ama gerçek dünyada maliyet, güvenilirlik ve availability açısından company data ile doğrulanması gerekir.

## Reorder Economics

Yeniden sipariş kararlarının maliyet ve fayda dengesidir. Ne zaman sipariş verileceği, ne kadar stok tutulacağı, emergency reorder'ın maliyeti ve reorder-none kararının riski gerçek şirket verisi olmadan tam doğrulanamaz.

## Action 24

Bu projede action 24 şu kombinasyonu temsil eder:

dispatch + shortest route + secondary fleet + no reorder

Mixed stress senaryosunda yoğunlaşması izlenen residual watch'lardan biridir.

## Action 32

Bu projede action 32 şu kombinasyonu temsil eder:

dispatch + low-congestion route + secondary fleet + no reorder

Route disruption senaryosunda yoğunlaşması izlenen residual watch'lardan biridir.

## Simulator-Production-Ready

Modelin proje içindeki simülasyon contract'ı, gate'ler, production manifest, registry ve monitoring kuralları açısından kabul edilebilir durumda olmasıdır. Bu, gerçek dünyada canlı deploy edildiği anlamına gelmez.

## Real-World Autonomous

Modelin gerçek şirket sistemlerine bağlı şekilde gerçek operasyon kararlarını güvenli ve doğrulanmış biçimde otomatik yürütmesidir. Bu proje henüz bu aşamada değildir.

