# Tez Danışmanı İçin Sade Türkçe Özet

Tarih: 2026-06-12

## Bu Proje Tek Cümleyle Nedir?

Bu sistem, lojistik operasyonlarını simülasyon içinde yöneten bir **yapay zeka kontrol kulesi prototipidir**.

Bir havaalanı kontrol kulesi uçakların kalkışını, inişini ve pist kullanımını koordine eder. Bu proje de benzer şekilde lojistik dünyasında sipariş, stok, araç, rota ve yeniden sipariş kararlarını birlikte düşünmeye çalışır. Farkı şudur: şu anda gerçek şirket operasyonuna bağlı değildir; kararları bir digital twin simülasyonu içinde verir.

## En Basit Anlatım

Proje şunu yapar:

1. Simüle edilmiş bir lojistik dünyası kurar.
2. Bu dünyada siparişler, stoklar, araçlar, rotalar ve stres durumları vardır.
3. Yapay zeka mevcut durumu okur.
4. "Şimdi ne yapmalıyım?" diye karar verir.
5. Karar simülasyonda uygulanır.
6. Sistem iyi mi kötü mü performans verdiğini ölçer.

Eğitim sırasında bu döngüden toplanan deneyimler modelin simülasyon içindeki lojistik kararlarını daha iyi vermeyi öğrenmesini sağlar. Eval veya production-like kullanım sırasında ise model yeni öğrenme yapmaz; o ana kadar öğrenilmiş policy ile karar verir.

## Gerçek Şirket Sistemine Bağlı mı?

Hayır. Bu proje şu anda gerçek TMS, WMS, ERP, taşıyıcı sistemi veya canlı şirket operasyonuna bağlı değildir.

Bu çok önemli bir sınırdır:

- Simülasyonda otonom karar verir.
- Proje içinde production-like şekilde yönetilir.
- Ama gerçek dünyada otomatik dispatch, otomatik reorder veya gerçek araç ataması yapmaz.

## Hangi Kararları Birlikte Düşünür?

Model tek bir rota seçme problemi çözmez. Birden fazla karar katmanını birlikte düşünür:

- Siparişi şimdi dispatch et mi, beklet mi?
- Hangi rota tipi seçilsin?
- Primary fleet mi, secondary fleet mi kullanılsın?
- Reorder yapılmasın mı, konservatif mi, agresif mi, emergency mi olsun?
- Stok ve kapasite baskısı nasıl dengelensin?

Bu yüzden proje basit bir route optimizer değildir. Daha çok bir lojistik karar orkestrasyon prototipidir.

## Digital Twin Ne Demek?

Digital twin, gerçek bir sistemin simüle edilmiş kopyasıdır.

Basit benzetme:

> Uçuş simülatörü pilotların gerçek uçağa binmeden pratik yapmasını sağlar. Bu projedeki digital twin de lojistik kararlarının gerçek şirkete zarar vermeden denenmesini sağlar.

Bu projede digital twin içinde şunlar bulunur:

- siparişler,
- stok,
- araç kapasitesi,
- rota yoğunluğu,
- gecikme,
- talep artışı,
- premium müşteri baskısı,
- araç kıtlığı,
- rota bozulması,
- karma stres senaryoları.

## Model Nasıl Karar Veriyor?

Model iki parçalı düşünür:

### PPO

PPO sürekli ayarları üretir. Bunu arabanın gaz/fren/direksiyon hassasiyeti gibi düşünebiliriz. Bu projede PPO 5 adet sürekli kontrol değeri üretir.

### DQN

DQN taktik seçim yapar. Menüden seçenek seçmek gibidir. Bu projede DQN 48 farklı lojistik action'dan birini seçer.

Bu 48 action şu kombinasyondan gelir:

`2 dispatch seçeneği x 3 rota seçeneği x 2 filo seçeneği x 4 reorder seçeneği = 48`

## Current Production Model Ne Durumda?

Mevcut model:

`joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Bu model proje içinde production-promoted durumdadır. Yani model dosyaları, registry, manifest ve monitoring belgeleriyle production-like şekilde yönetilmektedir.

Ancak bu "gerçek şirkette canlıya alındı" anlamına gelmez. Doğru anlamı:

> Proje ortamında onaylı production artifact olarak yönetiliyor.

## 1M Step Başarısı Ne Anlama Geliyor?

Model 1M step'te başarılı oldu, ama bu "tüm lojistiği sıfırdan öğrendi" demek değildir.

Daha doğru açıklama:

- Önceden başarılı bir production parent vardı.
- Model flat teacher distillation ile warm-start aldı.
- DQN action space hierarchical hale getirildi.
- 250k, 500k ve 1M gate'leri sırasıyla geçti.
- 500k ve 1M devamları exact resume ile yapıldı.
- Replay buffer ve RNG state korundu.
- Aynı simülasyon sözleşmesi içinde 8 senaryo test edildi.

Yani 1M step, iyi tasarlanmış ve sınırları belli bir simülasyon problemi için yeterli oldu. Gerçek dünyadaki tüm lojistik problemleri için evrensel kanıt değildir.

## Hangi Testlerden Geçti?

Model şu kanıtlara sahiptir:

- 250k PASS
- 500k PASS
- 1M PASS
- 8/8 scenario PASS
- hard blockers zero
- long-run gate PASS
- equal-budget residual-watch gate PASS
- `OPS_BUNDLE_CLEAN`

Bu, proje içindeki simülasyon ve governance contract için güçlü bir sonuçtur.

## Action 24 ve Action 32 Neden Önemli?

Action 24 ve action 32 bazı stres senaryolarında sık seçilen iki action'dır:

- Action 24: dispatch + shortest route + secondary fleet + no reorder
- Action 32: dispatch + low-congestion route + secondary fleet + no reorder

Bunlar şu anda hata olarak sınıflandırılmadı; çünkü gate'lerden geçtiler ve hard blocker üretmediler. Ama gerçek şirket verisi olmadan bu kararların ekonomik olarak her zaman doğru olduğu iddia edilmiyor. Bu yüzden monitoring watch olarak izleniyorlar.

## Gerçek Dünyaya Geçmek İçin Ne Eksik?

Gerçek dünya kullanımı için şirket verisi, yani **company data**, gerekir:

- gerçek sipariş akışı,
- dispatch denemeleri,
- teslimat sonuçları,
- araç ve carrier availability,
- rota başarısızlık sebepleri,
- stok ve reorder olayları,
- maliyet bilgileri,
- TMS/WMS/ERP entegrasyon kuralları.

Bu veriler gelmeden gerçek dünya otonomluğu iddia edilmez.

## Tez Danışmanına En Güvenli Açıklama

Bu proje, 5PL lojistik için simülasyon tabanlı otonom karar verme sistemi geliştirir. Model, simülasyon içinde sipariş, stok, rota, filo ve reorder kararlarını birlikte verir. Mevcut hierarchical v1 1M modeli proje sözleşmesi içinde testlerden geçmiştir ve production-like süreçle yönetilmektedir. Ancak sistem gerçek şirket verisine ve canlı operasyon sistemlerine bağlanmadığı için henüz gerçek dünyada full autonomous değildir. Projenin akademik değeri, digital twin, reinforcement learning, hierarchical action design ve MLOps governance bileşenlerini tek bir izlenebilir prototipte birleştirmesidir.
