# Tez Danismani Icin Sade Operasyon Walkthrough

Tarih: 2026-06-12

Bu dokuman, detayli operasyon anlatiminin daha sade versiyonudur. Ornekler birebir step trace degil, kodla uyumlu temsili operasyon ornegidir.

## 1. Ana Benzetme

Projeyi bir **lojistik kontrol kulesi** gibi dusunebiliriz.

- `env_5pl`: Simule lojistik dunyasi. Siparisler, stok, araclar, rotalar, kapasite ve teslimatlar burada yasar.
- Observation: Kontrol kulesinin ekrani. Sistem bu ekranda 73 sayilik operasyon fotografi gorur.
- PPO: Ince ayar kollaridir. Stok, dispatch yogunlugu, hiz, safety stock ve kapasite buffer gibi surekli ayarlari yapar.
- DQN: Ana operasyon karari verir. "Dispatch mi hold mu?", "hangi rota?", "hangi filo?", "reorder var mi?" sorularini cevaplar.
- Action mapper: DQN'in sectigi tek sayiyi insanin anlayacagi lojistik komuta cevirir.
- Safety projector: Karar tehlikeliyse sinirlar veya bloke eder.
- Monitoring: Karar sonrasi denetimdir. Sistem "bu karar gercekten ise yaradi mi?" diye bakar.

## 2. Tek Adimda Ne Oluyor?

Bir karar adimi soyle akar:

1. Simulasyon mevcut operasyon durumunu hazirlar.
2. Bu durum 73 boyutlu observation olur.
3. PPO 5 tane surekli ayar uretir.
4. DQN 0 ile 47 arasinda bir action secer.
5. Action mapper bu sayiyi lojistik karara acar.
6. Safety katmani karari kontrol eder.
7. Simulasyon dispatch, rota, filo ve stok etkilerini uygular.
8. Sonuc service, lateness, dispatch success, route failure ve no-work metrikleriyle denetlenir.

## 3. Action 24 Ne Demek?

Action 24 su anlama gelir:

`dispatch + shortest route + secondary_fleet + no reorder`

Yani sade dille:

> "Bir siparisi simdi yola cikar, en kisa rota ailesini kullan, secondary fleet kullan, bu adimda ekstra reorder yapma."

Bu karar mixed stress gibi zor bir ortamda mantikli olabilir. Cunku:

- talep artmistir;
- teslimat baskisi vardir;
- primary fleet sinirli olabilir;
- secondary fleet ek kapasite saglayabilir;
- shortest route gecikmeyi azaltmak icin cekici olabilir;
- no reorder, her dispatch adiminda gereksiz stok hamlesi yapmamayi temsil edebilir.

Ama bu "her zaman dogru" demek degildir. Bu yuzden action 24 monitoring altindadir.

Equal-budget eval'de action 24 icin onemli nokta:

- mixed stress scenario PASS;
- hard blocker zero;
- action 24 no-current-work: 0;
- action 24 no-unassigned: 0;
- action 24 failed-noop: 0;
- action 24 dispatch success ratio: 1.0;
- action 24 concentration yuksek oldugu icin izlenmeye devam ediyor.

## 4. Action 32 Ne Demek?

Action 32 su anlama gelir:

`dispatch + low_congestion route + secondary_fleet + no reorder`

Yani sade dille:

> "Bir siparisi simdi yola cikar, trafik/aksama daha az olan rota ailesini kullan, secondary fleet kullan, bu adimda ekstra reorder yapma."

Bu karar route disruption/congestion ortaminda mantikli olabilir. Cunku:

- en kisa rota trafik veya disruption yuzunden iyi rota olmayabilir;
- low-congestion route daha guvenilir olabilir;
- secondary fleet ek kapasite saglayabilir;
- reorder karari rota probleminden bagimsiz tutulabilir.

Equal-budget eval'de action 32 icin onemli nokta:

- route disruption scenario PASS;
- hard blocker zero;
- action 32 no-current-work: 0;
- action 32 no-unassigned: 0;
- action 32 failed-noop: 0;
- action 32 no_vehicle: 0;
- action 32 dispatch success ratio: 1.0;
- action 32 concentration yuksek oldugu icin izlenmeye devam ediyor.

## 5. Neden Bu Bir "Akilli Karar" Sayiliyor?

Model sadece "rota sec" yapmiyor. Ayni anda sunlari dusunuyor:

- siparis bekliyor mu?
- arac var mi?
- rota feasible mi?
- gecikme riski var mi?
- stok ve kapasite durumu nasil?
- primary/secondary fleet karari mantikli mi?
- karar fake dispatch veya no-work yaratir mi?

Bu nedenle proje, tek bir optimizasyon script'i degil, digital twin icinde calisan kapali dongu karar sistemidir.

## 6. Nerede Dikkatli Konusmaliyiz?

Danismana soylemek guvenli:

> Model simule 5PL ortaminda otonom karar verir, 8 senaryoda PASS almistir, hard blocker uretmemistir ve action 24/32 davranislari monitoring altinda kabul edilmistir.

Danismana soylemek henuz guvenli degil:

> Bu model gercek sirket operasyonunda action 24/32'nin ekonomik olarak optimal oldugunu kanitlamistir.

Bunu soylemek icin gercek order, dispatch, route, fleet, inventory, cost ve failure verisi gerekir.

