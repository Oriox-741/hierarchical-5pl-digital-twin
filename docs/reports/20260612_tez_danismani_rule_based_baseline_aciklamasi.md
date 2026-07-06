# Tez Danismani Icin Rule-Based Baseline Aciklamasi

Tarih: 2026-06-13

Durum: Danismana anlatim paketi. Bu dokuman yalnizca aciklama ve planlama amaclidir; egitim, offline eval, long-run gate, benchmark, veri indirme, dependency install, registry/production/baseline/DB/checkpoint degisikligi, mevcut eval ciktisi duzenleme, training config olusturma veya sirket verisi talebi/ingest islemi yapmaz.

## Kisa Cevap

Rule-based baseline, yapay zeka modelini basit ve insanlarin anlayabilecegi lojistik kurallarla karsilastirmaktir.

Bu proje icin temel soru sudur:

> Model, sadece "FIFO yap, en kisa rotayi sec, primary fleet kullan" gibi basit kurallardan daha iyi mi?

Bu soru danisman icin onemlidir, cunku modelin iyi oldugunu sadece "gate PASS" diyerek anlatmak yetmez. Karsi tarafin anlayacagi referans noktasi gerekir.

## Neden Gerekli?

Mevcut hierarchical v1 1M model proje simulator sozlesmesi icinde gucludur:

- 250k PASS;
- 500k PASS;
- 1M PASS;
- 8/8 scenario PASS;
- hard blockers zero;
- equal-budget residual-watch gate PASS;
- ops bundle clean.

Fakat akademik anlatimda su soru kalir:

> Bu iyi sonuc, gercekten ogrenilmis karar kalitesinden mi geliyor, yoksa basit bir kural da aynisini yapar miydi?

Rule-based baseline bu soruyu test etmek icin ilk ve en temiz benchmark ailesidir.

## "Yapay Zeka Neye Gore Iyi?" Sorusuna Cevap

Danismana soyle anlatilabilir:

> Modelin basarisini sadece kendi icinde olcmuyoruz. Onu, lojistikte sezgisel olarak mantikli basit kurallarla karsilastiracagiz: once gelen siparisi once gonder, en yakin deadline'i once gonder, premium siparisi oncele, congestion varsa low-congestion rota sec, stok dusukse reorder yap, arac azsa secondary fleet fallback kullan. Eger RL modeli bu kurallari ayni simulator kosullarinda gecerse, "model sadece rastgele iyi sonuc vermedi; basit operator kurallarindan daha iyi karar koordinasyonu yapti" diyebiliriz.

Bu, gercek dunya optimumlugu iddiasi degildir. Ama tez savunmasi icin guclu ve anlasilir bir karsilastirma katmanidir.

## Basit Kurallar Neden Guclu Bir Karsilastirmadir?

Basit kurallar zayif gorunebilir, ama benchmark olarak degerlidir:

- FIFO ve earliest-due lojistikte bilinen dispatch sezgileridir.
- Shortest route bircok operasyonun ilk rota varsayimidir.
- Premium-first SLA baskisi icin dogal bir kuraldir.
- Low-congestion route, route disruption altinda makul bir insan kararidir.
- Stock-threshold reorder, en temel inventory kontrol kuralidir.
- Primary-first/secondary-fallback, filo kisitlari icin anlasilir bir kapasite kuralidir.
- High-holding no-overstock, maliyet baskisi altinda dogal bir ekonomi kuralidir.

Model bu kurallari gecerse, sonuc daha anlamli olur. Gecemezse, bu da degerli bir arastirma sonucudur: modelin hangi kisimda basit sezgiden geri kaldigini gosterir.

## Karsilastirilacak Kurallar

Planlanan rule-based baseline ailesi:

1. FIFO + shortest + primary + no reorder
2. Earliest due date + shortest + primary + no reorder
3. Premium-first + shortest + primary/secondary fallback + no reorder
4. Route disruption varsa low-congestion route
5. Conservative stock-threshold reorder
6. Emergency stockout-prevention reorder
7. Vehicle-scarcity primary-first/secondary-fallback
8. High-holding no-overstock rule

Bu kurallar mevcut 48 action uzayina cevrilecek. Ornek:

- action 24 = `dispatch + shortest + secondary_fleet + none`
- action 28 = `dispatch + shortest + primary_fleet + none`
- action 32 = `dispatch + low_congestion + secondary_fleet + none`
- action 36 = `dispatch + low_congestion + primary_fleet + none`

## RL Modelinin Muhtemel Avantaji

Rule-based baseline tek boyutlu karar verir. Mesela:

- FIFO sadece sira bilir.
- Earliest due sadece deadline bilir.
- Shortest route sadece mesafe varsayimini kullanir.
- Stock-threshold sadece stok seviyesine bakar.

Hierarchical RL modeli ise ayni anda sunlari koordine etmeye calisir:

- dispatch mi hold mu?
- shortest mi low_congestion mi high_resilience mi?
- primary_fleet mi secondary_fleet mi?
- reorder none/conservative/aggressive/emergency mi?
- service, lateness, capacity, route disruption, inventory, vehicle scarcity ve holding cost baskilari.

Bu yuzden modelin asil iddiasi "tek bir kuraldan daha akilli" olmasi degil; birden fazla karar boyutunu ayni operasyon durumunda birlikte koordine edebilmesidir.

## Sonuc Nasil Yorumlanir?

Eger RL model rule baselines'i acik sekilde gecer:

- Modelin simulator icinde basit operator kurallarindan daha iyi karar koordinasyonu yaptigi soylenebilir.
- Action 24/32 concentration daha anlamli yorumlanabilir: modelin belirli streslerde basit fallback kurallarindan farkli bir strateji ogrendigi gosterilebilir.

Eger rule baseline modele yakin cikarsa:

- Modelin avantajinin sinirli oldugu kabul edilir.
- Daha guclu benchmark veya ablation gerekir.
- Bu kotu haber degil; akademik olarak durust bir bulgudur.

Eger rule baseline modeli gecer:

- Modelin mevcut simulator politikasinda zayiflik vardir.
- Hemen training baslatilmaz.
- Once hata mi, benchmark mapping sorunu mu, scenario bias mi, yoksa gercek model eksigi mi oldugu incelenir.

## Overclaim Etmeme Notlari

Soyle denmemeli:

- "Rule baseline'i gecersem model gercek dunyada optimaldir."
- "Action 24/32 sirket operasyonunda kesin dogrudur."
- "Bu benchmark sirket fleet ve reorder ekonomisini kanitlar."
- "Rule baseline sonucundan sonra direkt 3M training gerekir."

Soylenebilecek guvenli cumle:

> Rule-based benchmark, mevcut 5PL simulator sozlesmesi icinde hierarchical RL modelini insanlarin anlayacagi basit lojistik kurallarla karsilastirir. Bu, akademik anlatimi guclendirir; fakat gercek dunya optimumlugu icin sirket verisi, entegrasyon validasyonu ve ek benchmarklar gerekir.

## Danismana Tek Paragraf

Bu asamada modelimiz simulator icinde production-like olarak onaylandi, fakat akademik olarak "neye gore iyi?" sorusuna cevap vermemiz gerekiyor. Bu nedenle ilk benchmark olarak rule-based baselines tasarliyoruz. Modeli FIFO, earliest due date, premium-first, low-congestion route, stock-threshold reorder, secondary-fleet fallback ve high-holding no-overstock gibi basit lojistik kurallarla karsilastiracagiz. Bu benchmark gercek dunya optimumlugunu kanitlamaz; ama modelin ayni digital twin kosullarinda basit ve aciklanabilir operator kurallarindan daha iyi karar koordinasyonu yapip yapmadigini gosterir.

