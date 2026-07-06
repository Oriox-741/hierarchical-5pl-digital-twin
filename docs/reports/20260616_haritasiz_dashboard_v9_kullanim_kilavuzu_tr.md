# Haritasız Dashboard V9 Kullanım Kılavuzu

## Amaç

Bu dashboard, hiyerarşik v1 1M modelinin simüle 5PL dijital ikiz ortamındaki karar orkestrasyonunu gösterir. Ana ekran gerçek araç takip haritası değildir; PPO+DQN kararları, KPI eğilimleri, 48 aksiyon dağılımı, public replay analizi, benchmark kanıtları ve şirket verisi gereksinimlerini tek ürünsü panelde toplar.

## Harita Neden Kaldırıldı?

Sentetik harita gerçek GPS, gerçek rota polylinesi veya canlı operasyon lokasyonu varmış gibi yanlış beklenti oluşturabiliyordu. Projenin güçlü tarafı coğrafi görselleştirme değil; 73 boyutlu gözlemden PPO sürekli kontrolleri ve DQN taktik aksiyonları üreten karar döngüsüdür. V9 bu yüzden haritasız, karar ve kanıt odaklı bir kontrol paneli sunar.

## Önerilen Mod: Demo Akışı

`Demo Akışı`, danışman gösterimi için önerilen varsayılan moddur. İçeride deterministik iz/playback kullanır; kullanıcıya `Sunum Modu` veya `Operasyon Kaydı Oynatıcı` gibi teknik adlarla görünmez.

`Teknik Simülatör` ileri teknik inceleme içindir. `Kamu Replay` public proxy dağılımlarını açıklamak içindir.

## PPO Nasıl Açıklanır?

PPO çevrimiçi öğrenmez; karar anında mevcut gözleme göre beş sürekli kontrol çıktısı üretir.

- İkmal oranı
- Dispatch yoğunluğu
- Hız çarpanı
- Emniyet stoku
- Kapasite tamponu

## DQN Nasıl Açıklanır?

DQN, 48 taktik aksiyon içinden dispatch/rota/filo/ikmal bileşimini seçer. Aksiyon 24 ve aksiyon 32 izlenen aksiyonlardır; public replay ve residual-watch yorumları nedensel üstünlük iddiası değildir.

## Skor Kartı Ne Anlama Gelir?

`Benchmark / Kanıt Skor Kartı` performans yüzdesi değildir. Değerler `score / 5` kanıt olgunluğu olarak okunmalıdır. Limit satırları şirket telemetrisi, canlı TMS/WMS/ERP entegrasyonu ve nedensel OPE gibi henüz yapılmamış alanları açıkça sınırlar.

## Public Replay Ne Anlama Gelir?

Public replay, LaDe, NYC HVFHS ve Olist gibi public proxy kaynaklarından türetilen betimleyici aksiyon dağılımıdır. Şirket verisi doğrulaması değildir, tam trajectory/reward/propensity içermez ve nedensel public replay üstünlüğü kanıtlamaz.

## Ne İddia Edilmemeli?

- Canlı TMS/WMS/ERP devreye alma yapıldığı.
- Şirket verisi doğrulamasının tamamlandığı.
- Public replay analizinin nedensel üstünlük kanıtladığı.
- Global SOTA veya gerçek dünya optimumu.
- Dashboard’un gerçek GPS, gerçek rota çizgisi veya gerçek şirket lokasyonu gösterdiği.

## Çalıştırma

Önerilen Windows başlatıcı:

```bat
start_control_room.bat
```

Elle başlatma:

```bat
python scripts\control_room_server.py
```

PowerShell alternatif:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_control_room.ps1
```

## Statik Paket

```text
reports/demo_control_room_v9/index.html
reports/demo_control_room_v9/app.js
reports/demo_control_room_v9/styles.css
reports/demo_control_room_v9/traces/
reports/demo_control_room_v9/screenshots/
```

Statik önizleme canlı API çalıştırmaz. Canlı dashboard için `start_control_room.bat` veya `python scripts\control_room_server.py` kullanılmalıdır.
