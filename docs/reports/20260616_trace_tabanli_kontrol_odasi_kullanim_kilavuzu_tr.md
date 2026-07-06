# Trace Tabanlı Kontrol Odası V6 Kullanım Kılavuzu

## Amaç

Bu paket, hiyerarşik v1 1M modelinin karar akışını canlı şirket verisi kullanmadan, deterministik operasyon kayıtları üzerinden gösterir. Varsayılan ekran `Sunum Modu / Operasyon Kaydı Oynatıcı` olarak açılır.

## Başlatma

Önerilen yol:

```bat
start_control_room.bat
```

Elle başlatma:

```bat
python scripts\control_room_server.py
```

PowerShell alternatif yol:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_control_room.ps1
```

PowerShell execution policy engeli yaşanabileceği için Windows üzerinde önerilen başlatıcı `.bat` dosyasıdır.

## Ekran Bileşenleri

- `Senaryo`: oynatılacak operasyon senaryosunu seçer.
- `Operasyon Kaydı`: deterministik trace dosyasını seçer.
- `Zaman çizelgesi`: trace içinde istenen adıma atlar.
- `Oynat`: kaydı ilerletmeye başlar.
- `Duraklat`: otomatik ilerlemeyi durdurur.
- `Baştan al`: kaydı 0. adıma döndürür.
- `Geri sar`: bir adım geri gider.
- `İleri sar`: bir adım ileri gider.
- `PPO kaynağı`: PPO değerlerinin trace içindeki deterministik demo policy çıktısı olduğunu açıklar.
- `PPO geçmişi`: her sürekli kontrolün trace boyunca min, max ve güncel değerini gösterir.
- `DQN karar kartı`: adımda seçilen 0-47 arası ayrık aksiyonu ve Türkçe çözümlemesini gösterir.

## Kısıtlar

- Canlı şirket verisi kullanılmaz.
- Şirket verisi doğrulaması iddia edilmez.
- Kamu replay nedensel OPE değildir.
- Kontrol odası bir sunum ve güvenli gözlem aracıdır; registry, production, baseline, DB veya checkpoint yazmaz.

## Statik Paket

Statik önizleme:

```text
reports/demo_control_room_v6/index.html
```

Trace dosyaları:

```text
reports/demo_control_room_v6/traces/
```

Ekran görüntüleri:

```text
reports/demo_control_room_v6/screenshots/
```

## Doğrulama Özeti

- V6 trace API GET çağrıları salt-okunurdur.
- Zaman sadece açık POST oynatma komutlarıyla ilerler.
- Trace verileri deterministiktir.
- Araç hareketleri ardışık adımlarda süreklidir.
- Tüm 48 aksiyon sözlüğü ve kamu replay paneli korunmuştur.
