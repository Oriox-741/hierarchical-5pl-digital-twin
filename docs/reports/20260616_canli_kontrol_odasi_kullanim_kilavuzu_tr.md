# Canlı Kontrol Odası Kullanım Kılavuzu

## Amaç

Bu arayüz, çok katmanlı dijital ikiz çerçevesini Türkçe bir canlı kontrol odası olarak gösterir. Dijital ikiz, simüle edilmiş 5PL operasyon ortamıdır. PPO+DQN denetleyici bu simülasyon içinde karar üretir.

Bu arayüz canlı TMS/WMS/ERP entegrasyonu, şirket verisi doğrulaması veya gerçek şirket lokasyonu/siparişi iddia etmez.

## Başlatma

En kolay yol:

```powershell
.\start_control_room.ps1
```

veya çift tıklama:

```text
start_control_room.bat
```

Sunucu otomatik olarak uygun bir port bulur ve tarayıcıyı açar.

Elle başlatmak için:

```powershell
python scripts\control_room_server.py
```

Statik önizleme:

```text
reports/demo_control_room_v4/index.html
```

Statik önizleme canlı sunucu değildir. Canlı mod için `start_control_room.bat` veya `start_control_room.ps1` kullanılmalıdır.

## Ekran Alanları

- Üst bar: senaryo, mod, durum, simülasyon zamanı, adım ve model göstergesi.
- Sol panel: senaryo, mod, hız ve başlat/duraklat/sıfırla/tek adım kontrolleri.
- Orta alan: büyük dijital ikiz haritası.
- Sağ panel: güncel PPO + DQN kararı, DQN aksiyon yorumu, PPO kontrol çubukları ve senaryo stresi.
- Alt alan: KPI kartları, olay akışı ve aksiyon geçmişi.
- İkincil paneller: 48 aksiyon sözlüğü, kamu replay özeti, kanıtlar ve sınırlar.

## Modlar

- `Canlı Simülasyon`: uygun olduğunda mevcut `env_5pl` ve senaryo arena bağlantısını kullanır.
- `Güvenli Görsel Demo`: deterministic görsel akış kullanır.
- `Kamu Replay`: kamu verisi dağılımlarını betimleyici olarak gösterir; nedensel OPE veya şirket doğrulaması değildir.

## Dikkat Edilecek Sınırlar

- Harita koordinatları görselleştirme amaçlı sentetik koordinatlardır.
- Aksiyon 24 ve 32 izlenen aksiyonlardır; tek başına risk kanıtı değildir.
- Kamu replay dağılımları yönsel/proxy kanıttır.
- Şirket verisi olmadan ekonomik doğrulama yapılmış sayılmaz.
- Bu demo yeni eğitim, eval veya üretim mutasyonu yapmaz.

## Dosyalar

- Sunucu: `scripts/control_room_server.py`
- Başlatıcılar: `start_control_room.bat`, `start_control_room.ps1`
- Statik önizleme: `reports/demo_control_room_v4/index.html`
- Örnek iz: `reports/demo_control_room_v4/sample_trace.json`
- Ekran görüntüleri: `reports/demo_control_room_v4/screenshots/`

