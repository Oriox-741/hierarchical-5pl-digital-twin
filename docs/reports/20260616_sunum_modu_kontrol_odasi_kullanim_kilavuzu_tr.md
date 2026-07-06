# Sunum Modu Kontrol Odası Kullanım Kılavuzu

## Amaç

Bu kılavuz, hiyerarşik v1 1M modelini Türkçe, deterministik ve sunuma uygun bir dijital ikiz kontrol odası olarak göstermek içindir.

Varsayılan mod `SUNUM_MODU`dur. Bu mod doğrulanmış simülasyon izlerini oynatır; canlı şirket verisi, canlı TMS/WMS/ERP entegrasyonu veya canlı üretim operasyonu iddiası taşımaz.

## Önerilen Başlatma

Önerilen başlatıcı `.bat` dosyasıdır; PowerShell yürütme politikası `.ps1` dosyasını engelleyebilir.

Önerilen:

```text
start_control_room.bat
```

Elle başlatmak için:

```powershell
python scripts\control_room_server.py
```

PowerShell ile başlatmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_control_room.ps1
```

Statik paket:

```text
reports/demo_control_room_v5/index.html
```

Statik paket görsel inceleme içindir. Etkileşimli yerel sunum için önerilen yol `start_control_room.bat` dosyasıdır.

## Sunum Akışı

1. Kontrol odasını açın.
2. Varsayılan modun `Sunum Modu` olduğunu doğrulayın.
3. `Başlat` düğmesi ile deterministik iz oynatımını başlatın.
4. Harita, KPI, PPO kontrol çubukları ve DQN aksiyon kartını gösterin.
5. `Tek Adım` ile karar akışını kontrollü şekilde ilerletin.
6. `48 Aksiyon Sözlüğü` panelinde dış aksiyon kontratını gösterin.
7. `Kamu Replay` panelinde kamu verisinin yalnızca betimleyici/proxy kanıt olduğunu belirtin.
8. `Kanıtlar ve Sınırlar` paneliyle kapsam dışı iddiaları netleştirin.

## Modlar

- `Sunum Modu`: önerilen varsayılan mod. Deterministik, tekrarlanabilir ve canlı şirket verisinden bağımsızdır.
- `Canlı Simülasyon`: yalnızca teknik gösterim için seçilmelidir; mevcut `env_5pl` bağlantısı varsa kullanılabilir.
- `Güvenli Görsel Demo`: deterministik görsel akış için ayrı güvenli demo modu.
- `Kamu Replay`: kamu veri dağılımlarını betimleyici olarak gösterir; nedensel OPE veya şirket verisi doğrulaması değildir.

## Ekran Alanları

- Üst bar: senaryo, mod, durum, simülasyon zamanı, adım, model ve bağlantı durumu.
- Sol panel: senaryo, mod, hız ve başlat/duraklat/sıfırla/tek adım kontrolleri.
- Orta alan: sentetik koordinatlı dijital ikiz haritası.
- Sağ panel: güncel PPO + DQN kararı, DQN aksiyon yorumu, PPO kontrol çubukları ve senaryo stres göstergeleri.
- Alt alan: KPI kartları, olay akışı ve aksiyon geçmişi.
- İkincil paneller: 48 aksiyon sözlüğü, kamu replay özeti, kanıtlar ve sınırlar.

## Deterministik İzler

V5 sunum izleri:

- `reports/demo_control_room_v5/golden_traces/baseline_normal_trace.json`
- `reports/demo_control_room_v5/golden_traces/mixed_stress_trace.json`
- `reports/demo_control_room_v5/golden_traces/route_disruption_congestion_trace.json`

Her iz 120 adımdır ve araçlar, siparişler, rotalar, aktif rota, PPO kontrolleri, DQN aksiyon kimliği, çözümlenmiş aksiyon, KPI'lar ve olay satırı içerir.

## Exe Notu

`.exe` paketleme isteğe bağlıdır. `.exe` hatalı simülasyon mantığını düzeltmez; yalnızca dağıtım biçimidir. V5'in güvenli ve doğrulanmış yolu batch/PowerShell başlatıcılarıdır.

İsteğe bağlı paketleme komutu:

```powershell
python scripts\package_control_room_exe.py --dry-run
```

PyInstaller kurulu değilse bu yardımcı betik korumalı ürün artefaktlarını değiştirmeden açıklayıcı mesajla çıkar.

## İddia Edilmemesi Gerekenler

Bu sunumda aşağıdaki iddialar yapılmamalıdır:

- Canlı şirket TMS/WMS/ERP entegrasyonu.
- Gerçek şirket lokasyonu, siparişi veya müşteri verisi kullanımı.
- Şirket verisiyle doğrulanmış ekonomik üstünlük.
- Kamu replay verisinden nedensel OPE sonucu.
- Global SOTA veya canlı dağıtım üstünlüğü.
- İnsan müdahalesini ortadan kaldıran self-healing operasyon.

## Dosyalar

- Sunucu: `scripts/control_room_server.py`
- Exe yardımcı betiği: `scripts/package_control_room_exe.py`
- Başlatıcılar: `start_control_room.bat`, `start_control_room.ps1`
- Statik paket: `reports/demo_control_room_v5/index.html`
- JavaScript: `reports/demo_control_room_v5/app.js`
- Stil: `reports/demo_control_room_v5/styles.css`
- Golden traces: `reports/demo_control_room_v5/golden_traces/`
- Ekran görüntüleri: `reports/demo_control_room_v5/screenshots/`

## Kapanış Kontrolü

Sunumdan önce kontrol edin:

- Üst barda `Sunum Modu` görünüyor.
- Bağlantı durumu `Hazır`.
- 48 aksiyon sözlüğü açılıyor.
- Kamu replay paneli açılıyor.
- Görünür ham JSON/debug çıktısı yok.
- Tek sekme kullanılıyor.
