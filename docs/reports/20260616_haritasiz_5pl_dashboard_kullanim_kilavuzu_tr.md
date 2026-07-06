# Haritasız 5PL Dashboard Kullanım Kılavuzu

## Amaç

Bu dashboard, hiyerarşik v1 1M modelinin simülatör içi karar orkestrasyonunu gösterir. Ana ekran bir araç takip ekranı değildir; PPO+DQN kararları, KPI eğilimleri, 48 aksiyon sözlüğü, public replay analizi, benchmark kanıtları ve sınır beyanlarını tek ürünsü panelde toplar.

## Neden Harita Kaldırıldı?

Sentetik harita, gerçek GPS veya gerçek şirket rota verisi varmış gibi bir beklenti oluşturabiliyordu. Bu proje gerçek konum gösteriminden çok karar orkestrasyonu kanıtı üretir. V8 bu nedenle haritayı ana UI’dan kaldırır ve karar, KPI, aksiyon, replay ve kanıt panellerini öne çıkarır.

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

Windows’ta PowerShell execution policy engeli olabildiği için önerilen başlatıcı `start_control_room.bat` dosyasıdır.

## Danışmana Ne Gösterilmeli?

1. `Operasyon Özeti` panelinde model, kontrat, 73 boyutlu gözlem ve 48 aksiyon sözleşmesini gösterin.
2. `PPO Sürekli Kontroller` panelinde beş sürekli kontrolün karar anında üretildiğini açıklayın.
3. `DQN Karar Kartı` panelinde mevcut aksiyon id, dispatch/hold, rota ailesi, filo modu ve ikmal modunu gösterin.
4. `48 Aksiyon Dağılımı` panelinde tüm aksiyonların görünür ve filtrelenebilir olduğunu gösterin.
5. `Public Replay Analizi` panelinde 227.891 public proxy satırını ve sınır beyanını gösterin.
6. `Benchmark / Kanıt Skor Kartı` panelinde skorların kanıt olgunluğu olduğunu vurgulayın.
7. `Şirket Verisi Gereksinimleri` panelinde gelecek doğrulama için gerekli veri ailelerini gösterin.

## PPO Nasıl Açıklanır?

PPO çevrimiçi öğrenmez. Operasyon anında mevcut gözleme göre beş sürekli kontrol çıktısı üretir:

- `reorder_fraction`
- `dispatch_intensity`
- `speed_multiplier`
- `safety_stock_multiplier`
- `capacity_buffer_fraction`

Bazı kontrollerde düşük varyans görülürse bu UI hatası değildir; aynı karar/atama bağlamı devam ediyor olabilir.

## DQN Nasıl Açıklanır?

DQN dışarıya 0..47 arasında bir ayrık aksiyon üretir. Bu aksiyon dört bileşene ayrılır:

- dispatch veya hold
- rota ailesi
- filo modu
- ikmal modu

Aksiyon 24 ve 32 izlenen aksiyonlardır. Dashboard bu iki aksiyonda watch badge gösterir.

## Public Replay Nasıl Açıklanır?

Public replay paneli, LaDe, NYC HVFHS ve Olist proxy kaynaklarından türetilen betimleyici aksiyon dağılımını gösterir. Bu analiz şirket verisi doğrulaması değildir ve nedensel OPE kanıtı değildir.

## Ne İddia Edilmemeli?

- Canlı TMS/WMS/ERP deployment yapıldığı.
- Şirket verisiyle doğrulamanın tamamlandığı.
- Public replay’in nedensel üstünlük kanıtladığı.
- Global üstünlük veya gerçek dünya optimumu.
- Dashboard’un gerçek şirket lokasyonu, gerçek araç GPS’i veya gerçek rota polylinesi gösterdiği.

## Statik Paket

```text
reports/demo_control_room_v8/index.html
reports/demo_control_room_v8/app.js
reports/demo_control_room_v8/styles.css
reports/demo_control_room_v8/traces/
```

## Teknik Güvenlik Sınırı

Dashboard yerel Python stdlib server ile çalışır. Bu paket eğitim, offline eval, gate, registry update, production promotion, baseline update, DB yazımı, checkpoint yazımı veya özel şirket verisi ingestion işlemi yapmaz.

