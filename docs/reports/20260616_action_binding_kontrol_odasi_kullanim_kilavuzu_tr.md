# V7 Aksiyon Bağlı Kontrol Odası Kullanım Kılavuzu

## Amaç

Bu paket, hiyerarşik v1 1M modelinin karar akışını canlı şirket verisi kullanmadan, deterministik operasyon kayıtları üzerinden gösterir. V7 sürümünde her DQN kararı bir karar olayı ve atama kaydına bağlanır; araç aynı rota üzerinde ilerlerken sağ panel rastgele veya ilgisiz yeni karar göstermez.

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

Windows üzerinde PowerShell execution policy engeli yaşanabileceği için önerilen başlatıcı `start_control_room.bat` dosyasıdır.

## V7 Karar Semantiği

- DQN aksiyonu bir karar olayında üretilir.
- Bu aksiyon bir atamaya bağlanır: sipariş, araç, rota ve karar aynı kayıt içinde tutulur.
- Hareket kareleri yeni DQN kararı anlamına gelmez.
- Araç aynı atamada ilerliyorsa olay akışı `yeni karar üretilmedi` açıklamasını gösterir.
- Sağdaki karar paneli `Son karar olayı` veya `Seçili aktif operasyon` bağlamını gösterir.
- Kullanıcı aktif atamayı seçerse panel o atamanın bağlı DQN/PPO bilgisini gösterir.

## PPO Semantiği

PPO çevrimiçi öğrenmez ve operasyon sırasında kendini fine-tune etmez. V7 ekranındaki PPO değerleri karar anındaki inference çıktısıdır. Hareket karelerinde bu değerlerin aynı kalması UI hatası değildir; bağlı atama aynı kararın etkisi altında devam ediyor olabilir.

Ekranda kullanılan kaynak etiketi:

```text
Kaynak: deterministik sunum izi
```

## Harita

- Üçgenler araçları gösterir.
- Mavi sipariş noktaları bekleyen veya aktif siparişleri gösterir.
- Turkuaz rota çizgisi aktif rota/atamayı gösterir.
- Araç yanında görünen `DQN N` rozeti, o aracın bağlı atama kararını gösterir.
- Turuncu yarı saydam alanlar stres bölgesidir:
  - `Sıkışıklık bölgesi`
  - `Kesinti riski`
- `Stres bölgelerini göster` seçeneği bu alanları açar veya kapatır.
- Stres içermeyen normal senaryoda bu alanlar gösterilmez.

## Önerilen Danışman Demo Akışı

1. `Normal Operasyon` senaryosunu açın ve ilk karar olayında DQN/PPO kartını gösterin.
2. `Oynat` ile bir hareket karesine geçin; olay akışında yeni karar üretilmediğini gösterin.
3. `Karma Stres` senaryosunda 18-19. adımları gösterin: karar olayı ve sonraki hareket karesi aynı atama ve DQN aksiyonunu korur.
4. `Rota Kesintisi ve Sıkışıklık` senaryosunda turuncu stres bölgelerini açıp kapatın.
5. `Aksiyon Sözlüğü` ve `Kamu Replay` panellerini açarak 48 aksiyon sözlüğünün ve public replay toplamlarının korunduğunu gösterin.

## Sınırlar

- Canlı şirket verisi kullanılmaz.
- Canlı TMS/WMS/ERP entegrasyonu iddia edilmez.
- Şirket verisi doğrulaması iddia edilmez.
- Kamu replay nedensel OPE değildir.
- Bu paket registry, production, baseline, DB veya checkpoint yazmaz.

## Statik Paket

```text
reports/demo_control_room_v7/index.html
reports/demo_control_room_v7/traces/
reports/demo_control_room_v7/screenshots/
```

## Doğrulama Özeti

- V7 trace şeması karar olaylarını hareket karelerinden ayırır.
- DQN ve PPO kararları atama bazında taşınır.
- GET çağrıları salt-okunurdur.
- `tick` yalnızca açık playback komutuyla bir kare ilerletir.
- Tüm 48 aksiyon ve toplam `227891` public replay kaydı korunur.
