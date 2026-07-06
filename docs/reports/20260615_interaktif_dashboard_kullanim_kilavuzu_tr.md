# İnteraktif Türkçe Dijital İkiz Dashboard V2 Kullanım Kılavuzu

## Amaç

Bu dashboard, 5PL operasyon simülasyonu içindeki PPO+DQN kontrol politikasını Türkçe ve görsel bir demo arayüzüyle anlatmak için hazırlanmıştır.

Temel çerçeve:

- Dijital ikiz, 5PL operasyonlarının simülasyon ortamıdır.
- PPO+DQN, bu ortamın içinde çalışan kontrol politikasıdır.
- Dashboard, simülasyon ortamını, sentetik operasyon döngüsünü ve model karar izini görselleştirir.
- Sistem simülasyon içinde otonomdur; canlı TMS/WMS/ERP entegrasyonu iddia edilmez.
- Kamu replay betimleyici proxy analizidir; nedensel OPE veya şirket verisi doğrulaması değildir.

## Kurulum

Gerekli hafif paketler:

```powershell
python -m pip install -r requirements-dashboard.txt
```

Doğrudan kurulum:

```powershell
python -m pip install streamlit plotly pandas
```

`streamlit-autorefresh` opsiyoneldir. Kurulu değilse dashboard yine çalışır; simülasyon ilerletmek için `Tek Adım İlerle` düğmesi kullanılır.

## Streamlit Modu

Önerilen çalışma modu:

```powershell
streamlit run scripts/visual_digital_twin_dashboard.py
```

Arayüzdeki ana sekmeler:

1. Genel Bakış
2. Senaryo Çalıştırıcı
3. Dijital İkiz Haritası
4. PPO + DQN Karar İzi
5. 48 Aksiyon Çözücü
6. Kamu Replay Dağılımı
7. KPI Zaman Çizgisi
8. Kanıtlar ve Sınırlar

## Simülasyon Kontrolleri

Sol menüdeki kontroller:

- `Senaryo seçimi`: `configs/eval_scenarios/*.json` altındaki çalıştırılabilir senaryolar ve dokümante alias'lar.
- `Başlat`: otomatik yenileme eklentisi varsa akışı canlı ilerletir.
- `Durdur`: ilerlemeyi durdurur.
- `Sıfırla`: seçili senaryonun sentetik başlangıç durumuna döner.
- `Tek Adım İlerle`: araç pozisyonu, sipariş durumu, PPO kontrolleri, DQN aksiyonu ve KPI değerlerini deterministik bir adım ilerletir.
- `Hız seçimi`: otomatik yenileme varsa yenileme aralığını değiştirir.

## Statik HTML Yedeği

Streamlit yerine temiz bir statik HTML yedeği üretmek için:

```powershell
python scripts/visual_digital_twin_dashboard.py --export-html reports/demo_dashboard_v2/index.html
```

Statik çıktı:

```text
reports/demo_dashboard_v2/index.html
```

Not: Statik HTML etkileşimli değildir; danışman demosu için önerilen mod Streamlit arayüzüdür.

## Danışmana Anlatım Akışı

1. `Genel Bakış`: 73-D gözlem, beş PPO sürekli kontrol, 48 DQN ayrık aksiyon ve `hierarchical_v1` mimarisini göster.
2. `Senaryo Çalıştırıcı`: sekiz çalıştırılabilir senaryonun dinamik keşfedildiğini, `low_congestion` gibi alias'ların ayrıca işaretlendiğini göster.
3. `Dijital İkiz Haritası`: gerçek şirket koordinatı kullanmadan sentetik hub, araç, sipariş, rota adayı ve sıkışıklık/kesinti bayraklarını göster.
4. `PPO + DQN Karar İzi`: PPO'nun sürekli kontrollerini ve DQN'nin 0..47 arası ayrık aksiyonunu birlikte açıkla.
5. `48 Aksiyon Çözücü`: tüm aksiyon yüzeyini göster; aksiyon 24 ve 32'nin yalnızca izlenen aksiyonlar olduğunu belirt.
6. `Kamu Replay Dağılımı`: LaDe, NYC HVFHS, Olist ve toplam 227.891 satırlık all-action dağılımını göster.
7. `KPI Zaman Çizgisi`: sentetik görsel demo trajektorisinin offline eval olmadığını açıkça söyle.
8. `Kanıtlar ve Sınırlar`: şirket verisi gereksinimini, canlı deployment yapılmadığını ve global üstünlük iddiası olmadığını kapat.

## Ne İddia Edilmemeli?

- Canlı şirket deployment'ı yapıldığı.
- Şirket verisi doğrulamasının tamamlandığı.
- Kamu replay'in nedensel OPE olduğu.
- Kamu replay'in politika üstünlüğünü kanıtladığı.
- Küresel en-iyi-sistem üstünlüğü.
- Dashboard'un canlı operasyon yönetim sistemi olduğu.

## Sorun Giderme

`streamlit` tanınmıyor:

```powershell
python -m streamlit run scripts/visual_digital_twin_dashboard.py
```

Port doluysa:

```powershell
python -m streamlit run scripts/visual_digital_twin_dashboard.py --server.port 8502
```

Tarayıcı açılmıyorsa terminalde görünen `http://localhost:...` adresini elle aç.

Dosya kilidi varsa açık Streamlit/HTML görüntüleyici pencerelerini kapat ve komutu tekrar çalıştır.

Plotly eksikse:

```powershell
python -m pip install plotly
```
