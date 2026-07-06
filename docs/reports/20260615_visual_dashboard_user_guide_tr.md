# Görsel Dijital İkiz Dashboard Kullanım Kılavuzu

Bu kılavuz, yerel görsel dashboard'u tez ve danışman anlatımı için kullanmayı açıklar.

## Temel Konumlandırma

Dashboard'un ana mesajı şudur:

> The digital twin is the simulated 5PL operational environment; PPO+DQN is the controller inside it.

Bu nedenle:

- Dijital ikiz, 5PL operasyonlarının simülasyon ortamıdır.
- PPO+DQN, bu simülasyon ortamı içinde aksiyon üreten kontrol politikasıdır.
- Dashboard, bu ortamı ve politikanın karar izini görselleştiren arayüzdür.
- Sistem simülatör içinde otonomdur; canlı TMS/WMS/ERP devreye alma iddiası yoktur.
- Kamu replay çıktıları betimleyici proxy kanıttır; şirket verisi doğrulaması veya nedensel üstünlük sonucu değildir.

## Çalıştırma

Streamlit arayüzü:

```powershell
streamlit run scripts/visual_digital_twin_dashboard.py
```

Statik HTML üretimi:

```powershell
python scripts/visual_digital_twin_dashboard.py --export-html reports/demo_dashboard/index.html
```

Statik dosya:

```text
reports/demo_dashboard/index.html
```

## Sayfalar

| Sayfa | Ne gösterir? |
| --- | --- |
| Overview | Model kimliği, sözleşme, 73 boyutlu gözlem, 5 PPO kontrolü, 48 DQN aksiyonu ve `hierarchical_v1` mimarisi. |
| Scenario Runner | `configs/eval_scenarios` altındaki tüm çalıştırılabilir senaryoları dinamik bulur; dokümante edilip çalışma konfigürasyonu olmayan senaryoları ayrıca işaretler. |
| Digital Twin Map / Zone View | Şirket lokasyonu kullanmadan sentetik bölge haritası gösterir: hub, sipariş, araç, rota adayı, sıkışıklık/kesinti bayrağı ve teslimat bölgeleri. |
| PPO + DQN Decision Trace | 73-D gözlemin özetini, beş PPO sürekli çıktısını, DQN aksiyon kimliğini ve aksiyonun dispatch/rota/filo/ikmal faktörlerine ayrılmasını gösterir. |
| All-48 Action Decoder | 0..47 tüm aksiyonları görünür yapar. Aksiyon 24 ve 32 yalnızca izlenen aksiyonlardır; modelin tamamını temsil etmez. |
| Public Replay Action Distribution | LaDe `31.415`, NYC HVFHS `100.000`, Olist `96.476`, toplam `227.891` satır; top 10 aksiyon, sıfır sayımlı aksiyonlar ve aile dağılımları. |
| KPI Timeline | Eş bütçeli güvence çalışmasından statik servis, gecikme, dispatch, başarı ve hata göstergeleri. |
| Evidence and Limitations | Simülatör kapıları, eş bütçeli karşılaştırma, kamu proxy kanıtı, şirket verisi ihtiyacı ve kapsam sınırları. |

## Güvenli Demo Modu

Varsayılan mod statik/sentetik demodur:

- Eğitim çalıştırmaz.
- Offline eval çalıştırmaz.
- Long-run gate çalıştırmaz.
- Registry, production, baseline, DB, checkpoint veya eski eval çıktısını değiştirmez.
- Şirket konumu veya şirket siparişi uydurmaz.

## Danışman Anlatımı İçin Kısa Akış

1. Overview sayfasında mimariyi anlat: 73-D gözlem, 5 PPO kontrolü, 48 DQN aksiyonu.
2. Scenario Runner ile sekiz çalışma senaryosunun dinamik bulunduğunu göster.
3. Zone View ile dijital ikizin simülasyon ortamı olduğunu vurgula.
4. Decision Trace ile PPO'nun sürekli kontrolleri, DQN'nin taktik aksiyonu ürettiğini göster.
5. All-48 Action Decoder ile modelin yalnızca aksiyon 24/32'den ibaret olmadığını göster.
6. Public Replay sayfasında kamu verisinin açıklayıcı proxy niteliğini ve sınırını açıkça söyle.
7. Evidence and Limitations ile şirket verisi ve canlı operasyon sınırını kapat.

## Sınırlar

Bu dashboard şirket verisi doğrulaması yapmaz. Canlı operasyon yönetimi aracı değildir. Kamu replay sonuçları karşı-olgusal politika başarısı kanıtı olarak okunmamalıdır. Şirket özelinde maliyet, filo, stok, rota ve dispatch başarısının doğrulanması için ayrı onaylı veri çalışması gerekir.
