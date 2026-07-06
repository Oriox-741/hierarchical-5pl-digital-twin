# İnteraktif Türkçe Streamlit Dijital İkiz Dashboard V2 Raporu

Tarih: 2026-06-16

## Amaç

Mevcut görsel dijital ikiz dashboard'u V2 seviyesine yükseltildi: Türkçe Streamlit arayüzü, sentetik ve deterministik simülasyon döngüsü, hareketli dijital ikiz haritası, PPO+DQN karar izi, 48 aksiyon çözücü, kamu replay dağılımı ve iyileştirilmiş statik HTML yedeği.

## Kapsam Çerçevesi

- Dijital ikiz, 5PL operasyonlarının simülasyon ortamıdır.
- PPO+DQN, simülasyon ortamı içinde çalışan kontrol politikasıdır.
- Dashboard, simülasyon ortamını ve politika kararlarını görselleştiren Türkçe arayüzdür.
- Varsayılan mod sentetik/statik güvenlidir; üretim checkpoint'i yüklemez.
- Canlı şirket deployment'ı, şirket verisi doğrulaması, nedensel kamu replay/OPE veya global üstünlük iddiası yapılmaz.

## Bağımlılık Durumu

| Paket | Durum |
| --- | --- |
| Streamlit | Kurulu: `1.57.0` |
| pandas | Kurulu: `3.0.2` |
| Plotly | Kuruldu: `6.8.0` |
| streamlit-autorefresh | Opsiyonel, kurulu değil |

Minimal dosya: `requirements-dashboard.txt`.

## Oluşturulan/Güncellenen Artifacts

| Artifact | Yol |
| --- | --- |
| Streamlit V2 dashboard | `scripts/visual_digital_twin_dashboard.py` |
| Dashboard testleri | `tests/benchmarks/test_visual_digital_twin_dashboard.py` |
| Statik V2 HTML yedeği | `reports/demo_dashboard_v2/index.html` |
| Türkçe kullanım kılavuzu | `docs/reports/20260615_interaktif_dashboard_kullanim_kilavuzu_tr.md` |
| V2 raporu | `docs/runs/20260615_visual_digital_twin_dashboard_v2_report.md` |
| Minimal dashboard requirements | `requirements-dashboard.txt` |

## V2 Özellikleri

| Alan | Durum |
| --- | --- |
| Türkçe UI | Sayfa başlıkları, kontroller, açıklamalar, KPI kartları ve sınır notları Türkçeleştirildi. |
| Sekmeler | Genel Bakış, Senaryo Çalıştırıcı, Dijital İkiz Haritası, PPO + DQN Karar İzi, 48 Aksiyon Çözücü, Kamu Replay Dağılımı, KPI Zaman Çizgisi, Kanıtlar ve Sınırlar. |
| Simülasyon döngüsü | Streamlit `session_state` ile Başlat, Durdur, Sıfırla, Tek Adım İlerle, Hız seçimi ve Senaryo seçimi eklendi. |
| Hareketli harita | Soyut 0-100 ızgarada hub, araç, sipariş, rota adayları, sıkışıklık/kesinti ve seçili aksiyon rota etkisi gösteriliyor. |
| PPO+DQN iz | 73-D gözlem özet grupları, beş PPO sürekli kontrol, DQN aksiyon id ve Türkçe çözümleme gösteriliyor. |
| 48 aksiyon çözücü | Tüm 0..47 aksiyonlar, dispatch/route/fleet/reorder faktörleri, arama/filtreler ve kamu replay sayı/oranı ile görünür. |
| Aksiyon 24/32 | İzlenen aksiyon olarak işaretlenir; model davranışının tamamı olarak sunulmaz. |
| Kamu replay | LaDe `31,415`, NYC HVFHS `100,000`, Olist `96,476`, toplam `227,891`; top 10, sıfır sayımlı aksiyonlar ve aile dağılımları gösterilir. |
| KPI zaman çizgisi | Offline eval çalıştırmadan deterministik `synthetic visual demo trajectory` üretir. |
| KPI aksiyon dağılımı | Sentetik zaman çizgisindeki aksiyon dağılımı, aksiyon 24/32 izleme işaretleriyle özetlenir. |
| Statik HTML yedeği | Türkçe, CSS'li, harita snapshot'lı ve açıkça etkileşimsiz fallback olarak etiketli. |

## Doğrulama

| Komut / Kontrol | Sonuç |
| --- | --- |
| RED test | V2 testleri ilk çalıştırmada beklenen şekilde başarısız oldu: Türkçe sekmeler, simülasyon API'si ve V2 HTML metni eksikti. |
| Focused unittest | `python -m unittest tests.benchmarks.test_visual_digital_twin_dashboard -v` geçti: 11 test. |
| py_compile | `python -m py_compile scripts\visual_digital_twin_dashboard.py tests\benchmarks\test_visual_digital_twin_dashboard.py` geçti. |
| HTML export | `python scripts\visual_digital_twin_dashboard.py --export-html reports\demo_dashboard_v2\index.html` geçti. |
| Streamlit localhost smoke | Geçti: geçici `localhost:8504` HTTP 200 döndü ve süreç kapatıldı. |
| Browser interaction smoke | Geçti: in-app browser üzerinde sekmeleri, Türkçe kontrolleri, Plotly haritasını, `Tek Adım İlerle` ile adım 1 ilerlemesini, görünür lejantı ve Streamlit chrome temizliğini doğruladı. |
| Forbidden visible-output scan | Dashboard HTML, V2 raporu ve Türkçe kılavuzda yasak overclaim ifadeleri temizlendi. |
| Reviewer | İlk read-only reviewer verdict `INTERACTIVE_TURKISH_STREAMLIT_DASHBOARD_NEEDS_FIXES`; ardından KPI trace, harita lejantı, dataset selector, KPI aksiyon dağılımı ve Streamlit chrome temizliği sıkılaştırıldı. Re-review çalıştırılacak. |

## Protected Path Durumu

Bu çalışma eğitim, offline eval, long-run gate, registry güncellemesi, production/baseline/DB/checkpoint değişikliği veya mevcut eval çıktısı üzerine yazma yapmaz. İzinli yazma kapsamı dashboard script/test/docs, `requirements-dashboard.txt` ve `reports/demo_dashboard_v2/**` ile sınırlıdır.

## Beklenen Sınıflandırma

Final doğrulama ve reviewer onayından sonra beklenen sınıflandırma:

`INTERACTIVE_TURKISH_STREAMLIT_DASHBOARD_READY`
