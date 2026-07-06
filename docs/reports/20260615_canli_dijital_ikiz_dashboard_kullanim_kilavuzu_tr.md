# Canlı Dijital İkiz Dashboard V3 Kullanım Kılavuzu

## Amaç

Bu dashboard, hiyerarşik PPO+DQN üretim modelini 5PL dijital ikiz simülasyonu içinde Türkçe ve canlı çalışan bir kontrol odası olarak göstermeyi amaçlar. Streamlit arayüzü ürün demosu modudur; statik HTML yalnızca önizlemedir.

## Kurulum

Streamlit Cloud hesabı gerekmez. Yerel kurulum yeterlidir:

```powershell
python -m pip install -r requirements-dashboard.txt
```

Doğrudan kurulum gerekirse:

```powershell
python -m pip install streamlit plotly pandas streamlit-autorefresh
```

## Canlı Çalıştırma

Önerilen komut:

```powershell
streamlit run scripts/visual_digital_twin_dashboard.py
```

`streamlit` komutu tanınmıyorsa:

```powershell
python -m streamlit run scripts/visual_digital_twin_dashboard.py
```

Port doluysa:

```powershell
python -m streamlit run scripts/visual_digital_twin_dashboard.py --server.port 8502
```

## Kontroller

- `Başlat`: seçili senaryoyu otomatik ilerletir.
- `Durdur`: otomatik ilerlemeyi durdurur.
- `Sıfırla`: seçili senaryoda reset yapar.
- `Tek Adım`: bir env/demo adımı ilerletir.
- `Hız`: `0.25x`, `0.5x`, `1x`, `2x`, `5x`.
- `Senaryo seç`: `configs/eval_scenarios/*.json` içindeki senaryolar ve dokümante alias'lar.
- `Mod seç`: canlı simülasyon, güvenli sentetik mod veya kamu replay modu.
- `Production policy read-only kullan`: aktif üretim Torch joint policy'yi CPU üzerinde read-only yüklemeyi dener.

## Modlar

- `CANLI_SIMULASYON_MODU`: gerçek `env_5pl` ve `real_world_scenario_arena` reset/step döngüsü. Varsayılan moddur.
- `GÜVENLİ_SENTETİK_MOD`: canlı env bağlanamazsa veya kullanıcı seçerse görünür fallback.
- `PUBLIC_REPLAY_MODU`: all-action kamu replay dağılımı ve 48 aksiyon sözlüğü için betimleyici mod.

## Statik Önizleme

Statik HTML üretmek için:

```powershell
python scripts/visual_digital_twin_dashboard.py --export-html reports/demo_dashboard_v3/index.html
```

Statik çıktı:

```text
reports/demo_dashboard_v3/index.html
```

Bu dosya etkileşimli ürün demosu değildir. Dosya içinde de "Bu statik önizlemedir. Canlı ürün demosu için Streamlit çalıştırın." uyarısı görünür.

## Danışmana Anlatım

1. `Kontrol Odası`: model kimliği, canlı backend, policy kaynağı ve KPI kartlarını göster.
2. `Canlı Senaryo`: seçili senaryonun gerçekten `configs/eval_scenarios` altından geldiğini anlat.
3. `Dijital İkiz Haritası`: gerçek şirket koordinatı kullanılmadığını, soyut simülasyon haritası olduğunu açıkla.
4. `PPO + DQN Karar Akışı`: 73-D gözlemin operasyonel gruplara çevrildiğini, PPO'nun 5 continuous kontrol ve DQN'nin 0..47 aksiyon ürettiğini göster.
5. `48 Aksiyon Sözlüğü`: tüm aksiyon yüzeyini aç; aksiyon 24/32'nin yalnızca izlenen residual aksiyonlar olduğunu belirt.
6. `Kamu Replay Analizi`: 227,891 kamu replay satırının betimleyici proxy olduğunu vurgula.
7. `Kanıtlar ve Sınırlar`: şirket verisi, canlı deployment ve nedensel OPE sınırlarını kapat.

## Ne İddia Edilmemeli?

- Canlı TMS/WMS/ERP deployment yapıldığı.
- Şirket verisi doğrulamasının tamamlandığı.
- Kamu replay'in nedensel OPE olduğu.
- Kamu replay'in politika üstünlüğünü kanıtladığı.
- Global SOTA iddiası.
- Dashboard'un bir decision-support system olduğu.

## Sorun Giderme

- Streamlit açılmazsa terminaldeki `http://localhost:...` adresini tarayıcıda elle aç.
- Port doluysa `--server.port 8502` gibi farklı port kullan.
- Production policy read-only yükleme hatası görünürse demo otomatik deterministik policy fallback etiketiyle devam eder.
- Plotly eksikse `python -m pip install plotly` çalıştır.
- Otomatik yenileme çalışmazsa `python -m pip install streamlit-autorefresh` çalıştır.
