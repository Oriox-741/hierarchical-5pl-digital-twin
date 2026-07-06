# Canlı Türkçe Dijital İkiz Dashboard V3 Raporu

## Kısa Teşhis

V2 dashboard ürün demosu beklentisini karşılamıyordu. Streamlit tarafı etkileşimli görünse de ana akış güvenli deterministik sentetik demo durumundan ilerliyordu; gerçek `env_5pl` reset/step döngüsü varsayılan backend değildi. Üretim policy de varsayılan olarak yüklenmiyor ve araç/sipariş/KPI değerleri gerçek env snapshot'ından değil sentetik görsel trajektoriden geliyordu.

V3 değişikliği bu boşluğu kapatır: varsayılan mod `CANLI_SIMULASYON_MODU` olarak gerçek `configs/eval_scenarios/*.json` senaryosunu `real_world_scenario_arena` üzerinden `FivePLDigitalTwinEnv` içine bağlar, `reset/step` çağırır, 73 boyutlu gözlemden gruplanmış karar izini üretir ve KPI kartlarını env snapshot/reward bileşenlerinden günceller. Harita hâlâ gerçek şirket koordinatı kullanmaz; canlı env durumunu soyut `sentetik dijital ikiz haritası` üzerinde gösterir.

## Modlar

- `CANLI_SIMULASYON_MODU`: `env_5pl + real_world_scenario_arena` reset/step döngüsü. Varsayılan policy kapalıyken bounded deterministik demo policy ile env adımı atar.
- `GÜVENLİ_SENTETİK_MOD`: canlı env bağlanamazsa veya kullanıcı seçerse görünür fallback. Üretim policy çıktısı gibi sunulmaz.
- `PUBLIC_REPLAY_MODU`: kamu replay dağılımı ve 48 aksiyon sözlüğü için betimleyici proxy görünümü.

## Üretim Policy Toggle

`Production policy read-only kullan` açıkken dashboard `PolicyService.predict_joint(..., fallback_to_heuristic=False)` yoluyla aktif hiyerarşik Torch joint policy'yi CPU üzerinde read-only yükler. Yerel smoke testte `algorithm=torch_joint`, continuous uzunluk `5`, DQN aksiyon `0..47` döndü. Hata olursa UI hatayı gösterir ve deterministik demo policy fallback etiketi kullanır.

## Ürün Demosu Değişiklikleri

- Streamlit sol panel: `Başlat`, `Durdur`, `Sıfırla`, `Tek Adım`, hız `0.25x/0.5x/1x/2x/5x`, senaryo, mod ve production-policy toggle.
- `streamlit-autorefresh` ile `Başlat` aktifken otomatik ilerleme.
- Sekmeler Türkçe V3 adlarına çevrildi: `Kontrol Odası`, `Canlı Senaryo`, `Dijital İkiz Haritası`, `PPO + DQN Karar Akışı`, `48 Aksiyon Sözlüğü`, `Kamu Replay Analizi`, `KPI ve Zaman Çizgisi`, `Kanıtlar ve Sınırlar`.
- 48 aksiyon, aksiyon 24/32 residual-watch etiketi ve all-action public replay toplamı `227,891` korundu.
- Statik fallback `reports/demo_dashboard_v3/index.html` açıkça "Bu statik önizlemedir. Canlı ürün demosu için Streamlit çalıştırın." der.

## Sınırlar

- Canlı şirket TMS/WMS/ERP entegrasyonu yoktur.
- Şirket verisi doğrulaması yoktur.
- Kamu replay nedensel OPE veya şirket verisi doğrulaması değildir.
- Bu çalışma global SOTA veya canlı deployment iddiası içermez.
- Dashboard bir karar-destek sistemi olarak adlandırılmamıştır.

## Doğrulama Durumu

- `python -m py_compile scripts\visual_digital_twin_dashboard.py tests\benchmarks\test_visual_digital_twin_dashboard.py`: PASS.
- `python -m unittest tests.benchmarks.test_visual_digital_twin_dashboard -v`: PASS, 9 test.
- `python scripts\visual_digital_twin_dashboard.py --export-html reports\demo_dashboard_v3\index.html`: PASS.
- Üretim policy read-only smoke: PASS, `torch_joint`, continuous len `5`, legal DQN action.
- Streamlit smoke: HTTP `200` on `localhost:8516`; browser DOM confirmed Turkish tabs, `Başlat`/`Durdur`/`Tek Adım`, map SVG/canvas, KPI step values, action-tab filters, action 24/32 watch text, public replay proxy warning, and `227,891` total.
- Auto-run browser smoke: `Başlat` advanced `Adım` from `0` to `3`.
- Registry hash proof: `active_models.json` and `models.jsonl` hashes unchanged after V3 implementation/export.
- Process proof after smoke: `NO_MATCHING_PROCESSES` for train/eval/gate/AWS/registry/Streamlit dashboard processes.

## Sonuç

Dashboard artık static/report ağırlıklı V2 yerine canlı env bağlı Türkçe Streamlit kontrol odası olarak çalışacak şekilde yeniden kuruldu. Statik HTML yalnızca güvenli önizleme artefaktıdır.

Final sınıflandırma: `LIVE_TURKISH_DIGITAL_TWIN_DASHBOARD_READY`
