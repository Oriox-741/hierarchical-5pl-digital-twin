# Danisman Icin Full Benchmark Sonuclari

Siniflandirma: `FULL_BENCHMARK_ADVISOR_REPORT_READY`

Bu rapor, `hierarchical_v1` 1M modelinin yeniden egitim yapilmadan tamamlanan benchmark sprintini ozetler. Sonuc guclu ama sinirlidir: model, mevcut 5PL dijital ikiz contracti icin iyi kanitlanmistir; sirket verisi olmadan gercek operasyonel ekonomi ve entegrasyon iddialari yapilmaz.

## Kanit Tablosu

| Kanit | Sonuc | Tezde nasil soylenmeli |
| --- | --- | --- |
| Simulator ladder | 250k/500k/1M PASS | Model, tanimli digital-twin gate suite icinde basarili. |
| Multi-seed robustness | 5 seed, 800 episode row, hard blocker 0 | Sonuc tek seed'e bagli gorunmuyor. |
| Rule-based baseline | 1280 row, hard blocker 0 | Deterministik taktik baseline karsilastirmasi tamamlandi; neutral continuous caveat var. |
| Amazon full route proxy | 6112 route, 1.46M package | Route-side shortest/low-congestion tercihler yon olarak makul. |
| OR-Tools route suite | Solomon/Homberger/CVRPLIB | Klasik route optimizasyon hatti calisiyor; 5PL dispatch/fleet/reorder degil. |
| Runtime latency | p99 mean 1.5655 ms | CPU inference pratik hizda. |
| Historical ablation | flat/teacher/exact/hierarchical karsilastirmasi | En guclu neden-sonuc: hierarchical factorization + exact resume. |

## Savunulabilir Ana Cumle

Bu calisma, 5PL dijital ikizinde PPO+DQN tabanli hiyerarsik aksiyon mimarisinin, gate'li egitim ladderi ve ek benchmarklarla simulator seviyesinde guvenilir hale getirilebilecegini gostermektedir. Calisma gercek dunya optimalitesi iddia etmez; sirket verisiyle fleet/reorder/cost validasyonu sonraki zorunlu adimdir.

## Risk ve Sinirlar

- Action 24/32 concentration monitoring watch olarak kalir.
- Public data route-side proxy'dir, tam operasyonel gerceklik degildir.
- Sirket verisi olmadan secondary_fleet/reorder_none ekonomisi kanitlanamaz.
- Blind 3M/5M/10M egitim onerilmez; gerekirse once 1.5M/2M gated plan.
