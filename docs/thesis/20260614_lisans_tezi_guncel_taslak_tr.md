# Lisans Tezi Guncel Taslak - 2026-06-14

## Classification

`FINAL_THESIS_DRAFT_READY_WITH_PLACEHOLDER_CITATIONS`

## Ozet

Bu tez, 5PL operasyonlarında dispatch, rota tercihi, filo modu ve reorder kararlarını aynı karar yüzeyinde ele alan CODEX-5PL dijital ikizini inceler. Sistem, 73 boyutlu gözlemden 5 continuous kontrol ve 48 ayrık taktik aksiyon üretir. Güncel üretim adayı `hierarchical_v1` mimarisi ve `flat_teacher_distillation_v1` başlangıcıyla eğitilmiş 1M modeldir.

## Katki

Tezin katkısı global SOTA iddiası değildir. Katkı; birleşik 5PL karar probleminin tanımı, gated simulator-production model, public proxy benchmark paketi, historical replay v0 adapter ve şirket verisiyle yapılacak replay/OPE için açık veri sözleşmesidir.

## Kanit Katmanlari

1. Simulator evidence: 1M hierarchical production candidate, long-run gates, equal-budget residual-watch PASS.
2. Public proxy evidence: route, inventory, fleet/dispatch, ecommerce, delivery timing and historical replay.
3. Company-data-required evidence: secondary fleet economics, reorder safety, dispatch failure causality, causal OPE.

## Historical Replay v0

Large replay ran on LaDe `31415`, NYC HVFHS `100000`, and Olist `96476` rows. It reports missingness/confidence and model action tendencies. It is not causal OPE.

## Kaynaklar

Akademik kaynaklar eklenecek. Public dataset ve tool references proje raporlarında listelenmiştir; final tezde formal citation formatına dönüştürülecektir.
