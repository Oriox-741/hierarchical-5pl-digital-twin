# Sunum Icin SOTA ve Benchmark Anlatimi

Tarih: 2026-06-13

## Tek Slaytlik Ana Mesaj

Bu proje global SOTA iddiası yapmadan, 5PL karar kontrolü için production-promoted bir hierarchical model ve onu çevreleyen güvenli benchmark kanıt paketi sunar.

## Neyi Gösteriyoruz?

1. Model production'a copy-only terfi etti.
2. Active registry hierarchical v1 1M modeline işaret ediyor.
3. 250k, 500k ve 1M gate'leri geçti.
4. Equal-budget residual watch gate PASS.
5. Continuous-aware rule baseline benchmark tamam.
6. Public route, inventory ve fleet-dispatch analogları çalıştı.
7. Blocked benchmarklar açıkça işaretlendi.

## Benchmarklar Nasıl Anlatılmalı?

### Internal 5PL

Asıl ürün benchmarkı budur. 73 observation, 5 continuous action, 48 discrete action sözleşmesiyle dijital ikizde çalışır.

### Rule-Based

Danışmanın kolay anlayacağı deterministik politika aileleriyle kıyas zemini sağlar. Bu sprintte tam continuous-aware koşu tamamlandı.

### Route References

PyVRP ve daha önceki OR-Tools/Amazon route çalışmaları route tarafı için public referans sağlar. Bu, full 5PL kanıtı değildir.

### Inventory References

`gym-invmgmt` ve MABIM reorder kararları için public analog sağlar. Şirket maliyetleri olmadan kesin ekonomik kalibrasyon yapılmaz.

### Fleet Dispatch

FleetPy küçük public dispatch senaryosu çalıştırıldı. Bu, fleet/dispatch tarafının public analogudur, inventory veya 5PL birleşimini kanıtlamaz.

## Danışmana Söylenecek Denge

"Sistemin tamamı için en güçlü kanıt dijital ikiz gate'leridir. Public benchmarklar ise modelin alt karar ailelerini dış referanslarla açıklamak için kullanılıyor. Şirket verisi geldiğinde bu kanıtlar kalibrasyona dönüşebilir."

## Blocked Kısımlar

- SVRPBench: paket/source erişimi bulunamadı.
- CityFlow: PyPI install yok, source-build işi deferred.
- PyVRP VRPTW: format uyumsuzluğu.

## Sonuç Cümlesi

"SOTA pathway hazır ama kısmi: güçlü ürün ve benchmark kanıtı var; global SOTA iddiası için stochastic routing, congestion analog ve şirket verisi eksikleri ayrı çalışılmalı."

## Classification

`SOTA_PRESENTATION_NARRATIVE_READY_WITH_CAVEATS`
