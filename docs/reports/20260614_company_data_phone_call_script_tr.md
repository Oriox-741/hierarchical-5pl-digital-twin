# Company Data Phone Call Script - 2026-06-14

## Classification

`COMPANY_DATA_PHONE_CALL_SCRIPT_READY`

Merhaba, model egitimi icin degil; mevcut simulator ve public replay bulgularini sirket operasyon verisiyle dogrulamak icin minimum anonim veri alanlarini netlestirmek istiyoruz.

Join keyleri korunmus, anonimleştirilmiş orders, dispatch_attempts, routes, fleet, inventory, costs tablolarına ihtiyacımız var. Isim, telefon, adres gibi PII gerekmez; fakat `order_id`, `dispatch_attempt_id`, `vehicle_id`, `carrier_id`, `route_id`, `sku_id/site_id` gibi join keyler ve timestamp sırası korunmalı.

Bu alanlar action 24/32 icin secondary fleet, route reliability, no-reorder stockout ve dispatch failure nedenlerini test eder. Propensity/action probability varsa OPE mumkun olur; yoksa descriptive replay yapılır.
