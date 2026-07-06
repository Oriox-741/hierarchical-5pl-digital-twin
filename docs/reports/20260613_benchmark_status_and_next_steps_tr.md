# Benchmark Durumu ve Sonraki Adimlar

## Durum

Benchmark hattinin ilk guvenli surumu hazir:

- Rule-based baseline harness: calisiyor.
- Runtime latency benchmark: calisiyor.
- Amazon bounded route-proxy report: calisiyor.
- OR-Tools VRPTW smoke benchmark: calisiyor.

## Eksikler

### Rule-Based Full Budget

Tam 20 episode x 8 scenario x 8 baseline kosusu 30 dakikalik calisma limitini asti. Bounded 3-episode sonuc danisman icin yeterli ilk kanittir, ama nihai skor tablosu icin daha uzun bir islem penceresi gerekir.

Guvenli sonraki komut:

```powershell
python -m src.eval.rule_based_baseline_arena --scenario-dir configs/eval_scenarios --output-dir reports/benchmarks/rule_based_baseline_20ep_YYYYMMDD --episodes 20 --seed 42
```

Not: Bu komut `reports/benchmarks` altina fresh output yazmali; `models/eval` kullanilmamali.

### Amazon Full Dataset

Full Amazon listing:

- 42 object.
- 3.1 GiB.
- En buyuk training travel-times: 1.7 GiB.

Mevcut analyzer JSON'u eager load eder. Full dataset icin once streaming parser gerekir.

Guvenli sonraki is:

- `scripts/real_world_calibration/amazon_route_streaming_sampler.py`
- `tests/real_world_calibration/test_amazon_route_streaming_sampler.py`
- Fresh `reports/benchmarks/amazon_route_proxy_streamed_YYYYMMDD/`

### OR-Tools Kapsami

Mevcut smoke benchmark tiny VRPTW instance cozer. Daha guclu karsilastirma icin:

- Solomon VRPTW instance parser.
- CVRPLIB parser.
- Identik route-only objective.
- Dijital ikizle karistirmadan route-only benchmark.

## Model Arastirmasi Tetikleri

Yeni training su an onerilmez. Sadece su durumlarda yeni gated research acilmali:

- Production monitoring service veya lateness regression gosterirse.
- Action 24/32 no-current, no-unassigned veya failed-noop uretirse.
- Company data secondary_fleet veya reorder_none ekonomisiyle celisirse.
- Equal-budget benchmark yeni bir operasyonel risk gosterirse.

## Onerilen Siralama

1. Monitoring ops bundle'i periyodik calistir.
2. Company data gelirse synthetic validator yerine gercek header/schema preflight yap.
3. Public dataset genisletilecekse once streaming parser.
4. Rule-based full-budget benchmark icin uzun calisma penceresi ayir.
5. Model extension gerekiyorsa 1.5M/2M gated planla basla.
