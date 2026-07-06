# CODEX-5PL Benchmark Suite Spec

Tarih: 2026-06-13

## Amaç

Bu doküman, CODEX PROJE için kendi benchmark standardını tanımlar. Problem, tekil route veya inventory problemi değildir; 5PL dijital ikizinde dispatch, route tercihi, fleet modu ve reorder kararlarının birlikte kontrol edilmesidir.

Makine-okunur spec:

`reports/benchmarks/sota_pathway_20260613/codex_5pl_benchmark_suite/codex_5pl_benchmark_suite_spec.json`

## Formal Problem

Görev: `physical_reality_v5_route_candidate_visibility` sözleşmesi altında, 73 boyutlu gözlemden 5 boyutlu continuous PPO kontrolü ve 48 aksiyonlu DQN taktik kararı üretmek.

Bu karar şu alt bileşenleri kapsar:

- dispatch veya hold
- route tercihi
- primary/secondary fleet modu
- reorder yok/konservatif/agresif/acil

## State

Observation dim: `73`

Ana feature grupları:

- order ve SLA baskısı
- araç/fleet uygunluğu
- route candidate görünürlüğü
- inventory ve reorder baskısı
- scenario stress bilgisi
- action feasibility bağlamı

## Action

Continuous:

- PPO output dim: `5`
- Normalized zero, fiziksel no-op değildir; midpoint fiziksel kontrole projekte olur.

Discrete:

- DQN external action count: `48`
- İzlenen kritik aksiyonlar:
  - action 24: dispatch + shortest + secondary_fleet + none
  - action 32: dispatch + low_congestion + secondary_fleet + none

## Scenario Set

Mevcut 8 stres senaryosu:

- baseline
- premium pressure
- route disruption
- vehicle scarcity
- high holding
- late time
- mixed stress
- low congestion

## Metrics

Zorunlu metrikler:

- service level
- lateness
- dispatch rate
- dispatch success
- no-work
- no-current
- no-unassigned
- failed-noop
- route failure
- no_vehicle
- already_assigned
- action concentration
- action 24/32 rate
- fleet mix
- reorder none rate

## Baselines

Benchmark suite içinde tutulması gereken aileler:

- mevcut hierarchical v1 production
- eski flat production comparator
- historical flat / teacher-retention adayları
- rule neutral continuous
- rule heuristic continuous
- PPO-assisted tactical rule
- OR-Tools ve PyVRP route-only references
- gym-invmgmt ve MABIM inventory references
- FleetPy public fleet/dispatch reference

## Statistical Protocol

- Varsayılan episode sayısı: 20
- Varsayılan seed: 42
- Equal-budget karşılaştırma zorunludur.
- Bootstrap CI kullanılabilir.
- Paired veri varsa permutation testi kullanılabilir.
- Senaryo seviyesinde sign test kullanılabilir.
- Hard blocker varsa aggregate skor tek başına yeterli değildir.

## Public Reproducibility

Public taraf:

- Amazon Last Mile route samples
- CVRPLIB/Solomon-compatible route data
- MABIM/ReplenishmentEnv
- FleetPy example study
- gym-invmgmt

Protected taraf:

- registry, production, baselines, DB, checkpoints ve eski eval outputları değiştirilemez.

## SOTA Claim Rules

Bu proje için güvenli iddia:

- "5PL dijital ikizinde gated hierarchical joint-control candidate üretildi."

Güvensiz iddia:

- "Global SOTA" veya "gerçek dünyada optimal" demek.

SOTA iddiası için ayrı ayrı gerekir:

- route-only benchmarklarda solver reference kanıtı
- inventory/fleet domain benchmarklarında public veya company-data kanıtı
- full 5PL için equal-budget scenario kanıtı ve company telemetry doğrulaması

## Classification

`CODEX_5PL_BENCHMARK_SUITE_SPEC_READY`
