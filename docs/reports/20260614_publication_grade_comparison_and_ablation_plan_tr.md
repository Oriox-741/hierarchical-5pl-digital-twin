# Publication-Grade Comparison And Ablation Plan - 2026-06-14

## Classification

`HISTORICAL_ABLATION_TABLE_READY`

## Evidence

Machine-readable ablation package:

- `reports/benchmarks/public_data_expansion_20260614/publication_ablation_package/historical_ablation_table.json`

## Recommended Comparison Stack

1. Old flat production parent as the operational baseline.
2. Failed reward/config branches V2/V2.1/V2.2/V2.3/V2.4 as negative ablations.
3. Exact-resume V2.5 as a continuation-mechanics ablation.
4. Teacher-retention ladder as a behavior-anchoring negative ablation.
5. Hierarchical v1 with flat-teacher distillation as the winning architecture.
6. Rule-based deterministic baselines as advisor-readable non-neural references.
7. Public datasets as external validity probes, not direct proof of company optimality.

## Public Evidence Additions

- LaDe-D: last-mile delivery timing/courier workload probe.
- Olist: ecommerce lateness and freight proxy.
- Tesco: public area-level grocery demand proxy.
- NYC TLC FHV: public dispatch-pressure proxy.
- SVRPBench: VRP geometry and dynamic-appearance reference.

## Claims Allowed

- The promoted hierarchical v1 model passed the project gate suite.
- The benchmark environment now includes public-data probes across route/logistics, ecommerce lateness, demand, dispatch pressure, and VRP geometry.
- Negative architecture branches are documented and can be used as ablation evidence.

## Claims Not Allowed

- Global SOTA.
- Real-world operational optimality.
- Company-data validation.
- Secondary-fleet or no-reorder economic optimality from public route-only data.

