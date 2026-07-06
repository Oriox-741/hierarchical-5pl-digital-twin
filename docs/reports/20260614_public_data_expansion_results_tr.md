# Public Data Expansion Results - 2026-06-14

## Final Evidence Classification

`PUBLIC_DATA_EXPANSION_PARTIALLY_READY_WITH_BLOCKERS`

## What Is Ready

- Public dataset catalog with explicit ready/blocked status.
- LaDe-D bounded last-mile delivery sample summary.
- Olist ecommerce logistics summary.
- Tesco area-level demand proxy summary.
- NYC TLC FHV Gurobi-free dispatch-pressure proxy.
- SVRPBench base geometry/demand/capacity integration.
- Synthetic company replay validation harness.
- Publication ablation package.

## What Is Blocked

- Mendeley planned-vs-actual route data: API 403.
- OpenMines: Python 3.12 / `numpy==1.25.0` installation blocker.
- Instacart: legacy public direct links 404.
- M5: direct Zenodo file access 403.
- SVRPBench stochastic fields: released parquet lacks persisted congestion/delay/accident/time-window fields.
- LaDe publication reuse: license ambiguity must be resolved.

## Operational Interpretation

The project now has credible bounded public-data probes for route/logistics timing, ecommerce lateness, public demand, dispatch pressure, and VRP geometry. That improves benchmark maturity and thesis defensibility. It does not remove the need for company data to validate secondary-fleet cost, no-reorder stockout risk, inventory economics, and action-level operational causality.

## Protected-State Result

This sprint wrote only under approved documentation, scripts/tests, reports, and `data/public` paths. It did not train, run project offline eval, run gates, update registry, mutate production, mutate baselines, mutate DB, mutate checkpoints, or edit existing eval outputs.

