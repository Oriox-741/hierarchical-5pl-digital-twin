# CODEX PROJE Project Development Master Plan

Date: 2026-06-13

Scope: comprehensive future development plan for CODEX PROJE. Non-project personal planning topics named by the user are out of scope. The focus is project development, thesis explanation, benchmarks, real-world data, company data request readiness, academic/technical strengthening, productization, monitoring, governance, and safe future roadmap.

This plan is documentation only. It does not authorize training, offline eval, long-run gate, dataset download, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output edits, training config creation, private company-data ingestion, or 1.5M/2M/3M/5M/10M/100M runs.

## Executive Summary

CODEX PROJE is currently a 5PL digital-twin autonomous decision-control prototype. It is stronger than a simple route optimizer because it coordinates dispatch, route choice, fleet mode, reorder behavior, inventory pressure, service/lateness, and operational governance inside one traceable simulator-production workflow.

Current production truth:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- runtime: `torch_joint`
- architecture: `hierarchical_v1`
- init: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: `73`
- PPO continuous action dimension: `5`
- DQN external discrete action count: `48`
- production status: active registry plus copy-only production promotion complete
- ladder: 250k PASS, 500k PASS, 1M PASS
- equal-budget residual-watch gate: PASS
- ops bundle: `OPS_BUNDLE_CLEAN`
- accepted watches: action 24 and action 32 concentration
- main remaining evidence gap: company data for secondary_fleet and reorder economics

Best next direction:

1. Protect and explain the current production model.
2. Turn the thesis walkthrough into advisor/demo material.
3. Prepare company-data request and benchmark packages.
4. Add external baselines and public-data benchmark plans in a staged, approval-gated way.
5. Productize only around existing runtime and monitoring boundaries.
6. Do not start blind longer training.

Recommended first next safe executable task:

`Create a firm-facing company-data request email/template plus field dictionary handoff package.`

Why this first: it directly targets the biggest blocker between simulator readiness and real-world validation, requires no protected mutation, and builds on the existing company-data request package and intake validator.

## A. Current Project Level

The project is best classified as:

> A 5PL digital-twin autonomous decision-control prototype with production-like lifecycle governance.

Academic/product level, stated conservatively:

- Above a typical undergraduate software project: the repo contains a trained simulator policy, runtime serving path, registry lifecycle, artifact health, monitoring, ops bundle, and release handoff.
- Suitable as a master's/applied-research core: it combines a domain simulator, hierarchical RL policy, evaluation gates, monitoring, and reproducible governance.
- Not yet sufficient for doctoral-strength scientific claims by itself: that would require external benchmarks, ablations, statistical robustness, and company-data validation.
- Suitable as a technical MVP core for a startup-style product direction: it has the model/runtime/monitoring foundation, but still needs customer data, integration adapters, UX, security, operational support, and human override workflows.

Current capability level:

| Dimension | Current level | Evidence | Boundary |
| --- | --- | --- | --- |
| Simulator autonomy | Strong | `env_5pl`, `real_world_scenario_arena`, obs 73/action 48, 8-scenario gates | Simulator only |
| Model maturity | Strong under project contract | hierarchical v1 1M passed 250k/500k/1M, 8/8 PASS, hard blockers zero | No real-world optimum claim |
| Runtime maturity | Medium-high | `PolicyService`, `torch_joint_runtime`, active registry, artifact health smoke | No live TMS/WMS/ERP integration |
| Governance maturity | Strong for local project | registry, active activation, copy-only production, manifest, hashes, ops bundle | Local filesystem governance |
| Real-world validation | Partial | Amazon small sample supports route-side plausibility | Company fleet/reorder/cost data absent |
| Academic maturity | Applied research core | digital twin + hierarchical RL + MLOps governance | Needs benchmarks, ablations, company data for stronger thesis/scientific claims |
| Product maturity | Technical MVP core | runtime service path, monitoring specs, ops bundle | Needs API, dashboard, telemetry, integration adapters |

Advisor-safe statement:

> This is not a live commercial logistics deployment. It is a simulator-production-ready research prototype: autonomous in the 5PL digital twin, governed like a production artifact in the repo, and ready for benchmark/company-data validation before real-world claims.

What the project can do now:

- run autonomous decision loops inside the existing 5PL simulator;
- load and serve the promoted hierarchical v1 1M model through `PolicyService`;
- decode 48 tactical DQN actions into dispatch, route, fleet, and reorder components;
- report artifact health, synthetic monitoring, company-schema readiness, and ops-bundle state;
- explain representative action 24 and action 32 behavior using source code plus aggregate evidence.

What the project cannot do yet:

- operate as a live TMS/WMS/ERP integration;
- prove real-world optimality for route, fleet, or inventory decisions;
- validate secondary_fleet or reorder-none economics without company data;
- dispatch real orders automatically;
- justify longer training without monitoring, benchmark, or company-data triggers.

## B. Existing Autonomous Loop

The autonomous operations loop already exists. A new synthetic sandbox is not the next default move.

Existing components:

- `src/act/env_5pl.py`: owns synthetic 5PL operations physics and action application.
- `src/act/observation_builder.py`: builds the fixed 73-dimensional state representation.
- `src/act/discrete_action_mapper.py`: maps DQN action ids 0..47 into dispatch/route/fleet/reorder decisions.
- `src/eval/real_world_scenario_arena.py`: runs closed-loop simulator rollouts using PPO+DQN until episode end.
- `src/orchestration/torch_joint_runtime.py`: loads the torch joint checkpoint and predicts continuous/discrete actions.
- `src/orchestration/policy_service.py`: resolves active registry and serves `algorithm=torch_joint`.
- `scripts/monitoring_report_generator.py`: summarizes action 24/32 watch metrics from telemetry-like rows.
- `scripts/run_production_ops_bundle.py`: aggregates artifact health, monitoring, and synthetic company schema into one local ops report.

Conclusion:

- A full new synthetic operations sandbox would duplicate `env_5pl`.
- A docs-only demo guide is safe and useful.
- A thin read-only demo runner may be useful later only if a stakeholder needs a runnable presentation surface over existing `env_5pl` and `PolicyService`.

## C. Detailed Development Tracks

### Track 1: Thesis Explanation And Demo Clarity

Purpose:

- Make the project understandable to a thesis advisor without overclaiming.
- Show one operation step by step: observation -> PPO/DQN -> action mapper -> env step -> metrics -> monitoring.

Current assets:

- `docs/reports/20260612_tez_danismani_operasyon_walkthrough_detayli.md`
- `docs/reports/20260612_tez_danismani_operasyon_walkthrough_sade_versiyon.md`
- `docs/reports/20260612_tez_danismani_operasyon_walkthrough_diyagram_notlari.md`
- `docs/runs/20260612_operation_walkthrough_traceability_audit.md`

Future deliverables:

- final polished one-page advisor handout;
- diagram pack in advisor-ready format;
- docs-only demo guide over existing eval/runtime artifacts;
- no-code walkthrough of action 24 and action 32;
- explicit "simulasyonda otonom, gercek dunyada full autonomous degil" section.

Safe-now work:

- write docs and diagrams only.

Requires approval:

- runnable demo runner;
- any fresh eval output;
- any live telemetry ingestion.

### Track 2: Real-World Company-Data Calibration

Purpose:

- Validate route/fleet/reorder/cost assumptions using real company data.
- Close the main evidence gap for action 24 and action 32.

Current assets:

- `docs/runbooks/20260611_hierarchical_v1_company_data_request_package.md`
- `scripts/company_data_intake_validator.py`
- `reports/company_data/20260611_company_data_synthetic_schema_report.json`

Minimum viable first extract:

- orders;
- dispatch attempts;
- deliveries/outcomes;
- routes;
- fleet/carrier availability;
- inventory/reorder;
- costs.

Key validation questions:

- Does action 24's secondary_fleet + no reorder behavior make economic sense under mixed stress?
- Does action 32's low_congestion route preference match actual route reliability under disruption?
- Are no_vehicle, already_assigned, route_failure, no_current, no_unassigned, and failed_noop semantics comparable to simulator counters?
- Does reorder none coincide with stockout/backlog/emergency-reorder risk?

Safe-now work:

- company data request email/template;
- field dictionary handoff;
- privacy/anonymization instructions;
- schema validator usage guide.

Requires approval:

- any real private data ingestion;
- any DB write;
- any calibration run using company data;
- any model training triggered by company mismatch.

### Track 3: Benchmark Testing

Purpose:

- Compare the project against external, rule-based, and internal baselines.
- Strengthen academic claims without jumping directly into training.

Benchmark families:

1. Internal old production vs hierarchical equal-budget comparator.
2. OR-Tools routing baseline.
3. Rule-based dispatch baselines.
4. Amazon Last Mile full route-proxy analysis.
5. Solomon VRPTW.
6. Homberger VRPTW.
7. CVRPLIB/CVRP.
8. Multi-seed robustness benchmark.
9. Runtime latency benchmark.
10. Ablation benchmark.
11. Historical company replay.

Safe-now work:

- benchmark design docs;
- benchmark input/output schema;
- no-run feasibility plan;
- route-proxy method documentation.

Requires approval:

- new dependencies such as OR-Tools;
- public dataset download beyond approved small files;
- any fresh eval/benchmark output;
- any training/eval/long-run gate.

### Track 4: Productization

Purpose:

- Turn the project from a research artifact into an operable technical MVP core.

Potential components:

- API/service layer around `PolicyService`;
- model metadata endpoint;
- health-check endpoint;
- operator dashboard;
- action-watch visualization;
- human override workflow;
- audit log;
- integration adapter plan for TMS/WMS/ERP;
- local demo guide or thin demo runner.

Safe-now work:

- API design doc;
- health-check spec;
- dashboard wireframe/spec;
- human override workflow;
- integration adapter interface plan.

Requires approval:

- running a service;
- writing persistent operational logs;
- connecting to any live company system;
- ingesting private data.

### Track 5: Monitoring And Governance

Purpose:

- Keep the production state auditable and protect against accidental drift.

Existing tools:

- Step 2 artifact-health CLI: `ARTIFACT_HEALTH_CLEAN`
- Step 3 monitoring generator: `MONITORING_REPORT_CLEAN`
- Step 4 company-data intake validator: `COMPANY_DATA_SCHEMA_READY`
- Step 5 ops bundle: `OPS_BUNDLE_CLEAN`
- lifecycle governance runbook

Future work:

- periodic ops bundle execution checklist;
- generated health status dashboard;
- report archive and naming convention;
- incident/rollback playbook;
- protected-hash audit workflow;
- dashboard truth-source maintenance.

Safe-now work:

- documentation and scheduled checklist design.

Requires approval:

- automated recurring execution if it writes reports;
- any registry/production/baseline/DB mutation;
- any rollback.

### Track 6: Future Model Research

Current position:

- No blind 3M/5M/10M/100M.
- No 1.5M/2M unless a concrete trigger appears.

Valid triggers:

- monitoring regression;
- company-data mismatch;
- no-current/no-unassigned/failed-noop reappears on action 24/32;
- route failure/no_vehicle becomes material;
- public benchmark contradicts route assumptions;
- runtime/checkpoint bug found and fixed with tests.

Possible future research:

- bounded 1.5M/2M exact-resume extension;
- hierarchical_v2 only after evidence;
- action concentration regularization;
- uncertainty/fallback policy;
- action attribution and explainability tools.

Required controls:

- exact resume;
- fresh output dirs;
- no failed parents;
- no `--allow-empty-replay-resume`;
- preflight tests and protected hashes;
- offline eval and gate at each rung;
- independent review.

## D. Company Data Strategy

The strongest next validation path is a company data request package, not more training.

Minimum viable fields:

| Field | Why needed | Model question | Validates action 24/32 component | Privacy/join note |
| --- | --- | --- | --- | --- |
| `order_id` | Join orders to dispatch, delivery, route, inventory | What work existed at decision time? | dispatch feasibility | can be hashed; must be stable |
| `created_at` | Demand arrival and queue age | Was dispatch urgent? | dispatch/hold | timezone required |
| `promised_window_start` | SLA lower bound | Is lateness pressure real? | route choice | timestamp semantics required |
| `promised_window_end` | SLA upper bound | On-time service definition | route choice | timestamp semantics required |
| `delivery_time` | Actual outcome | Was service achieved? | route/fleet/reorder impact | no PII needed |
| `delivered_on_time` | KPI target | Did policy improve service? | all action components | derived acceptable if definition documented |
| `dispatch_status` | Attempt outcome | Was dispatch executable? | dispatch | must distinguish accepted/failed/cancelled |
| `dispatch_failure_reason` | Failure taxonomy | Why dispatch failed | no-current/no-unassigned/no_vehicle/route_failure | controlled labels preferred |
| `vehicle_id` | Vehicle availability/capacity | Was a vehicle available? | secondary_fleet | surrogate id ok |
| `carrier_id` | Carrier performance/cost | Which carrier/fleet executed? | secondary_fleet | surrogate id ok |
| `fleet_type` | Primary vs secondary | Is secondary_fleet realistic? | secondary_fleet | controlled values primary/secondary |
| `planned_route_id` | Route plan join | Which route was planned? | shortest/low_congestion | route id or segment id |
| `planned_travel_time` | Expected route time | Was route efficient? | route | numeric seconds/minutes |
| `actual_travel_time` | Real route outcome | Did route perform? | route | actual/planned comparison |
| `stock_on_hand_at_decision` | Inventory state | Was reorder needed? | reorder none | SKU/site join needed |
| `stockout_flag` | Inventory failure | Did no-reorder hurt? | reorder none | boolean or event row |
| `primary_fleet_cost` | Baseline cost | Was primary cheaper? | secondary_fleet | aggregate or banded ok |
| `secondary_fleet_cost` | Overflow cost | Was secondary economical? | secondary_fleet | aggregate or banded ok |

Ideal additions:

- customer/location/zone;
- package volume/weight;
- pickup time;
- dispatch attempt id;
- carrier accepted/cancelled times;
- route failure reason;
- no_vehicle/already_assigned flags;
- route distance;
- congestion/delay flags;
- inventory/reorder events;
- supplier lead time;
- stockout cost;
- holding cost;
- SLA penalty cost;
- emergency reorder cost.

Data governance rules:

- first extract should be read-only;
- hash customer/order identifiers if possible;
- obfuscate precise coordinates if sensitive, but preserve zone/route comparability;
- no production DB writes;
- no private data in repo unless separately approved;
- run schema validator first on synthetic/header-only contract;
- stop if join keys or timestamp semantics are missing.

## E. Benchmark Strategy

### Benchmark 1: Internal Old Production vs Hierarchical Equal-Budget

- Purpose: preserve current fair comparator.
- Proves: hierarchical is not just winning through eval budget mismatch.
- Cannot prove: real-world optimality.
- Data: existing equal-budget summaries.
- Safe now: read-only analysis yes; rerun no.
- Approval needed: any fresh eval/gate output.
- Metrics: service, lateness, dispatch success, route failure, no_vehicle, action 24/32 rates.
- Stop conditions: output dir exists, protected hash drift, no explicit eval approval.
- Priority: P0 evidence baseline.

### Benchmark 2: OR-Tools Routing Baseline

- Purpose: compare route-side performance to a known operations-research solver.
- Proves: how route choices compare to a classical route optimizer on comparable instances.
- Cannot prove: reorder, inventory, secondary-fleet economics unless extended.
- Data: synthetic/order-location-fleet instance or approved public data.
- Required code: OR-Tools adapter, instance exporter, metric comparator.
- Safe now: plan only.
- Approval needed: dependency install, implementation, benchmark run.
- Metrics: route time, distance, service, lateness, vehicle count, infeasible orders.
- Stop conditions: dependency unavailable, incompatible instance mapping, would write protected outputs.
- Priority: P1 after design.

### Benchmark 3: Rule-Based Dispatch Baselines

- Purpose: compare learned policy with transparent logistics heuristics.
- Baselines:
  - FIFO + shortest + primary;
  - earliest due date + shortest;
  - premium-first;
  - low-congestion under disruption;
  - stock-threshold reorder.
- Proves: whether RL beats understandable rules under the simulator contract.
- Cannot prove: real-world optimality.
- Data: existing simulator scenarios.
- Safe now: design only.
- Approval needed: implementation and fresh eval outputs.
- Metrics: service, lateness, dispatch success, route failure, no_vehicle, cost proxies.
- Stop conditions: rule needs new simulator semantics or writes old eval dirs.
- Priority: P1.

### Benchmark 4: Amazon Last Mile Full Route Proxy

- Purpose: validate route-side behavior using public last-mile data.
- Proves: shortest/fastest and low-congestion/reliability proxies are directionally plausible.
- Cannot prove: fleet or reorder economics.
- Data: Amazon Last Mile full/sampled public dataset.
- Safe now: plan only; small sample already done.
- Approval needed: any larger download.
- Metrics: actual vs greedy/shortest ratio, travel-time asymmetry, stress buckets, route concentration.
- Stop conditions: size > approved limit, license concern, no exact file list.
- Priority: P1 for route-side evidence.

### Benchmark 5: Solomon VRPTW

- Purpose: classic time-window route feasibility benchmark.
- Proves: route/time-window reasoning can be compared to known benchmark structures.
- Cannot prove: fleet/reorder/company economics.
- Data: Solomon instances.
- Safe now: plan only.
- Approval needed: dataset download/implementation/run if not already local.
- Metrics: vehicles, distance, time-window violations, service/lateness proxy.
- Stop conditions: mapping requires unsupported simulator changes.
- Priority: P2.

### Benchmark 6: Homberger VRPTW

- Purpose: larger-scale VRPTW stress.
- Proves: route-side scalability under bigger time-window instances.
- Cannot prove: inventory/reorder/fleet economics.
- Data: Homberger instances.
- Safe now: plan only.
- Approval needed: download/implementation/run.
- Metrics: feasibility, distance/time, vehicles, time-window violations.
- Stop conditions: runtime too large, mapping too artificial.
- Priority: P2 after Solomon.

### Benchmark 7: CVRPLIB/CVRP

- Purpose: capacity and vehicle-routing sanity benchmark.
- Proves: capacity-aware routing behavior in synthetic standard instances.
- Cannot prove: time windows, inventory, dispatch failure taxonomy unless extended.
- Data: CVRPLIB public instances.
- Safe now: plan only.
- Approval needed: download/implementation/run.
- Metrics: capacity violations, route cost, vehicle count.
- Stop conditions: no time-window/fleet/reorder mapping possible.
- Priority: P3 supporting benchmark.

### Benchmark 8: Multi-Seed Robustness

- Purpose: test gate sensitivity and variance.
- Proves: whether scenario PASS is stable across seeds.
- Cannot prove: company reality.
- Data: existing simulator scenarios.
- Safe now: design only.
- Approval needed: offline eval/gate and fresh output dirs.
- Metrics: mean/sd/min/max service, lateness, dispatch success, action 24/32 concentration.
- Stop conditions: no eval approval, output dirs not fresh, protected drift.
- Priority: P1 if stronger academic evidence needed.

### Benchmark 9: Runtime Latency

- Purpose: measure whether the production runtime path is fast enough for interactive decision support.
- Proves: CPU load/predict/decode latency for `load_torch_joint_policy` and `PolicyService.predict_joint` under controlled local inputs.
- Cannot prove: live production throughput, network latency, TMS/WMS/ERP latency, or database performance.
- Data: synthetic observations only; no private data required.
- Safe now: design only in this planning task.
- Approval needed: implementation and any persistent report output.
- Metrics: cold-load time, warm prediction p50/p95/p99, action decode time, memory footprint, and failure count.
- Stop conditions: benchmark would start a long-running service, mutate registry/production, read private data, or write outside approved fresh report paths.
- Priority: P1 for productization readiness after documentation and company-data request handoff.

### Benchmark 10: Ablation

- Purpose: prove which design choices mattered.
- Comparisons:
  - flat DQN vs hierarchical_v1;
  - distillation vs no distillation;
  - exact resume vs unsafe resume;
  - teacher retention vs hierarchical;
  - action concentration regularization if later added.
- Proves: causal contribution under project contract.
- Cannot prove: real-world deployment.
- Data: new training/eval branches.
- Safe now: not executable; plan only.
- Approval needed: training configs, training, evals, gates.
- Metrics: scenario PASS, hard blockers, residual watches, action quality, runtime compatibility.
- Stop conditions: no architecture decision, no training approval, unsafe parent.
- Priority: P2/P3 only after company/benchmark triggers.

### Benchmark 11: Historical Company Replay

- Purpose: strongest pre-live real-world test.
- Proves: whether policy choices align with historical outcomes under real orders/fleet/routes/inventory/costs.
- Cannot prove: live causal lift without shadow or A/B testing.
- Data: approved company extracts.
- Safe now: no, requires company data approval.
- Metrics: service, lateness, dispatch success, fleet cost, route time, stockout/backlog, action counter alignment.
- Stop conditions: missing join keys, privacy unresolved, no data owner approval.
- Priority: P0 after company data arrives.

## F. Prioritized Execution Roadmap

### Safe-Now Tasks

1. Company data request email/template and data dictionary handoff.
2. Advisor one-page handout and diagram polish.
3. Benchmark master design doc with no downloads/runs.
4. OR-Tools baseline feasibility plan only.
5. Public benchmark plan only.
6. Runtime latency benchmark design only.
7. Existing artifact/ops bundle periodic checklist.
8. Demo guide using existing artifacts only.

### Requires Approval

1. OR-Tools implementation if dependencies are needed.
2. Full Amazon download or larger bounded sample analysis.
3. Offline eval multi-seed.
4. Any benchmark run that writes fresh outputs.
5. Any model training.
6. Any real company data ingestion.
7. Any service process or dashboard that persists live telemetry.

### Requires Company Data

1. Action 24/32 economics validation.
2. Historical replay.
3. Simulator calibration.
4. Business KPI mapping.
5. Secondary_fleet and reorder-none validation.

### Not Recommended Now

1. Blind 3M/5M/10M/100M.
2. Baseline update.
3. Live TMS/WMS/ERP integration.
4. Production mutation.
5. New architecture training.
6. Failed-checkpoint parent reuse.

## G. Recommended First Next Task

Chosen task:

`Create a company-data request email/template and field dictionary package for firm outreach.`

Why this task:

- It addresses the main remaining uncertainty: real-world secondary-fleet and reorder economics.
- It is useful for a thesis advisor, company discussion, and benchmark planning.
- It is safe-now: documentation only, no private data ingestion, no DB write, no training/eval.
- It builds directly on the Step 1 company-data package and Step 4 validator.
- It turns the abstract data need into an actionable request a firm can answer.

Expected future deliverables:

- firm-facing email template;
- one-page data request summary;
- technical field dictionary;
- privacy/anonymization note;
- schema-validator usage instructions;
- stop conditions for insufficient data.

Future goal file prepared:

`docs/goals/20260612_execute_next_safe_development_task_goal.txt`

## H. What Not To Do

- Do not write non-project personal planning content in this project roadmap.
- Do not train.
- Do not run offline eval.
- Do not run long-run gate.
- Do not download datasets.
- Do not update registry.
- Do not mutate active registry, production, baselines, DB, checkpoints, or existing eval outputs.
- Do not create training configs.
- Do not ingest private company data.
- Do not start 1.5M/2M/3M/5M/10M/100M.
- Do not claim action 24/32 are real-world optimal without company data.
- Do not call public route benchmarks validation of fleet/reorder economics.

## I. Final Classification

`PROJECT_DEVELOPMENT_MASTER_PLAN_READY`
