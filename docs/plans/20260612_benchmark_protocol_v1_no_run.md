# Benchmark Protocol V1 No Run

Date: 2026-06-13

Scope: benchmark protocol only. No benchmark is run by this document. No training, offline eval, long-run gate, dataset download, dependency install, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output edit, training-config creation, private company-data ingest, or firm-facing data request is authorized.

## Executive Summary

The current hierarchical v1 1M model has strong simulator/gate evidence, but academic strengthening requires benchmark protocols that separate:

- what can be designed safely now;
- what needs dependency approval;
- what needs dataset download approval;
- what needs offline eval/gate approval;
- what needs training approval;
- what needs future company data.

Recommended first path after this documentation package: rule-based baseline design/spec. It is advisor-friendly, requires no company data, and can be specified without running eval.

## Current Production Context

- Logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Runtime: `torch_joint`
- Architecture: `hierarchical_v1`
- Init method: `flat_teacher_distillation_v1`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`
- PPO continuous action dimension: `5`
- DQN external discrete action count: `48`
- Ladder: 250k PASS, 500k PASS, 1M PASS
- Equal-budget residual-watch gate: PASS
- Ops bundle: `OPS_BUNDLE_CLEAN`

## Benchmark Family 1: Internal Equal-Budget Comparator

Objective:

- Preserve the fair old production vs hierarchical v1 comparison as the baseline evidence table.

What it proves:

- Under the same scenario dir, 20 episodes, seed 42, CPU deterministic execution, hierarchical v1 passed the same gate and improved service in seven of eight scenarios while clearing mixed-stress service regression.

What it cannot prove:

- Real-world optimality, live integration readiness, or secondary-fleet/reorder economics.

Required data:

- Existing equal-budget summaries only for read-only discussion.

Required code/dependency:

- None for documentation.
- Fresh rerun would require `src.eval.evaluate_real_world_scenarios` and `src.eval.check_long_run_gate`.

Safe-now or approval-required:

- Safe now for read-only reporting.
- Approval required for any rerun because it is offline eval/gate and writes fresh eval outputs.

Expected metrics:

- service, lateness, dispatch rate, dispatch success, no-work per step, route failure, no_vehicle, already_assigned, action 24/32 rates.

Output artifact:

- Existing: `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`.
- Future rerun: fresh eval dirs and gate JSON only with approval.

Stop conditions:

- No explicit eval/gate approval.
- Output dirs not fresh.
- Protected hash drift.

Priority:

- P0 as current evidence baseline.

Recommended advisor wording:

> We compare the new model against old production under equal episode/seed/scenario budget before claiming improvement.

## Benchmark Family 2: Rule-Based Dispatch Baselines

Objective:

- Create transparent, human-understandable baselines that show whether learned policy behavior beats simple logistics rules inside the simulator contract.

Candidate rules:

- FIFO + shortest + primary
- earliest due date + shortest
- premium-first
- low-congestion under disruption
- stock-threshold reorder

What it proves:

- Whether the learned policy adds value over understandable operations heuristics in the same scenario family.

What it cannot prove:

- Real-world optimum, carrier economics, inventory finance, or live integration safety.

Required data:

- Existing scenario definitions and simulator state abstractions.

Required code/dependency:

- Future baseline policy adapter and metric comparator.
- No external dependency required in the likely design.

Safe-now or approval-required:

- Safe now to design/spec only.
- Implementation requires separate code approval.
- Any scenario run requires offline eval approval and fresh output dirs.

Expected metrics:

- service, lateness, dispatch success, route failure, no_vehicle, no-work, action mix, fleet mix, reorder mix.

Output artifact:

- Future design: rule definitions and expected telemetry schema.
- Future run: fresh baseline comparison report.

Stop conditions:

- Rule semantics require changing `env_5pl` physics.
- Eval run not approved.
- Output path would edit existing eval outputs.

Priority:

- P1 and recommended first concrete benchmark path.

Recommended advisor wording:

> Before adding heavier solvers, we compare the RL policy to rules a logistics operator can understand.

## Benchmark Family 3: OR-Tools Routing Baseline

Objective:

- Compare route-side behavior against a known operations-research solver where mapping is faithful.

What it proves:

- Route/time/distance/service tradeoff relative to a classical routing optimizer on compatible instances.

What it cannot prove:

- Dispatch failure semantics, fleet economics, reorder economics, or full 5PL control quality unless the benchmark is extended.

Required data:

- Synthetic exported route/order/fleet instance or approved public instances.

Required code/dependency:

- OR-Tools dependency, instance exporter, solver adapter, result comparator.

Safe-now or approval-required:

- Protocol only is safe now.
- Dependency install approval required.
- Benchmark run approval required.

Expected metrics:

- route time, route distance, vehicle count, late orders, infeasible orders, service/lateness proxy.

Output artifact:

- Future OR-Tools feasibility report and comparison JSON/markdown.

Stop conditions:

- Dependency not approved.
- Mapping from 5PL simulator to VRP/VRPTW instance is unfaithful.
- Output dir not fresh.

Priority:

- P1 after rule-based baseline design.

Recommended advisor wording:

> OR-Tools can test route-side reasonableness, but it is not a full substitute for the 5PL policy because it does not cover inventory/reorder/fleet economics by default.

## Benchmark Family 4: Amazon Last Mile Full Route-Proxy Analysis

Objective:

- Extend the small-sample route-proxy work to a larger public route sample.

What it proves:

- Whether shortest/fastest-like and low-congestion/reliability-like route preferences are directionally plausible in public last-mile route data.

What it cannot prove:

- Secondary-fleet choice, reorder-none economics, company cost tradeoffs, or live dispatch feasibility.

Required data:

- Amazon Last Mile Routing Research Challenge files or bounded approved subset.

Required code/dependency:

- Existing `scripts/real_world_calibration` helpers may be extended.
- No model training dependency.

Safe-now or approval-required:

- Protocol only is safe now.
- Download approval required for anything beyond already approved small sample.

Expected metrics:

- actual/greedy route ratio, travel-time asymmetry, stress buckets, route concentration proxy, invalid sequence score, time-window availability.

Output artifact:

- Future public route-proxy analysis report.

Stop conditions:

- Download size not approved.
- License/access unclear.
- Analysis starts claiming fleet/reorder economics from route-only data.

Priority:

- P1 for route-side validation after rule-based design.

Recommended advisor wording:

> Public route data can support route-choice plausibility, not full logistics economics.

## Benchmark Family 5: Solomon VRPTW

Objective:

- Compare time-window routing feasibility against a classic benchmark family.

What it proves:

- Whether route/time-window logic can be mapped and compared to standard VRPTW-style tasks.

What it cannot prove:

- Inventory, reorder, primary/secondary fleet economics, or company failure taxonomy.

Required data:

- Solomon VRPTW instances.

Required code/dependency:

- Instance parser and mapper into a route/time-window comparison harness.

Safe-now or approval-required:

- Protocol only is safe now.
- Dataset/download and benchmark run approval required if files are not already local and approved.

Expected metrics:

- service, time-window violations, vehicles, travel time/distance proxy, infeasible orders.

Output artifact:

- Future Solomon mapping and result report.

Stop conditions:

- Mapping distorts the simulator problem beyond advisor usefulness.
- Dataset access not approved.

Priority:

- P2.

Recommended advisor wording:

> Solomon gives a recognizable academic routing benchmark, but it only covers the route/time-window slice.

## Benchmark Family 6: Homberger VRPTW

Objective:

- Test larger time-window routing stress than Solomon.

What it proves:

- Scalability of a route/time-window comparison protocol on larger benchmark structures.

What it cannot prove:

- Full 5PL operational control or company economics.

Required data:

- Homberger VRPTW instances.

Required code/dependency:

- Parser/mapper and comparator, likely shared with Solomon work.

Safe-now or approval-required:

- Protocol only safe now.
- Download/run approval required.

Expected metrics:

- feasibility, vehicles, distance/time, time-window violations, runtime.

Output artifact:

- Future Homberger scale report.

Stop conditions:

- Runtime too high.
- Mapping becomes too artificial.

Priority:

- P2 after Solomon.

Recommended advisor wording:

> Homberger is useful for larger route stress once the smaller Solomon mapping is credible.

## Benchmark Family 7: CVRPLIB/CVRP

Objective:

- Test capacity-aware route sanity on standard CVRP-style instances.

What it proves:

- Capacity and vehicle-routing behavior under a standard route benchmark abstraction.

What it cannot prove:

- Time windows, inventory/reorder, dispatch failure taxonomy, route disruption, or company economics.

Required data:

- CVRPLIB public instances or approved local subset.

Required code/dependency:

- Parser/mapper and comparator.

Safe-now or approval-required:

- Protocol only safe now.
- Download/run approval required.

Expected metrics:

- route cost, vehicle count, capacity violations, infeasible demand.

Output artifact:

- Future CVRPLIB capacity sanity report.

Stop conditions:

- No meaningful mapping to the current 5PL abstractions.
- Dataset access not approved.

Priority:

- P3 supporting benchmark.

Recommended advisor wording:

> CVRPLIB is useful as a capacity routing sanity check, not as proof of full system behavior.

## Benchmark Family 8: Multi-Seed Robustness Over Existing 8 Scenarios

Objective:

- Quantify stability of current scenario outcomes across multiple random seeds.

What it proves:

- Whether PASS behavior is robust to seed variation within the same scenario family.

What it cannot prove:

- Real-world deployment, company economics, or external benchmark superiority.

Required data:

- Existing `configs/eval_scenarios`.

Required code/dependency:

- Existing eval/gate tooling.

Safe-now or approval-required:

- Protocol only safe now.
- Offline eval and gate approval required to run.

Expected metrics:

- mean/sd/min/max service, lateness, dispatch success, hard blockers, action 24/32 rates, top-action concentration.

Output artifact:

- Future fresh multi-seed eval dirs, summary JSON/CSV, markdown report, gate result.

Stop conditions:

- No eval approval.
- Output dirs already exist.
- Protected hash drift.

Priority:

- P1 after a decision to strengthen statistical evidence.

Recommended advisor wording:

> Multi-seed testing strengthens simulator evidence, but it is still simulator evidence.

## Benchmark Family 9: Runtime Latency Benchmark

Objective:

- Measure whether the current runtime path is suitable for interactive decision support.

What it proves:

- CPU cold-load and warm prediction latency for `load_torch_joint_policy` and `PolicyService.predict_joint` on synthetic observations.

What it cannot prove:

- Network latency, DB latency, live TMS/WMS/ERP latency, or concurrent production throughput.

Required data:

- Synthetic zero/random observation fixtures only.

Required code/dependency:

- Future small benchmark runner; no new external dependency expected.

Safe-now or approval-required:

- Protocol only safe now.
- Implementation/report output approval required.

Expected metrics:

- cold load time, warm predict p50/p95/p99, continuous shape/finite/bounded, discrete 0..47, memory footprint if feasible.

Output artifact:

- Future `reports/benchmark/runtime_latency/...` JSON/markdown.

Stop conditions:

- Benchmark starts a long-lived service.
- Reads private data.
- Writes protected artifacts.

Priority:

- P1 for productization readiness after rule-based benchmark design.

Recommended advisor wording:

> Latency testing is a productization benchmark, not a model-quality benchmark.

## Benchmark Family 10: Ablation Benchmark

Objective:

- Establish causal contribution of key design choices.

Comparisons:

- flat DQN vs hierarchical_v1
- distillation vs no distillation
- exact resume vs unsafe resume
- teacher retention vs hierarchical

What it proves:

- Which architecture/training-stability choices mattered under the project contract.

What it cannot prove:

- Real-world optimality or external SOTA without broader benchmarks.

Required data:

- New controlled training/eval branches.

Required code/dependency:

- Training configs, trainer, eval/gate tooling, run reports.

Safe-now or approval-required:

- Not executable now.
- Requires training approval, eval approval, fresh dirs, and architecture plan.

Expected metrics:

- scenario PASS, hard blockers, service/lateness, dispatch success, residual watches, action concentration, runtime compatibility.

Output artifact:

- Future ablation matrix and run reports.

Stop conditions:

- No training approval.
- No eval approval.
- No exact-resume-safe parent.
- Would reuse failed checkpoints as parents.

Priority:

- P2/P3 after simpler benchmarks or concrete research trigger.

Recommended advisor wording:

> Ablation is scientifically valuable but expensive and should follow clear hypotheses, not curiosity training.

## Benchmark Family 11: Historical Company Replay, Future Only

Objective:

- Replay policy decisions against historical company-like event streams after approved company data exists.

What it proves:

- Stronger pre-live evidence that policy recommendations align with real operational outcomes and failure/cost semantics.

What it cannot prove:

- Live causal lift without shadow testing or A/B-style operational study.

Required data:

- Approved private company data: orders, dispatch attempts, deliveries, routes, fleet/carriers, inventory/reorder, costs.

Required code/dependency:

- Future replay harness, schema validator, privacy controls, KPI mapper.

Safe-now or approval-required:

- Not safe-now.
- Requires company data approval, privacy approval, no-DB-write constraints, and separate replay plan.

Expected metrics:

- service, lateness, dispatch success, route failure, no_vehicle, stockout/backlog, fleet cost, reorder cost, action alignment.

Output artifact:

- Future historical replay report in fresh approved location.

Stop conditions:

- No private data approval.
- Missing join keys.
- Ambiguous timestamp semantics.
- Data owner restrictions unresolved.

Priority:

- P0 after company data exists, but out of scope now.

Recommended advisor wording:

> Historical replay is the strongest pre-live validation, but it must wait for approved data and privacy rules.

## Cross-Benchmark Stop Conditions

Stop if any action would:

- train a model;
- run offline eval or long-run gate without explicit approval;
- download data without size/license approval;
- install dependencies without approval;
- write into registry, production, baselines, DB, checkpoints, or existing eval outputs;
- create training configs;
- ingest private company data;
- prepare or send firm-facing data request material in this task;
- overclaim public route data as proof of fleet/reorder economics.

## Priority Roadmap

1. P0: preserve equal-budget comparator as current evidence.
2. P1: design/spec rule-based baselines first.
3. P1: plan OR-Tools route baseline after dependency decision.
4. P1: expand Amazon route-proxy analysis only after download approval.
5. P1: runtime latency benchmark after implementation/report approval.
6. P1: multi-seed robustness only after eval/gate approval.
7. P2/P3: ablation only after training approval and hypotheses.
8. Future P0: historical company replay only after company data approval.

## Final Classification

`BENCHMARK_PROTOCOL_V1_NO_RUN_READY`

