# Non-Route Benchmark Expansion Plan

Date: 2026-06-13

Status: planning package only. This plan does not authorize dataset download, dependency installation, benchmark execution, offline eval, long-run gate, training, private company-data ingestion, or protected artifact mutation.

Recommended classification: `NON_ROUTE_BENCHMARK_EXPANSION_PLAN_READY`

## Executive Summary

The current benchmark package already covers:

- old-production comparison inside the 5PL simulator;
- rule-based tactical baselines;
- Amazon Last Mile route-side plausibility;
- OR-Tools route-only classical references;
- multi-seed robustness;
- runtime latency;
- historical ablation synthesis.

The next rigor gap is not "more OR-Tools only." OR-Tools and PyVRP help route benchmarking, but the project is a wider 5PL decision-control system. The stronger next benchmark expansion should cover inventory/reorder, fleet dispatch, stochastic dispatch uncertainty, and congestion-control analogs without claiming those external systems are exact 5PL substitutes.

Priority recommendation:

1. Continuous-aware rule baselines in the native simulator first.
2. Inventory/reorder external benchmark path: OR-Gym and MABIM.
3. Fleet/dispatch analog path: FleetPy or OpenMines, chosen after dependency/data preflight.
4. Stochastic route/dispatch uncertainty: SVRPBench.
5. Route-solver reference strengthening: PyVRP, as a better route reference but not a non-route benchmark.
6. Congestion-control analogs: CityFlow/LibSignal, lower priority unless action 32 external congestion-control evidence becomes important.

## Benchmark Family Matrix

| Family | Tests project capability | Maps to | Required data/dependencies | Fair comparison design | What it proves | What it cannot prove | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OR-Gym / inventory management | Inventory and reorder decision quality under OR/RL environments | inventory, reorder, governance | Python package or source checkout; Gym-era compatibility risk; inventory management configs | Implement separate adapters that compare simple reorder rules vs project-inspired heuristic; do not use production checkpoint directly unless environment interface is mapped | Whether reorder heuristics behave sensibly on standard OR inventory tasks | Does not validate route/fleet/dispatch or company economics | High |
| MABIM / ReplenishmentEnv | Multi-agent, multi-echelon replenishment and SKU/warehouse behavior | inventory, reorder | Source package, SKU/warehouse configs, algorithm requirements; dependency preflight needed | Start with schema/readiness only, then synthetic or built-in env; compare base-stock, sS, and project heuristic | Whether project reorder logic can be discussed against multi-echelon inventory benchmark families | Does not validate delivery routing, fleet cost, or 5PL simulator physics | High |
| FleetPy | Fleet control, assignment, demand-responsive service, charging/fleet operations | fleet, dispatch, route secondary | Conda env/dependencies; scenario data; potentially substantial setup | Use built-in examples first; compare dispatch/assignment heuristics, not production checkpoint | Whether fleet dispatch/assignment ideas transfer to external fleet simulation | Ride-pooling/on-demand mobility is not parcel 5PL; inventory/reorder absent | Medium-high |
| SVRPBench | Stochastic routing under congestion, delays, accidents, time windows | route, dispatch uncertainty, reliability | Dataset/code access; likely nontrivial download and solver interface | Treat as stochastic route/dispatch uncertainty benchmark; compare route-choice heuristics and robust solvers under same instances | Whether action 32-style reliability logic is plausible under stochastic routing uncertainty | Does not validate inventory/reorder/fleet economics; still route-centric | Medium |
| PyVRP | Strong route solver reference and richer VRP variant support | route, route reference | `pyvrp` install; VRPLIB/Solomon-compatible data | Use as route-only reference against classical instances; do not compare full 5PL policy objective directly | Stronger route benchmark coverage than the current small OR-Tools suite | Not non-route; cannot validate PPO+DQN full decision-control | Medium |
| CityFlow / LibSignal | Congestion-control analog and multi-agent traffic control benchmark discipline | route congestion, governance, monitoring | CityFlow/SUMO/CBEngine dependencies; traffic networks and flows | Use only as analog for congestion-control metrics, not logistics dispatch; compare policy families in their native metrics | Whether congestion-control evaluation concepts can inform action 32 monitoring | Does not validate 5PL routing, fleet assignment, inventory, or order service | Low-medium |
| OpenMines | Truck dispatch under stochastic operational events and fleet constraints | dispatch, fleet, queueing, governance | `openmines` package, mine configs, dispatcher interface | Use built-in dispatchers first; map project-inspired dispatch heuristic only after interface study | Whether dispatch policy behavior can be compared in an external vehicle-dispatch simulator | Mining haulage is not 5PL parcel/logistics; inventory/reorder semantics differ | Medium |

## Source Notes

Public source review used official or primary project pages:

- OR-Gym: operations-research Gym environments with inventory management and network management tasks, including `InvManagement-v0/v1`.
- MABIM/ReplenishmentEnv: multi-agent benchmark for inventory management with replenishment environment, SKU data/config structure, and OR algorithm baselines.
- FleetPy: open-source fleet simulation framework for modeling and controlling vehicle fleets, with multi-fleet management and routing/user-assignment features.
- SVRPBench: NeurIPS 2025 Datasets and Benchmarks work for stochastic VRP with time-dependent congestion, delays, accidents, and time windows.
- PyVRP: high-performance VRP solver package with tutorials for capacities, time windows, profiles, optional clients, groups, and reloading.
- CityFlow: large-scale multi-agent traffic simulator with Python RL interface.
- LibSignal: cross-simulator traffic signal control library with unified metrics and baseline/RL model support.
- OpenMines: Python simulation environment for open-pit mining truck dispatch algorithms with probabilistic events.

## Detailed Expansion Path

### 1. OR-Gym / Inventory Management

Capability:

- Tests reorder/inventory decision logic outside the 5PL simulator.
- Good first external non-route target because project weakness is company-data-free reorder economics.

Project mapping:

- `reorder_fraction`
- `safety_stock_multiplier`
- reorder discrete modes: `none`, `conservative`, `aggressive`, `emergency`
- inventory coverage, stockout risk, holding pressure

Required preflight:

- Check package compatibility with current Python.
- Determine whether original OR-Gym or a maintained Gymnasium-compatible fork is safer.
- No install until user approval.

Fair comparison:

- Use native inventory reward/cost metrics.
- Compare base-stock/order-up-to/sS baselines against a project-inspired reorder heuristic.
- Do not run the production PPO+DQN checkpoint directly unless a formal adapter maps observations/actions correctly.

Priority: high.

### 2. MABIM / ReplenishmentEnv

Capability:

- Tests multi-echelon and multi-agent inventory/replenishment logic.
- Better than single-echelon inventory if the thesis wants a supply-chain benchmark.

Project mapping:

- inventory/reorder;
- SKU/site semantics;
- reorder governance and cost tradeoffs.

Required preflight:

- Inspect built-in environments and data sizes.
- Check dependencies from `algo_requirements.txt`.
- Decide whether synthetic built-in configs are enough.

Fair comparison:

- First run built-in OR baselines only after approval.
- Then add a project-inspired deterministic reorder policy.
- Report costs, service/fill rate, stockout, holding, order quantities.

Priority: high, after OR-Gym preflight.

### 3. FleetPy

Capability:

- External fleet assignment/dispatch simulation.
- Useful for secondary-fleet and vehicle-scarcity questions.

Project mapping:

- dispatch;
- fleet assignment;
- vehicle availability pressure;
- secondary/primary capacity analogs;
- route/user assignment.

Required preflight:

- Environment setup can be heavier than simple Python package install.
- Need determine whether built-in examples are small enough.

Fair comparison:

- Use FleetPy native dispatch/assignment metrics.
- Compare built-in fleet-control strategies against a transparent project-inspired dispatch/fallback rule.
- Do not claim direct parcel-logistics validation; it is a fleet analog.

Priority: medium-high.

### 4. SVRPBench

Capability:

- Tests robustness to stochastic routing conditions: time-dependent congestion, log-normal delays, accidents, and realistic time windows.
- Stronger action 32 evidence than static OR-Tools route-only tests.

Project mapping:

- route reliability;
- dispatch under uncertainty;
- low-congestion and high-resilience route choices.

Required preflight:

- Dataset/code size and install requirements.
- Whether route instances can be evaluated without full solver training.

Fair comparison:

- Compare route-choice heuristics and solvers under the same stochastic instances.
- Keep this route/uncertainty-specific.

Priority: medium.

### 5. PyVRP

Capability:

- Stronger route-only solver reference than the existing OR-Tools smoke/suite.
- Useful for classical VRP literacy and advisor confidence.

Project mapping:

- route only;
- not inventory/fleet/reorder full control.

Required preflight:

- Install approval for `pyvrp`.
- Instance selection and objective definitions.

Fair comparison:

- Use PyVRP as a route solver reference on CVRP/VRPTW-like instances.
- Never compare full 5PL service-level policy to PyVRP distance objective as if they were the same task.

Priority: medium.

### 6. CityFlow / LibSignal

Capability:

- Congestion-control and multi-agent traffic signal evaluation analogs.
- Useful for monitoring discipline and congestion pressure thinking.

Project mapping:

- route congestion;
- action 32 monitoring;
- governance/benchmark methodology.

Required preflight:

- Simulator dependencies and scenarios.
- Decide whether LibSignal's unified interface is safer than direct CityFlow integration.

Fair comparison:

- Treat as analogy only.
- Do not map 5PL actions directly to traffic signals without a separate architecture decision.

Priority: low-medium.

### 7. OpenMines

Capability:

- External dispatch simulator for trucks under stochastic operational conditions.
- Closer to "fleet dispatch" than traffic signals, though domain differs from 5PL parcel logistics.

Project mapping:

- dispatch algorithm;
- fleet capacity;
- route/queueing delays;
- stochastic events;
- governance and monitoring.

Required preflight:

- Package install and built-in config size.
- Dispatcher interface study.

Fair comparison:

- Use built-in dispatchers such as naive/random/nearest/fixed-group/SPTF/SQ as references.
- Add project-inspired dispatcher only after skeleton tests.

Priority: medium.

## Recommended Sequencing

1. Implement continuous-aware rule baseline skeleton.
2. Run no benchmarks until separately approved.
3. Plan OR-Gym/MABIM preflight next, because reorder/inventory is the largest non-route evidence gap.
4. Plan FleetPy/OpenMines preflight for fleet dispatch evidence.
5. Treat SVRPBench/PyVRP as route/uncertainty strengthening, not the main non-route fix.
6. Keep CityFlow/LibSignal as analog-only unless congestion-control framing becomes thesis-critical.

## Non-Goals

- No dependency installation in this task.
- No dataset download in this task.
- No benchmark execution in this task.
- No production checkpoint use in external environments.
- No claim that any external benchmark exactly represents this 5PL digital twin.
- No SOTA claim without benchmark-specific protocol and leaderboard-quality evidence.

## Recommended Future Goals

Continuous-aware baseline skeleton:

```text
/goal Read docs/goals/20260613_execute_continuous_aware_baseline_skeleton_goal.txt and execute it exactly.
```

Later, after the skeleton is reviewed, create a separate preflight goal for OR-Gym/MABIM dependency and data-size inspection. Do not combine external benchmark installation with rule baseline implementation.
