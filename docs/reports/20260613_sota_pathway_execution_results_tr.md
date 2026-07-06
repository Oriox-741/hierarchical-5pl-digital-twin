# SOTA Pathway Execution Results

Tarih: 2026-06-13

## Final Classification

`SOTA_PATHWAY_EXECUTION_PARTIALLY_READY_WITH_EXTERNAL_BLOCKERS`

## Ne Tamamlandı?

| Track | Durum | Ana çıktı |
| --- | --- | --- |
| Continuous-aware rule baselines | READY | 5120 episode row |
| Inventory OR-Gym compatible path | READY | `gym-invmgmt` 20 episode benchmark |
| MABIM/ReplenishmentEnv | READY | public source benchmark |
| Fleet/dispatch analog | READY | FleetPy example dispatch run |
| PyVRP route reference | READY | 5 CVRPLIB success, 3 VRPTW format blocker |
| SVRPBench stochastic routing | BLOCKED | PyPI/source path unavailable |
| Congestion analog | DEFERRED | CityFlow PyPI unavailable, source-build deferred |
| CODEX-5PL suite spec | READY | formal benchmark suite spec |

## Continuous-Aware Result

Full benchmark completed:

- 8 baselines
- 8 scenarios
- 20 episodes
- 4 continuous modes
- 5120 episode rows

Mean service stayed around `0.9876-0.9879` across modes. PPO-assisted mode remained close to neutral/heuristic, with slightly higher dispatch success. Heuristic continuous did not materially improve baseline discrimination in this scenario set.

## Inventory Evidence

Two public inventory paths now exist:

- `gym-invmgmt`: conservative reorder had the lowest cost in the tested serial environment.
- MABIM: no-reorder stress floor failed by final balance, while `(s,S)` and project-inspired reorder policies were positive.

This improves reorder-side evidence but does not prove real company reorder economics.

## Fleet/Dispatch Evidence

FleetPy public example scenario ran successfully after installing declared geospatial runtime dependencies.

Important metrics:

- 93 users
- served online users: 93%
- average waiting time: 151.996
- fleet utilization: 74.579%
- empty vkm: 34.433%
- shared rides: 32.258%

This validates a public fleet-dispatch analog, not full 5PL behavior.

## Route Evidence

PyVRP solved five CVRPLIB instances within about 2 seconds each. Three extracted VRPTW TXT files were blocked because PyVRP rejected their format as non-VRPLIB compatible.

This strengthens route-only evidence but does not validate dispatch/fleet/reorder composition.

## External Blockers

SVRPBench:

- `pip install svrpbench` failed with no matching distribution.
- Search found the paper but no directly runnable source/package path inside the sprint.

CityFlow:

- `pip install cityflow` failed with no matching distribution.
- Source-build traffic-signal work is analog-only and deferred.

## Advisor-Safe Interpretation

The project now has a stronger evidence package across internal 5PL simulation, rule baselines, route references, inventory references, and fleet-dispatch analogs. It is still not a global SOTA claim and not a live TMS/WMS/ERP deployment claim.

## Company Data Still Needed

Company data remains required for:

- secondary fleet economics
- reorder-none economics
- true dispatch failure causes
- cost/service tradeoff calibration
- real route/fleet/inventory joint validation

## Classification

`SOTA_PATHWAY_EXECUTION_PARTIALLY_READY_WITH_EXTERNAL_BLOCKERS`
