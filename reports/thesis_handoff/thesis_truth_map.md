# Thesis Truth Map

Generated UTC: `2026-06-21T22:38:57.155101+00:00`

Official locked English title: `Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration`

Official Turkish title status: `MANUAL_CONFIRMATION_REQUIRED`

## Current Verified Values

| Field | Verified value |
| --- | --- |
| Official locked English title | `Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration` |
| Official Turkish title | `MANUAL_CONFIRMATION_REQUIRED` |
| Logical production model | `joint_torch_v5_prod_hierarchical_v1_1m_20260611` |
| Runtime | `torch_joint` |
| DQN architecture | `hierarchical_v1` |
| Initialization method | `flat_teacher_distillation_v1` |
| Environment/action contract | `physical_reality_v5_route_candidate_visibility` |
| Observation dimension | `73` |
| PPO continuous-control count | `5` |
| DQN action count | `48` |
| Training step | `1000000` |
| Eval verdict | `PASS` |
| Hard blocker status | `zero` |
| Long-run gate verdict | `PASS` |
| Residual watches accepted | `True` |

## Evaluation Scenarios and Ladder

| Evaluation | Scenarios | Episode rows | PASS count | Hard blocker failure count |
| --- | ---: | ---: | ---: | ---: |
| hierarchical_250k | 8 | 160 | 8 | 0 |
| hierarchical_500k | 8 | 160 | 8 | 0 |
| hierarchical_1m | 8 | 160 | 8 | 0 |
| equal_budget_hierarchical | 8 | 160 | 8 | 0 |
| equal_budget_old_production | 8 | 160 | 8 | 0 |

### 1M Scenario Snapshot

| Scenario | Verdict | Service | Dispatch success/attempt | Dispatch rate | Route failures | Top actions |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline Normal Operations | PASS | 0.9901530513121075 | 1.0 | 0.3715277777777778 | 3 | `[{'action_id': 0, 'count': 2549, 'percentage': 0.44253472222222223}, {'action_id': 1, 'count': 1071, 'percentage': 0.1859375}, {'action_id': 25, 'count': 919, 'percentage': 0.1595486111111111}, {'action_id': 24, 'count': 544, 'percentage': 0.09444444444444444}, {'action_id': 29, 'count': 268, 'percentage': 0.04652777777777778}, {'action_id': 28, 'count': 226, 'percentage': 0.03923611111111111}, {'action_id': 33, 'count': 117, 'percentage': 0.0203125}, {'action_id': 32, 'count': 39, 'percentage': 0.0067708333333333336}, {'action_id': 37, 'count': 21, 'percentage': 0.0036458333333333334}, {'action_id': 36, 'count': 5, 'percentage': 0.0008680555555555555}]` |
| Demand Spike / Demand Volatility Scenario | PASS | 0.8586115225709623 | 1.0 | 0.7934027777777778 | 13 | `[{'action_id': 29, 'count': 1820, 'percentage': 0.3159722222222222}, {'action_id': 37, 'count': 1124, 'percentage': 0.1951388888888889}, {'action_id': 1, 'count': 1119, 'percentage': 0.19427083333333334}, {'action_id': 33, 'count': 577, 'percentage': 0.10017361111111112}, {'action_id': 45, 'count': 414, 'percentage': 0.071875}, {'action_id': 41, 'count': 369, 'percentage': 0.0640625}, {'action_id': 25, 'count': 243, 'percentage': 0.0421875}, {'action_id': 0, 'count': 71, 'percentage': 0.012326388888888888}, {'action_id': 36, 'count': 16, 'percentage': 0.002777777777777778}, {'action_id': 32, 'count': 7, 'percentage': 0.0012152777777777778}]` |
| High Holding-Cost Scenario | PASS | 0.9660184286420443 | 0.9991735537190083 | 0.44861111111111107 | 8 | `[{'action_id': 0, 'count': 2745, 'percentage': 0.4765625}, {'action_id': 24, 'count': 1674, 'percentage': 0.290625}, {'action_id': 32, 'count': 466, 'percentage': 0.08090277777777778}, {'action_id': 1, 'count': 431, 'percentage': 0.07482638888888889}, {'action_id': 28, 'count': 333, 'percentage': 0.0578125}, {'action_id': 29, 'count': 41, 'percentage': 0.007118055555555555}, {'action_id': 25, 'count': 35, 'percentage': 0.006076388888888889}, {'action_id': 33, 'count': 27, 'percentage': 0.0046875}, {'action_id': 37, 'count': 4, 'percentage': 0.0006944444444444445}, {'action_id': 36, 'count': 4, 'percentage': 0.0006944444444444445}]` |
| Lead-Time Volatility / Supplier Delay Scenario | PASS | 1.0 | 0.9994949494949494 | 0.36041666666666666 | 2 | `[{'action_id': 1, 'count': 3673, 'percentage': 0.6376736111111111}, {'action_id': 44, 'count': 1116, 'percentage': 0.19375}, {'action_id': 36, 'count': 750, 'percentage': 0.13020833333333334}, {'action_id': 32, 'count': 82, 'percentage': 0.01423611111111111}, {'action_id': 37, 'count': 76, 'percentage': 0.013194444444444444}, {'action_id': 45, 'count': 47, 'percentage': 0.008159722222222223}, {'action_id': 0, 'count': 11, 'percentage': 0.0019097222222222222}, {'action_id': 40, 'count': 5, 'percentage': 0.0008680555555555555}]` |
| Mixed Stress Scenario | PASS | 0.9423472160631124 | 0.9992932328922539 | 0.9866319444444445 | 10 | `[{'action_id': 24, 'count': 3342, 'percentage': 0.5802083333333333}, {'action_id': 32, 'count': 1361, 'percentage': 0.23628472222222222}, {'action_id': 28, 'count': 653, 'percentage': 0.11336805555555556}, {'action_id': 36, 'count': 321, 'percentage': 0.05572916666666667}, {'action_id': 0, 'count': 75, 'percentage': 0.013020833333333334}, {'action_id': 41, 'count': 3, 'percentage': 0.0005208333333333333}, {'action_id': 45, 'count': 3, 'percentage': 0.0005208333333333333}, {'action_id': 1, 'count': 2, 'percentage': 0.00034722222222222224}]` |
| Premium SLA Pressure Scenario | PASS | 1.0 | 0.9994949494949494 | 0.35677083333333337 | 2 | `[{'action_id': 1, 'count': 2409, 'percentage': 0.41822916666666665}, {'action_id': 28, 'count': 1809, 'percentage': 0.3140625}, {'action_id': 0, 'count': 1296, 'percentage': 0.225}, {'action_id': 29, 'count': 204, 'percentage': 0.035416666666666666}, {'action_id': 37, 'count': 29, 'percentage': 0.0050347222222222225}, {'action_id': 36, 'count': 13, 'percentage': 0.0022569444444444442}]` |
| Route Disruption / Congestion Scenario | PASS | 0.9007785656472469 | 0.9993198529411765 | 0.49861111111111106 | 8 | `[{'action_id': 0, 'count': 2779, 'percentage': 0.48246527777777776}, {'action_id': 32, 'count': 2737, 'percentage': 0.4751736111111111}, {'action_id': 1, 'count': 109, 'percentage': 0.01892361111111111}, {'action_id': 36, 'count': 78, 'percentage': 0.013541666666666667}, {'action_id': 24, 'count': 23, 'percentage': 0.003993055555555555}, {'action_id': 40, 'count': 19, 'percentage': 0.003298611111111111}, {'action_id': 28, 'count': 7, 'percentage': 0.0012152777777777778}, {'action_id': 25, 'count': 6, 'percentage': 0.0010416666666666667}, {'action_id': 33, 'count': 2, 'percentage': 0.00034722222222222224}]` |
| Vehicle Scarcity / Capacity Shock Scenario | PASS | 0.9486773613640545 | 1.0 | 0.8057291666666666 | 8 | `[{'action_id': 25, 'count': 1369, 'percentage': 0.2376736111111111}, {'action_id': 29, 'count': 1157, 'percentage': 0.20086805555555556}, {'action_id': 41, 'count': 863, 'percentage': 0.14982638888888888}, {'action_id': 33, 'count': 778, 'percentage': 0.13506944444444444}, {'action_id': 1, 'count': 745, 'percentage': 0.1293402777777778}, {'action_id': 0, 'count': 374, 'percentage': 0.06493055555555556}, {'action_id': 37, 'count': 187, 'percentage': 0.03246527777777778}, {'action_id': 24, 'count': 160, 'percentage': 0.027777777777777776}, {'action_id': 28, 'count': 104, 'percentage': 0.018055555555555554}, {'action_id': 45, 'count': 18, 'percentage': 0.003125}]` |

## Final Benchmark Rows and Core Outcomes

| Dimension | Final score | Limiter | Required claim boundary | Strongest evidence |
| --- | ---: | --- | --- | --- |
| simulator-production readiness | 4.7 | company telemetry still absent | Do not call this live deployment proof. | internal gate and production audits |
| old-production comparison | 4.6 | synthetic scenario comparator | Do not claim real-world superiority. | 20260611 equal-budget residual-watch eval |
| multi-seed robustness | 4.2 | not company telemetry replay | No global SOTA claim. | multi-seed robustness benchmark root |
| runtime latency | 4.7 | hardware dependent | Not a production SLA guarantee. | runtime latency repeated benchmark and replay smoke |
| rule-based baseline maturity | 4.5 | simulator baselines only | No claim that rules are industry optimal. | 5120 episode continuous-aware benchmark |
| inventory/reorder external evidence | 4.5 | company reorder economics absent | No no-reorder economic safety claim. | inventory/MABIM reports |
| route/stochastic public evidence | 4.4 | stochastic raw fields missing | No full 5PL route optimality claim. | route reference reports |
| fleet/dispatch external evidence | 4.2 | Chicago/OpenMines blockers and no company contracts | No secondary-fleet economics claim. | fleet dispatch upgrade sprint |
| public historical replay maturity | 4.45 | no propensities/rewards/full trajectories | Descriptive proxy replay only. | large replay reports |
| company-data validation readiness | 4.45 | no private extract ingested | No company validation claim. | company replay bridge and request package |
| SOTA pathway maturity | 4.45 | external blockers and no company OPE | Frame as pathway, not global SOTA. | SOTA pathway and final scorecard |
| thesis/advisor defensibility | 4.75 | citation placeholders remain | Do not overstate literature novelty. | final advisor/thesis docs |
| real-world deployment readiness | 3.25 | no company replay/OPE/live telemetry | Not live TMS/WMS/ERP ready. | production handoff and ops bundles |

## Public Replay Dataset Sizes

| Dataset | Rows | Predictions | Limitation / note |
| --- | ---: | ---: | --- |
| LaDe | 31415 | 31415 | large sample improves descriptive public replay stability but does not add propensities, company rewards, or full trajectories |
| NYC_HVFHS | 100000 | 100000 | large sample improves descriptive public replay stability but does not add propensities, company rewards, or full trajectories |
| Olist | 96476 | 96476 | large sample improves descriptive public replay stability but does not add propensities, company rewards, or full trajectories |

## Current Claim Boundaries

- The work may be described as simulator-grounded autonomous supply-chain orchestration using a protected runtime path.
- The production artifact is active/promoted locally and passed the internal scenario and gate evidence recorded above.
- Public replay and public benchmark evidence are proxy/descriptive evidence only.
- Do not claim global state-of-the-art superiority.
- Do not claim live TMS/WMS/ERP deployment.
- Do not claim completed private-company validation.
- Do not interpret public replay as causal off-policy superiority.
