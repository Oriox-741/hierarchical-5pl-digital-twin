# Visual Digital Twin Dashboard Report

Date: 2026-06-16

## Classification

`VISUAL_DIGITAL_TWIN_DASHBOARD_READY`

## Purpose

This task built a local visual dashboard for presenting the project as a multi-layered digital twin framework for autonomous supply-chain orchestration. The dashboard frames the simulated 5PL operational environment as the digital twin and frames PPO+DQN as the controller operating inside that environment.

Required statement included in the dashboard:

> The digital twin is the simulated 5PL operational environment; PPO+DQN is the controller inside it.

The dashboard is a visual interface for explaining and running safe demonstrations. It is autonomous inside the simulator only. It does not represent a live TMS/WMS/ERP deployment and does not use company data.

## Artifacts

| Artifact | Path |
| --- | --- |
| Streamlit dashboard script | `scripts/visual_digital_twin_dashboard.py` |
| Smoke tests | `tests/benchmarks/test_visual_digital_twin_dashboard.py` |
| Static HTML export | `reports/demo_dashboard/index.html` |
| Turkish user guide | `docs/reports/20260615_visual_dashboard_user_guide_tr.md` |

## Launch

Streamlit mode:

```powershell
streamlit run scripts/visual_digital_twin_dashboard.py
```

Standalone HTML export:

```powershell
python scripts/visual_digital_twin_dashboard.py --export-html reports/demo_dashboard/index.html
```

The default mode is safe static/synthetic demo mode. Production policy loading is not performed by default.

## Implemented Pages

| Page | Evidence |
| --- | --- |
| Overview | Shows title, model id, contract, 73-D observation, five PPO controls, 48 DQN actions, `hierarchical_v1`, and the required digital-twin/controller statement. |
| Scenario Runner | Discovers all runnable JSON scenarios from `configs/eval_scenarios` and marks documented aliases without a config as documented/not runnable. |
| Digital Twin Map / Zone View | Uses a synthetic zone grid with hubs, orders, vehicles, route candidates, congestion/disruption flags, and delivery zones. It does not invent company locations. |
| PPO + DQN Decision Trace | Shows a summarized 73-D observation, five PPO continuous outputs, DQN action id, decoded action factors, and the hierarchical DQN external-contract note. |
| All-48 Action Decoder | Shows actions `0..47` with dispatch, route family, fleet mode, reorder mode, interpretation, and action 24/32 watch labels. |
| Public Replay Action Distribution | Shows LaDe `31,415`, NYC HVFHS `100,000`, Olist `96,476`, total `227,891`, top actions, zero-count actions, and family distributions. |
| Public Replay Action Distribution | Also shows the full all-action public replay distribution for action IDs `0..47`. |
| KPI Timeline | Shows static equal-budget assurance metrics without running offline eval, including available no-vehicle/route-failure fields, unavailable no-current/no-unassigned markers, top-action rate, and action 24/32 watch fields. |
| Evidence and Limitations | Shows internal simulator gates, equal-budget comparison, public proxy evidence, company-data limits, no live deployment, and no public-replay overclaim. |

## Scenario Discovery

Runnable scenarios discovered from `configs/eval_scenarios`:

- `baseline_normal`
- `demand_spike_volatility`
- `high_holding_cost`
- `lead_time_volatility`
- `mixed_stress`
- `premium_sla_pressure`
- `route_disruption_congestion`
- `vehicle_scarcity_capacity_shock`

Documented alias handled without a runnable config:

- `low_congestion`

## Verification

| Check | Result |
| --- | --- |
| Red test observed | `python -m unittest tests.benchmarks.test_visual_digital_twin_dashboard -v` initially failed because `scripts.visual_digital_twin_dashboard` did not exist. |
| CLI regression observed | Direct script execution initially failed because `src` was not importable when invoked by path; a failing regression test was added before the bootstrap fix. |
| Focused smoke tests | `python -m unittest tests.benchmarks.test_visual_digital_twin_dashboard -v` passed: 6 tests. |
| Compile | `python -m py_compile scripts\visual_digital_twin_dashboard.py tests\benchmarks\test_visual_digital_twin_dashboard.py` passed. |
| HTML export | `python scripts\visual_digital_twin_dashboard.py --export-html reports\demo_dashboard\index.html` passed. |
| Streamlit smoke | Temporary localhost Streamlit server responded with HTTP 200 and was stopped. |
| Browser smoke | Temporary localhost static dashboard page showed all major sections, one SVG map, nine tables, required digital-twin statement, public replay boundary, and no forbidden advisory-system wording. |
| Fix recheck | Browser DOM recheck confirmed scenario selector, action filters, 48 action rows, all-action replay section, delivery zones, KPI watch fields, and no forbidden advisory-system wording. |
| Protected registry hashes | `models/registry/active_models.json` and `models/registry/models.jsonl` remained unchanged during the task. |
| Final process scan | Clean after verification: no train/eval/gate/AWS/LibreOffice/Python leftovers. |

## Protected Paths

No registry, production, baseline, DB, checkpoint, or existing eval-output path was intentionally modified. Writes were limited to the allowed dashboard script, test, docs, dashboard report, and `reports/demo_dashboard`.

## Reviewer

Initial independent reviewer verdict: `VISUAL_DIGITAL_TWIN_DASHBOARD_NEEDS_FIXES`

Final independent reviewer verdict: `VISUAL_DIGITAL_TWIN_DASHBOARD_APPROVED`
