# Control Room V8 No-Map Product Dashboard Report

Status: independently approved
Final classification: `NO_MAP_5PL_DASHBOARD_READY`

## Objective

V8 rebuilds the Turkish control room as a no-map 5PL digital-twin decision-orchestration dashboard. The primary demo now focuses on PPO+DQN decisions, KPI timelines, 48-action analytics, scenario comparison, public replay, evidence maturity, and claim boundaries.

## Diagnosis

The synthetic map looked unconvincing and semantically confusing. Vehicle/order marker movement created the wrong expectation that the project had real GPS, route polyline, or live logistics-location data.

The project strength is not location rendering. Its real strength is PPO+DQN decision orchestration inside the 5PL simulator, KPI evidence, all-action analytics, public replay coverage, benchmark maturity, and explicit claim boundaries. V8 therefore removes the map from the primary demo and replaces it with a decision-centric dashboard.

## Implementation Summary

- Replaced the served primary UI with `5PL Dijital İkiz Karar Orkestrasyon Paneli`.
- Added no-map dashboard panels for operation summary, PPO controls, DQN decision card, 48-action analytics, KPI timeline, scenario comparison, public replay, evidence scorecard, event flow, company-data requirements, and claim boundaries.
- Kept the existing deterministic operation traces and V7 assignment-bound semantics intact.
- Added read-only API endpoints:
  - `GET /api/replay`
  - `GET /api/scorecard`
  - `GET /api/company-data-requirements`
- Added a `dashboard` payload to `/api/state`.
- Changed static/server target to `reports/demo_control_room_v8/`.
- Updated `start_control_room.bat` and `start_control_room.ps1` to launch V8.

## Boundary Checks

- No training run.
- No offline eval run.
- No long-run gate.
- No registry, production, baseline, DB, checkpoint, or existing eval-output mutation.
- No private company-data ingestion.
- No live TMS/WMS/ERP deployment claim.
- No company-data validation claim.
- No causal public replay superiority claim.
- No global superiority claim.

## Verification

Initial TDD evidence:

- RED check failed against V7 because the static export still contained the map UI, `/api/replay` was missing, and `/api/state` had no `dashboard` payload.
- Focused GREEN check:
  - `python -m py_compile scripts\control_room_server.py tests\benchmarks\test_control_room_dashboard.py` -> PASS
  - `python -m unittest tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v8_static_export_contains_no_map_product_dashboard_and_traces tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_api_dispatch_returns_json_payloads tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_control_room_state_advances_and_contains_orchestration_dashboard_entities tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_static_preview_and_primary_ui_are_turkish_and_not_debug_report -v` -> PASS

Full verification:

- Static export:
  - `python scripts\control_room_server.py --export-static reports\demo_control_room_v8\index.html` -> PASS, classification `NO_MAP_5PL_DASHBOARD_READY_WITH_SAFE_FALLBACK`.
- Compile:
  - `python -m py_compile scripts\control_room_server.py tests\benchmarks\test_control_room_dashboard.py` -> PASS.
- Full dashboard tests:
  - `python -m unittest tests.benchmarks.test_control_room_dashboard -v` -> PASS, 26 tests.
- Browser smoke:
  - Loaded `http://127.0.0.1:8788/`.
  - Verified all eleven V8 panels exist.
  - Verified 48 action cards render.
  - Verified public replay totals render: total `227.891`, LaDe `31.415`, NYC HVFHS `100.000`, Olist `96.476`.
  - Verified no `#mapSvg`, no `canvas`, no `.map-card`, no map/coordinate/polyline forbidden text.
  - Clicked `Tek adım`; playback advanced to step `1` and KPI/DQN state updated.
  - Verified no `<pre>` and no primary raw debug table.
- Browser screenshot path:
  - In-app browser verification succeeded, but its screenshot capture timed out through CDP.
  - Headless Chrome/Playwright was used only as screenshot fallback.
- Screenshot files:
  - `reports/demo_control_room_v8/screenshots/dashboard_overview.png`
  - `reports/demo_control_room_v8/screenshots/ppo_dqn_panel.png`
  - `reports/demo_control_room_v8/screenshots/action_analytics.png`
  - `reports/demo_control_room_v8/screenshots/public_replay.png`
  - `reports/demo_control_room_v8/screenshots/evidence_scorecard.png`
- Responsive smoke:
  - 390px viewport: `scrollWidth=390`, no horizontal overflow, 11 panels, 48 action cards, no map SVG.

Protected hash snapshot after implementation:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

These match the pre-edit snapshot recorded before V8 work.

## Reviewer

Independent reviewer verdict: `NO_MAP_5PL_DASHBOARD_APPROVED`

Reviewer notes:

- Active V8 served/exported bundle has no map, coordinate grid, fake GPS, route polyline, canvas, or primary map UI.
- Screenshots and static bundle read as a product dashboard, not a debug/report page.
- PPO/DQN panels, all 48 action analytics, public replay limitations, evidence scorecard, and company-data requirements are visible and clear.
- Overclaim boundaries are preserved.
- BAT launcher is ASCII-only and V8-oriented.
- Non-blocking note: stale inactive V6/V7 template strings remain in `scripts/control_room_server.py`, but the active V8 export/server bundle is clean and tested.
