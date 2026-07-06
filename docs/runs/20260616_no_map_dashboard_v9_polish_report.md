# Control Room V9 No-Map Dashboard Polish Report

Status: independently approved
Final classification: `NO_MAP_DASHBOARD_V9_POLISH_READY`

## Objective

V9 polishes the V8 no-map control room into an advisor-ready Turkish product dashboard. It keeps the map removed, preserves deterministic trace/playback internally, simplifies visible mode labels, and removes static/live and KPI interpretation confusion.

## Diagnosis

V8 no-map direction is correct. Removing the synthetic map made the dashboard more academically honest and more product-like because the project strength is not fake geography, vehicle dots, route lines, or coordinate grids. The strongest surface is the 5PL digital-twin decision loop: PPO continuous controls, hierarchical DQN tactical actions, KPI trends, public replay coverage, benchmark evidence, and explicit claim boundaries.

However, V8 still had advisor-facing issues:

- Visible `Sunum Modu / Canlı Simülasyon Modu / statik önizleme` wording was confusing.
- Static-preview text appeared in contexts where users could interpret the live dashboard as a static artifact.
- KPI values rendered as `0.0%` could be misread as model failure at the initial step.
- Some labels remained half-English, especially scorecard dimensions and replay/action family fields.
- The dashboard needed simpler mode naming, cleaner Turkish copy, and less technical clutter.

## V9 Design Decisions

- Visible default mode is now `Demo Akışı`.
- Technical simulator mode is visible only as `Teknik Simülatör`.
- Public replay mode is visible as `Kamu Replay`.
- Backend enum names can remain internal but are mapped to clean Turkish UI labels.
- Live server HTML does not show the static-preview warning.
- Static export clearly says: `Bu statik önizlemedir; canlı dashboard için start_control_room.bat çalıştırın.`
- KPI rendering now uses `Başlangıç`, `Henüz hesaplanmadı`, and source labels such as `demo trace`, `simülatör metriği`, and `public replay metriği`.
- The no-map operation flow is shown as a pipeline: `Bekleyen`, `Karar verildi`, `Sevk edildi`, `Tamamlandı`, `Gecikti / riskli`.
- Scorecard values are framed as evidence maturity, not performance percentages.
- Company data requirements are grouped in plain Turkish field families.

## Boundary Checks

- No training run.
- No offline eval run.
- No long-run gate.
- No registry, production, baseline, DB, checkpoint, or existing eval-output mutation.
- No private company-data ingestion.
- No live deployment claim.
- No company-data validation claim.
- No causal public replay superiority claim.
- No global SOTA claim.
- No map, fake geography, coordinates, vehicle markers, route lines, or map grid reintroduced.

## Verification

- TDD red check: the new V9 tests initially failed against V8 because the UI still exposed `Sunum Modu`, static-preview copy, stale operation-flow labels, and missing KPI context.
- Focused V9 tests:
  - `python -m unittest tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v9_static_and_live_mode_copy_are_polished_and_distinct tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v9_primary_ui_polish_labels_and_boundaries_are_present tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v9_dashboard_payload_defaults_to_demo_flow_and_operation_pipeline -v` -> PASS.
- Compile:
  - `python -m py_compile scripts\control_room_server.py tests\benchmarks\test_control_room_dashboard.py` -> PASS.
- Full dashboard regression suite:
  - `python -m unittest tests.benchmarks.test_control_room_dashboard -v` -> PASS, 29 tests.
- Static export:
  - `python scripts\control_room_server.py --export-static reports\demo_control_room_v9\index.html` -> PASS, classification `NO_MAP_5PL_DASHBOARD_READY_WITH_SAFE_FALLBACK`.
- Browser smoke:
  - In-app browser loaded the live server, but the click path timed out through CDP.
  - Headless Microsoft Edge through Playwright was used as screenshot and interaction fallback.
  - Live server checks: default mode `Demo Akışı`; no static-preview warning; no `Sunum Modu`; no `Operasyon Kaydı Oynatıcı`; no map/canvas/map-card artifacts; 11 panels; 48 action cards; public replay total `227.891`; operation flow present; PPO/DQN text present; scorecard/company panels present.
  - Playback check: `Tek adım` advanced to step `1`, time `08:06`, and KPI/DQN panels updated.
  - Mobile smoke: viewport `390px`, `scrollWidth=390`, 11 panels, 48 action cards, no map artifacts.
- Screenshot files:
  - `reports/demo_control_room_v9/screenshots/dashboard_overview.png`
  - `reports/demo_control_room_v9/screenshots/ppo_dqn_panel.png`
  - `reports/demo_control_room_v9/screenshots/evidence_scorecard.png`
  - `reports/demo_control_room_v9/screenshots/public_replay.png`
  - `reports/demo_control_room_v9/screenshots/company_data_requirements.png`
- Protected hashes after implementation matched pre-edit hashes:
  - `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
  - `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
  - `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`
- Final process scan found no Python, Streamlit, AWS, train, eval, or gate process.

## Reviewer

First independent read-only reviewer verdict: `NO_MAP_DASHBOARD_V9_POLISH_NEEDS_FIXES`.

Fixes applied:

- Replaced this report's placeholder status and pending verification section with actual evidence.
- Removed the unreachable legacy V6/V7 map template block from `scripts/control_room_server.py`, including stale `mapSvg`, `Sunum Modu`, and `Operasyon Kaydı Oynatıcı` primary UI strings.

Final independent read-only re-review verdict: `NO_MAP_DASHBOARD_V9_POLISH_APPROVED`.

The final re-review confirmed that the V9 report records concrete verification, the stale legacy map/template symbols are gone from `scripts/control_room_server.py`, exported V9 assets have no map/canvas/debug markers, live/static warning behavior is split correctly, and probes confirmed `Demo Akışı`, 48 actions, replay total `227891`, evidence-maturity framing, and no visible forbidden overclaim phrases.
