# Control Room V4 Report

## Classification

`PRODUCT_CONTROL_ROOM_V4_READY_WITH_BAT_LAUNCHER`

## Executive Summary

V4 replaces the Streamlit dashboard as the primary visual demo with a local browser-based Turkish control room. The implementation uses a Python standard-library HTTP server plus static HTML/CSS/JavaScript under `reports/demo_control_room_v4/`.

The app presents the hierarchical v1 1M model as a simulator-internal PPO+DQN controller inside a 5PL digital twin. It does not claim live TMS/WMS/ERP integration, company-data validation, real company locations/orders, or causal public replay superiority.

## V3 Diagnosis

The V3 Streamlit surface was functional but demo-weak:

- It still felt like a report/debug dashboard rather than a product control room.
- The map area was too small for a live operations demo.
- Raw tables and JSON-style inspection hurt the product narrative.
- Streamlit chrome and widget language made the experience feel generic.

V4 keeps the old Streamlit implementation only as a fallback and makes `scripts/control_room_server.py` the primary demo entry point.

## Architecture

- Server: `scripts/control_room_server.py`
- Primary launchers:
  - `start_control_room.bat`
  - `start_control_room.ps1`
- Static preview:
  - `reports/demo_control_room_v4/index.html`
  - `reports/demo_control_room_v4/app.js`
  - `reports/demo_control_room_v4/styles.css`
- Sample trace:
  - `reports/demo_control_room_v4/sample_trace.json`

API endpoints implemented:

- `GET /`
- `GET /app.js`
- `GET /styles.css`
- `GET /api/status`
- `GET /api/scenarios`
- `GET /api/actions`
- `GET /api/state`
- `POST /api/control/start`
- `POST /api/control/pause`
- `POST /api/control/reset`
- `POST /api/control/step`
- `POST /api/control/set_scenario`
- `POST /api/control/set_speed`
- `POST /api/control/set_mode`
- `POST /api/control/set_policy`

## Modes

- `CANLI_SIMULASYON_MODU`: connects to `env_5pl` through the existing real-world scenario arena where available. Smoke check confirmed `env_connected=true`.
- `GUVENLI_GORSEL_DEMO_MODU`: deterministic visual demo with synthetic map coordinates and simulator-style KPIs.
- `KAMU_REPLAY_MODU`: descriptive public replay distribution surface. It is not OPE and not company-data validation.

Map coordinate notice is shown in the UI:

`Harita koordinatları görselleştirme amaçlı sentetik koordinatlardır; gerçek şirket lokasyonu içermez.`

## Live Adapter Result

Direct smoke:

- Mode: `CANLI_SIMULASYON_MODU`
- Env connected: `true`
- Env blocker: empty
- Read-only production policy step: OK
- DQN action from read-only policy smoke: legal action in `0..47`
- PPO vector length: `5`

Observed third-party warning: Gym deprecation warning for NumPy 2.0 compatibility guidance. It did not block the live adapter smoke.

## UI Evidence

Screenshots:

- `reports/demo_control_room_v4/screenshots/control_room.png`
- `reports/demo_control_room_v4/screenshots/map_live.png`
- `reports/demo_control_room_v4/screenshots/action_trace.png`
- `reports/demo_control_room_v4/screenshots/public_replay.png`

Browser DOM smoke verified:

- Turkish title present.
- Scenario rendered as `Normal Operasyon`.
- Status rendered as `Duraklatıldı`.
- Step rendered.
- 4 vehicle nodes, 6 order nodes, 3 route nodes.
- KPI cards rendered.
- DQN action text rendered.
- No Streamlit text.
- No visible raw `<pre>` block.

The in-app browser screenshot API timed out on capture, so screenshots were captured with local Chrome headless after DOM verification.

## Action and Replay Coverage

- 48 actions are loaded through the project discrete action mapper.
- Action 24 and action 32 are flagged as watched actions.
- Public replay totals:
  - Total: `227,891`
  - LaDe: `31,415`
  - NYC HVFHS: `100,000`
  - Olist: `96,476`

The UI labels public replay as descriptive proxy evidence only.

## Verification

Commands run:

- `python -m py_compile scripts\control_room_server.py tests\benchmarks\test_control_room_dashboard.py`
- `python -m unittest tests.benchmarks.test_control_room_dashboard -v`
- `python scripts\control_room_server.py --export-static reports\demo_control_room_v4\index.html`
- Direct live adapter smoke with `ControlRoomApp`.
- HTTP server smoke on `http://127.0.0.1:8765/api/status`.
- Browser DOM smoke through the in-app browser.
- Chrome headless screenshot capture.

Focused test result:

- `tests.benchmarks.test_control_room_dashboard`: 8 tests, all passed.

## Protected No-Mutation Proof

No training, offline eval, gate, registry update, production promotion, baseline update, DB mutation, checkpoint mutation, or private-data ingestion was run.

Registry hashes after implementation:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`

The V4 local server was stopped after smoke verification. An older Streamlit fallback process was observed and left untouched.

## Reviewer

Independent read-only review verdict:

`PRODUCT_CONTROL_ROOM_V4_APPROVED`

Reviewer notes: V4 is the primary non-Streamlit demo path, uses stdlib HTTP server plus HTML/CSS/JS, includes Turkish product-like map/KPI/PPO+DQN/action/replay/evidence panels, passes 8/8 focused tests, preserves public replay totals, and leaves protected registry hashes unchanged.
