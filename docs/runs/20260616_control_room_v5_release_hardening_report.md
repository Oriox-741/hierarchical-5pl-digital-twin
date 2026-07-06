# Control Room V5 Release Hardening Report

## Classification

`CONTROL_ROOM_V5_RELEASE_READY_WITH_BAT_LAUNCHER`

## Executive Summary

V5 hardens the Turkish control room into a deterministic presentation release. The default mode is now `SUNUM_MODU`, backed by golden presentation traces rather than live environment stepping or production-policy dependency.

The primary bug was confirmed: V4 mutated simulation state inside `GET /api/state` whenever `running=true`. Browser polling therefore advanced the simulator, and multiple tabs could accelerate the demo. V5 makes `GET /api/state` read-only and moves all time advancement to explicit POST controls.

## Root Cause And Fix

Observed V4 reproduction:

- Default mode: `CANLI_SIMULASYON_MODU`
- Initial step: `0`
- Repeated `GET /api/state` while running advanced steps: `1,2,3,4`

V5 changes:

- `GET /api/state`: read-only, never advances.
- `POST /api/control/tick`: advances exactly one step only when `running=true`.
- `POST /api/control/step`: advances exactly one step even when paused.
- `POST /api/control/start` and `pause`: idempotent running-flag controls.
- `POST /api/control/reset`: resets step/history and keeps mode-safe adapter state.
- Browser loop: polling reads state; running loop calls `/api/control/tick`; timer is single-instance guarded.
- Multiple-tab warning: visible UI warning when another presentation tab is detected.
- Asset cache busting: `app.js?v=20260616v5`, `styles.css?v=20260616v5`.
- Connection/error surface: visible connection status and error panel.
- Server static root: default launcher/server path now writes and serves `reports/demo_control_room_v5`, not the older V4 static directory.

## Default Presentation Mode

`SUNUM_MODU` is the default. It uses deterministic generated traces and does not attempt live env initialization or production-policy loading.

The UI labels the boundary:

`Sunum modu: doğrulanmış simülasyon izi oynatılır; canlı şirket verisi değildir.`

Live simulation remains available as an explicit mode only. If the live adapter cannot connect, V5 reports the blocker instead of silently corrupting the selected mode.

## Golden Traces

Generated under `reports/demo_control_room_v5/golden_traces/`:

- `baseline_normal_trace.json`
- `mixed_stress_trace.json`
- `route_disruption_congestion_trace.json`

Validation:

- Each trace has `120` steps.
- Each trace is marked deterministic.
- Required per-step fields are present:
  - `step`
  - `simulated_time`
  - `scenario_id`
  - `vehicles`
  - `orders`
  - `routes`
  - `active_route`
  - `ppo_controls`
  - `dqn_action_id`
  - `decoded_action`
  - `kpis`
  - `event_log_line`

## Artifacts

Server and launchers:

- `scripts/control_room_server.py`
- `scripts/package_control_room_exe.py`
- `start_control_room.bat`
- `start_control_room.ps1`

Static release:

- `reports/demo_control_room_v5/index.html`
- `reports/demo_control_room_v5/app.js`
- `reports/demo_control_room_v5/styles.css`
- `reports/demo_control_room_v5/golden_traces/`
- `reports/demo_control_room_v5/screenshots/`

Optional exe wrapper:

- `python scripts\package_control_room_exe.py --dry-run`
- PyInstaller is not required for release readiness. The supported ready path is the batch/PowerShell launcher.

## Browser And Screenshot Evidence

In-app browser DOM smoke verified:

- Title: `Çok Katmanlı Dijital İkiz Orkestrasyon Kontrol Odası`
- Default mode: `SUNUM_MODU`
- Top mode: `Sunum Modu`
- Connection: `Hazır`
- Vehicles: `4`
- Orders: `6`
- Routes: `3`
- Selected route: `1`
- KPI cards: `7`
- Action dictionary cards: `48`
- Public replay metrics: `4`
- Public replay top actions: `10`
- No visible raw `<pre>`/`<code>` debug block.
- No raw `{"state" ...}` JSON body.

The in-app browser screenshot API timed out during capture, so DOM smoke was performed through the in-app browser and screenshots were captured with local Chrome headless.

Screenshots:

- `reports/demo_control_room_v5/screenshots/control_room.png`
- `reports/demo_control_room_v5/screenshots/map_live.png`
- `reports/demo_control_room_v5/screenshots/action_trace.png`
- `reports/demo_control_room_v5/screenshots/public_replay.png`

PNG verification:

- `control_room.png`: `1440x1000`, nonblank.
- `map_live.png`: `1440x1000`, nonblank.
- `action_trace.png`: `1440x1000`, nonblank.
- `public_replay.png`: `1440x1000`, nonblank.

## HTTP Contract Smoke

Fresh V5 server on `127.0.0.1:8766`:

- Initial mode: `SUNUM_MODU`
- Repeated GET steps: `0,0,0`
- GET steps after start: `0,0`
- First running tick: step `1`
- Paused tick: remains step `1`
- Manual step while paused: step `2`

## Verification

Commands run:

- `python -m py_compile scripts\control_room_server.py scripts\package_control_room_exe.py tests\benchmarks\test_control_room_dashboard.py`
- `python -m unittest tests.benchmarks.test_control_room_dashboard -v`
- `python -m unittest tests.benchmarks.test_control_room_dashboard tests.benchmarks.test_visual_digital_twin_dashboard -v`
- `python scripts\control_room_server.py --export-static reports\demo_control_room_v5\index.html`
- `python scripts\package_control_room_exe.py --dry-run`
- Direct HTTP API smoke on `http://127.0.0.1:8766`.
- Direct server static smoke on `http://127.0.0.1:8767/`: V5 cache tag present, `SUNUM_MODU` present, no `demo_control_room_v4` path.
- In-app browser DOM smoke.
- Chrome headless screenshots.
- Golden trace structure validation.

Results:

- Control-room dashboard tests: `15/15` passed.
- Control-room + visual dashboard tests: `24/24` passed after the reviewer fix.
- `py_compile`: passed.
- Static export: passed.
- Package dry-run: passed.

## Protected No-Mutation Proof

No training, offline eval, long-run gate, registry update, active model mutation, production promotion, baseline update, DB mutation, checkpoint mutation, or private-data ingestion was run.

Registry hashes observed before and after the V5 work:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`

The V5 smoke server was used only for local HTTP/browser proof and is not a train/eval/gate process.

## Reviewer

Initial independent read-only reviewer verdict:

`CONTROL_ROOM_V5_RELEASE_HARDENING_NEEDS_FIXES`

Reviewer finding: `run_server()` still used `reports/demo_control_room_v4` as the default static root. This would make the supported launcher rewrite and serve V4 assets.

Applied fix:

- `run_server()` now uses `reports/demo_control_room_v5`.
- `run_server()` passes the active root into `write_static_assets()`.
- Added `test_run_server_uses_v5_static_directory`.
- Real server smoke confirmed `/` serves V5 content with no V4 path reference.

Final independent read-only reviewer verdict:

`CONTROL_ROOM_V5_RELEASE_HARDENING_APPROVED`

Reviewer note: `run_server()` now writes and serves `reports/demo_control_room_v5`, passes `root=root`, and the new regression test covers the corrected static directory.
