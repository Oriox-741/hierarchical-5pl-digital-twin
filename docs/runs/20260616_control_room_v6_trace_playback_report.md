# Control Room V6 Trace-Based Operation Playback Report

Date: 2026-06-16

Final classification: `CONTROL_ROOM_V6_TRACE_PLAYBACK_READY`

## Objective

Control Room V6 turns the Turkish control room into a deterministic operation-record player. The default mode is still `SUNUM_MODU`, but the visible product framing is now `OPERASYON_KAYDI_OYNATICI`: a trace playback surface, not a live-company deployment claim.

## What Changed

- Added deterministic operation traces with stable hubs, orders, route candidates, vehicles, active route choices, congestion/disruption zones, KPI snapshots, PPO controls, DQN actions, event logs, and action history.
- Added playback API endpoints:
  - `GET /api/traces`
  - `GET /api/trace/{trace_id}`
  - `GET /api/playback/state`
  - `POST /api/playback/load`
  - `POST /api/playback/play`
  - `POST /api/playback/pause`
  - `POST /api/playback/reset`
  - `POST /api/playback/seek`
  - `POST /api/playback/tick`
- Preserved V5 semantics: GET calls are read-only; only POST playback controls advance or seek.
- Repaired step-level PPO display with explicit source text and per-control variance summaries.
- Repaired route/vehicle continuity by separating stable vehicle motion from the currently highlighted route decision.
- Updated the UI to V6 labels: operation trace selector, timeline slider, play/pause/reset/rewind/forward controls, PPO history, and `20260616v6` static cache marker.
- Kept `start_control_room.bat` as the recommended launcher and ASCII-only.

## Output Artifacts

- Static package: `reports/demo_control_room_v6/index.html`
- Static JS/CSS: `reports/demo_control_room_v6/app.js`, `reports/demo_control_room_v6/styles.css`
- Operation traces:
  - `reports/demo_control_room_v6/traces/baseline_normal_trace.json`
  - `reports/demo_control_room_v6/traces/mixed_stress_trace.json`
  - `reports/demo_control_room_v6/traces/route_disruption_congestion_trace.json`
- Screenshots:
  - `reports/demo_control_room_v6/screenshots/replay_overview.png`
  - `reports/demo_control_room_v6/screenshots/map_playback.png`
  - `reports/demo_control_room_v6/screenshots/ppo_dqn_trace.png`
  - `reports/demo_control_room_v6/screenshots/public_replay.png`

## Verification

- `python -m py_compile scripts\control_room_server.py tests\benchmarks\test_control_room_dashboard.py` -> PASS
- `python -m unittest tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v6_operation_traces_are_deterministic_and_schema_valid tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v6_playback_api_is_read_only_and_tick_seek_work tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v6_static_export_contains_trace_player_ui_and_traces -v` -> PASS
- `python scripts\control_room_server.py --export-static reports\demo_control_room_v6\index.html` -> PASS
- `python -m unittest tests.benchmarks.test_control_room_dashboard -v` -> PASS, 20 tests
- Browser DOM smoke against `http://127.0.0.1:8786` -> PASS
- API smoke:
  - repeated `GET /api/playback/state` did not advance `step`
  - `POST /api/playback/play` + `POST /api/playback/tick` advanced one step
  - paused tick did not advance
  - seek to step 44 worked
  - load `mixed_stress` worked
  - playback mode returned `OPERASYON_KAYDI_OYNATICI`
  - PPO source returned `deterministic_demo_policy`

## Protected No-Mutation Proof

- No training, offline evaluation, long-run gate, registry update, production promotion, baseline update, DB mutation, checkpoint mutation, or dataset download was run.
- Registry hashes recorded after work:
  - `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
  - `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- Production directories remained at existing timestamps:
  - `joint_torch_v5_balanced_retention_ft_200k_20260608`: 2026-06-09 05:38:51
  - `joint_torch_v5_prod_hierarchical_v1_1m_20260611`: 2026-06-11 14:53:41
  - `not_allowed_for_rule_baseline`: 2026-06-13 07:04:20
- DB files were read-only inspected; no DB writes were made.
- Final process scan found no active `train_joint`, offline eval, long-run gate, AWS S3, or `control_room_server` process.

## Notes

V6 is a deterministic demonstration and assurance surface. It does not claim live TMS/WMS/ERP integration, company-data validation, causal OPE, global SOTA, or live autonomous deployment.
