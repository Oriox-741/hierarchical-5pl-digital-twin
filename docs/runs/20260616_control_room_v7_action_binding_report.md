# Control Room V7 Route-Bound Action Binding Report

Status: independently approved  
Final classification: `CONTROL_ROOM_V7_ACTION_BINDING_READY`

## Objective

V7 repairs the remaining semantic gap in the trace-based Turkish control room: route/vehicle movement must remain bound to the DQN action that created the assignment, and movement frames must not look like fresh DQN decisions.

## Diagnosis

V6 trace playback improved the deterministic presentation mode, but it still allowed the visible right-side DQN action panel to follow a global per-step action sequence while vehicles continued along an existing route. That was visually confusing: the map could show the same route assignment while the action card changed as if a new operation decision had occurred.

V7 separates decision events from movement frames. A DQN action now belongs to a specific decision event and assignment. The assignment binds together order, vehicle, route, DQN action, decoded action, and PPO controls until it is completed or superseded. PPO values also belong to decision events; movement frames carry the bound PPO controls forward and label them as deterministic presentation inference, not online learning.

## Implementation Summary

- Extended operation traces with `frame_type`, decision event ids, `active_assignment_id`, assignment lists, stable route ids, active route assignment ids, stress-zone visibility metadata, and Turkish phase labels.
- Bound DQN/PPO values to assignment records through `bound_dqn_action_id`, `bound_decoded_action`, and `bound_ppo_controls`.
- Updated playback state so the decision panel shows either the selected active assignment or the latest decision event.
- Added read-only `GET /api/assignment/{assignment_id}` and explicit `POST /api/playback/select_assignment`.
- Added `POST /api/playback/set_stress_zones`; hidden stress zones now report `stress_zones_visible=false`.
- Repaired event log and action history text so movement frames say that the vehicle continues and no new decision was produced.
- Labeled orange stress zones as `Sıkışıklık bölgesi` / `Kesinti riski` and added a visible toggle.
- Updated static export to `reports/demo_control_room_v7/`.
- Updated launchers to V7; `start_control_room.bat` remains ASCII-only and recommended.

## Static Package

- Static preview: `reports/demo_control_room_v7/index.html`
- JS/CSS: `reports/demo_control_room_v7/app.js`, `reports/demo_control_room_v7/styles.css`
- Operation traces:
  - `reports/demo_control_room_v7/traces/baseline_normal_trace.json`
  - `reports/demo_control_room_v7/traces/mixed_stress_trace.json`
  - `reports/demo_control_room_v7/traces/route_disruption_congestion_trace.json`
- Screenshots:
  - `reports/demo_control_room_v7/screenshots/control_room.png`
  - `reports/demo_control_room_v7/screenshots/assignment_action_binding.png`
  - `reports/demo_control_room_v7/screenshots/event_log.png`
  - `reports/demo_control_room_v7/screenshots/ppo_trace.png`

## Verification

- RED check:
  - Stress-zone toggle regression failed before fix: `stress_zones_visible` stayed true while zones were hidden.
- Focused V7 tests:
  - `python -m unittest tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v7_trace_binds_decisions_to_assignments_and_movement_frames tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v7_state_panel_uses_active_assignment_bound_decision tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v7_event_log_and_action_history_are_operational_turkish_text tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v7_stress_zones_are_labeled_toggleable_and_hidden_when_irrelevant tests.benchmarks.test_control_room_dashboard.ControlRoomDashboardV4Tests.test_v7_assignment_api_is_read_only_and_select_assignment_is_explicit -v` -> PASS
- Compile:
  - `python -m py_compile scripts\control_room_server.py tests\benchmarks\test_control_room_dashboard.py` -> PASS
- Full dashboard tests:
  - `python -m unittest tests.benchmarks.test_control_room_dashboard -v` -> PASS, 25 tests.
- Static export:
  - `python scripts\control_room_server.py --export-static reports\demo_control_room_v7\index.html` -> PASS.
- Browser/API smoke:
  - In-app browser loaded `http://127.0.0.1:8787/`, clicked `Oynat`, and verified movement frame event text with stable DQN action.
  - API smoke verified two GETs are read-only, step 18 is `decision_event`, step 19 is `movement_frame`, both keep `mixed_stress-ASG-02` and DQN 32, assignment API is read-only, stress zones hide/show through the toggle, and event text contains `yeni karar`.
  - Headless Chrome wrote four non-empty PNG screenshots.

## Evidence Snapshot

`mixed_stress` static trace:

- Step 18: `decision_event`, assignment `mixed_stress-ASG-02`, DQN 32.
- Step 19: `movement_frame`, assignment `mixed_stress-ASG-02`, DQN 32.
- Same assignment/action across movement frame: true.
- Operation phase label: `Talep alımı`, not raw internal phase text.

## Boundary Checks

The control room remains a deterministic presentation playback. It does not train, evaluate, run gates, register, activate, promote, update baselines, mutate DB, mutate checkpoints, ingest private company data, or claim live company deployment.

Protected hash snapshot during verification:

- `models/registry/active_models.json`: `b7c6e79749044b92639b8a27ef38483022fc21d9f0ee08743af88725e42aa60a`
- `models/registry/models.jsonl`: `935d8bf34adc8dd956a68fb57581ec860b02d74ec26434dd6c084e63055c38e1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: `fdaf8de495d90bdfd69bcc2ee3772eb6a09661e104a5127ea3cb01c6d02cf99a`

Phrase scan found overclaim terms only in negative boundary statements or forbidden-phrase guard lists, not as positive claims.

## Reviewer

Independent read-only reviewer verdict:

`CONTROL_ROOM_V7_ACTION_BINDING_APPROVED`
