# Step 3 Monitoring Report Generator Plan

Date: 2026-06-11

Plan classification: `STEP3_MONITORING_REPORT_GENERATOR_PLAN_READY`

## Executive Summary

Step 2 completed a read-only artifact-health CLI/report and returned
`ARTIFACT_HEALTH_CLEAN`. Step 3 should build the next safe-now tooling layer:
a monitoring report generator skeleton/validator using only synthetic or local
fixture telemetry.

The future generator should not ingest private company data. It should not read
live telemetry. It should not train, evaluate, gate, download data, mutate
registry, mutate production, mutate baselines, mutate DB, mutate checkpoints,
edit existing eval outputs, or create training configs.

The generator should prove that the monitoring report shape and alert logic are
implementable before any real telemetry integration exists. It should combine:

- the Step 2 artifact-health JSON,
- synthetic telemetry-like JSON/CSV rows,
- the Step 1 KPI dictionary,
- the Step 1 monitoring report spec.

Future execution command:

```text
/goal Read docs/goals/20260611_execute_step3_monitoring_report_generator_goal.txt and execute it exactly.
```

## Why Step 3 Comes After Artifact Health

Monitoring conclusions are unsafe if the monitored artifact has drifted. Step 2
solves that prerequisite by producing a machine-readable protected-state report:

- active registry check: `PASS`,
- `models.jsonl` candidate/active row check: `PASS`,
- production manifest check: `PASS`,
- production artifact hash check: `PASS`,
- protected path profile check: `PASS`,
- dashboard consistency check: `PASS`,
- process scan: `PASS`,
- runtime smoke: `PASS`,
- final artifact-health classification: `ARTIFACT_HEALTH_CLEAN`.

Step 3 should consume this artifact-health JSON as a protected-state gate. If
artifact-health is not clean, monitoring metrics should not be interpreted as a
policy signal; the monitoring report should classify as
`MONITORING_REPORT_PROTECTED_STATE_DRIFT`.

## Inputs

### Artifact-Health Report JSON

Default input:

`reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json`

Required fields:

- `final_classification`
- `active_registry_checks.status`
- `models_jsonl_checks.status`
- `production_manifest_checks.status`
- `production_file_hash_checks.status`
- `protected_path_profile_checks.status`
- `process_scan_result.status`
- `runtime_smoke_result.status`
- protected hashes and production identity fields where present

### Synthetic Telemetry Fixture

No private data is allowed. The future CLI should support:

- `--telemetry-json PATH`
- `--telemetry-csv PATH`
- `--use-built-in-synthetic-fixture clean|warning_action24|warning_action32|critical_no_current`

Tests should create temporary JSON/CSV fixture files. The first real skeleton
report should use a built-in clean fixture so the future goal writes only the
approved output report path and no extra fixture file.

### KPI Dictionary

Source:

`docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`

Fields and thresholds to encode:

- action `24` = `dispatch + shortest + secondary_fleet + none`
- action `32` = `dispatch + low_congestion + secondary_fleet + none`
- route action `32` equal-budget baseline rate: `0.475174`
- mixed action `24` equal-budget baseline rate: `0.580208`
- material concentration margin: `0.10` absolute rate above comparable baseline
- no-current/no-unassigned/failed-noop on action `24` or `32` is critical
- route failure, no-vehicle, already-assigned are watches unless paired with
  service/lateness degradation or repeated threshold crossing

### Monitoring Report Spec

Source:

`docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`

The future JSON report must use the spec top-level shape:

- `report_metadata`
- `protected_state`
- `scenario_metrics`
- `action_metrics`
- `residual_watches`
- `alerts`
- `decision`

## Proposed Future Files

The future executable goal should allow only these writes:

- `scripts/monitoring_report_generator.py`
- `tests/orchestration/test_monitoring_report_generator.py`
- `reports/monitoring/20260611_hierarchical_v1_synthetic_monitoring_report.json`
- `docs/runs/20260611_step3_monitoring_report_generator_report.md`
- `docs/00_PROJECT_DASHBOARD.md` links/status only

No persistent synthetic fixture file is needed. Tests should use temp files, and
the first skeleton run should use a built-in synthetic clean fixture.

## Report Output Schema

### `report_metadata`

Required fields:

- `report_timestamp_utc`
- `monitoring_window_start_utc`
- `monitoring_window_end_utc`
- `logical_model_id`
- `runtime_family`
- `dqn_architecture`
- `hierarchical_init_method`
- `contract`
- `observation_dim`
- `continuous_action_dim`
- `external_discrete_action_count`
- `production_checkpoint`
- `active_ppo_id`
- `active_dqn_id`
- `source_telemetry_kind`
- `source_telemetry_rows`
- `generator_version`
- `synthetic_data=true|false`

Synthetic skeleton defaults:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- runtime family: `torch_joint`
- DQN architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation/action: `73 / 48`
- continuous action dimension: `5`

### `protected_state`

Required fields:

- `artifact_health_classification`
- `active_registry_status`
- `models_jsonl_status`
- `production_manifest_status`
- `production_hash_status`
- `protected_path_profile_status`
- `runtime_smoke_status`
- `process_scan_status`
- `status`

Mapping:

- all artifact-health checks clean/pass -> `status=clean`
- any artifact-health drift/failure/block -> `status=drift`

### `scenario_metrics`

Each scenario/regime row:

```json
{
  "scenario_or_regime": "route_disruption_congestion",
  "rows": 0,
  "orders": 0,
  "decision_steps": 0,
  "service_level": 0.0,
  "lateness_mean": 0.0,
  "lateness_p95": 0.0,
  "dispatch_rate": 0.0,
  "dispatch_success": 0.0,
  "delivered_per_step": 0.0,
  "no_work_per_step": 0.0,
  "route_failure_per_step": 0.0,
  "no_vehicle_per_step": 0.0,
  "already_assigned_per_step": 0.0,
  "top_action_id": 32,
  "top_action_share": 0.0,
  "verdict": "pass|watch|warning|critical",
  "notes": []
}
```

### `action_metrics`

Each action/regime row:

```json
{
  "action_id": 24,
  "decoded_action": "dispatch + shortest + secondary_fleet + none",
  "scenario_or_regime": "mixed_stress",
  "attempts": 0,
  "rate_per_step": 0.0,
  "no_current": 0,
  "no_unassigned": 0,
  "failed_noop": 0,
  "route_failure": 0,
  "no_vehicle": 0,
  "already_assigned": 0,
  "dispatch_success": 0.0,
  "service_level": 0.0,
  "warning_threshold_crossed": false,
  "escalation_threshold_crossed": false
}
```

### `residual_watches`

Required watches:

- `route_action_32_concentration`
- `mixed_action_24_concentration`
- `top_action_concentration`
- `mixed_success_route_failure`
- `secondary_fleet_economics_unvalidated`
- `reorder_none_economics_unvalidated`

Each watch should include:

- `watch_id`
- `action_id` when applicable
- `baseline_context`
- `equal_budget_baseline_rate` when applicable
- `current_rate` when applicable
- `paired_operational_metrics`
- `decision=info|watch|warning|critical`
- `notes`

### `alerts`

Each alert:

- `level=info|watch|warning|critical`
- `alert_id`
- `scenario_or_regime`
- `action_id` when applicable
- `metric`
- `observed_value`
- `threshold`
- `reason`

### `decision`

Required fields:

- `classification`
- `reason`
- `recommended_action`

No decision classification may authorize training or protected mutation.

## Synthetic Fixture Design

The future CLI should parse telemetry-like rows. Minimum row schema:

```json
{
  "window_start_utc": "2026-06-11T00:00:00Z",
  "window_end_utc": "2026-06-11T01:00:00Z",
  "scenario_or_regime": "mixed_stress",
  "decision_steps": 100,
  "orders": 90,
  "served": 85,
  "late_orders": 3,
  "lateness_minutes": 12.0,
  "lateness_p95": 5.0,
  "delivered": 85,
  "dispatch_attempts": 80,
  "dispatch_successes": 78,
  "action_id": 24,
  "action_attempts": 20,
  "no_current": 0,
  "no_unassigned": 0,
  "failed_noop": 0,
  "route_failure": 0,
  "no_vehicle": 0,
  "already_assigned": 0,
  "secondary_fleet_attempts": 20,
  "reorder_none_decisions": 20
}
```

CSV columns should use the same names. Numeric fields must be finite and
non-negative. `decision_steps`, `orders`, and `action_attempts` are denominators
and must be positive where their corresponding rates are computed.

### Clean Case

Purpose:

- prove the generator can produce `MONITORING_REPORT_CLEAN`.

Properties:

- artifact-health report is `ARTIFACT_HEALTH_CLEAN`,
- action `24` and `32` rates are below material warning thresholds,
- no-current/no-unassigned/failed-noop are zero,
- service and lateness are acceptable,
- route failure/no-vehicle/already-assigned are zero or low.

### Warning Case

Purpose:

- prove action concentration warnings remain nonfatal without operational
  degradation.

Properties:

- action `24` rate in `mixed_stress` exceeds `0.580208 + 0.10`, or
- action `32` rate in `route_disruption_congestion` exceeds `0.475174 + 0.10`,
- service/lateness/dispatch success remain acceptable,
- no-current/no-unassigned/failed-noop remain zero,
- classification is `MONITORING_REPORT_WARNINGS_ONLY`.

### Critical Case

Purpose:

- prove hard action-quality blockers escalate.

Properties:

- no-current, no-unassigned, or failed-noop is nonzero on action `24` or `32`,
- classification is `MONITORING_REPORT_PRODUCTION_RISK_FOUND`,
- alert level includes `critical`.

### Protected Drift Case

Purpose:

- prove artifact-health status controls interpretation.

Properties:

- artifact-health JSON final classification is not `ARTIFACT_HEALTH_CLEAN`,
- classification is `MONITORING_REPORT_PROTECTED_STATE_DRIFT`,
- monitoring metrics are still summarized but not treated as model research
  trigger evidence.

### Malformed Input Case

Purpose:

- prove bad telemetry is blocked.

Properties:

- missing denominator field, negative counter, non-finite metric, unknown action,
  or invalid timestamp,
- classification is `MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY`.

## Classifications

Future monitoring report classifications:

- `MONITORING_REPORT_CLEAN`
- `MONITORING_REPORT_WARNINGS_ONLY`
- `MONITORING_REPORT_NEEDS_INVESTIGATION`
- `MONITORING_REPORT_PRODUCTION_RISK_FOUND`
- `MONITORING_REPORT_PROTECTED_STATE_DRIFT`
- `MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY`

Priority order:

1. malformed telemetry -> `MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY`
2. protected artifact-health drift -> `MONITORING_REPORT_PROTECTED_STATE_DRIFT`
3. critical action-quality blocker -> `MONITORING_REPORT_PRODUCTION_RISK_FOUND`
4. warning plus service/lateness/fleet/route degradation ->
   `MONITORING_REPORT_NEEDS_INVESTIGATION`
5. concentration or route/fleet/reorder watch only ->
   `MONITORING_REPORT_WARNINGS_ONLY`
6. no alerts above info -> `MONITORING_REPORT_CLEAN`

## Tests

The future implementation should use TDD and temp fixtures.

Required tests:

- clean synthetic report returns `MONITORING_REPORT_CLEAN`,
- action `24` warning returns `MONITORING_REPORT_WARNINGS_ONLY`,
- action `32` warning returns `MONITORING_REPORT_WARNINGS_ONLY`,
- no-current/no-unassigned/failed-noop on action `24` or `32` escalates to
  `MONITORING_REPORT_PRODUCTION_RISK_FOUND`,
- protected-state drift escalates to
  `MONITORING_REPORT_PROTECTED_STATE_DRIFT`,
- malformed input returns `MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY`,
- CLI writes only the requested output JSON in fixture mode.

Recommended additional tests:

- CSV and JSON fixtures produce equivalent aggregates,
- zero denominator is blocked,
- negative counters are blocked,
- unknown watched action decode is explicit,
- report includes all required top-level schema blocks.

## Independent Review Requirement

After future implementation and verification, spawn one gpt-5.5 read-only
reviewer.

Reviewer must inspect:

- `scripts/monitoring_report_generator.py`,
- tests,
- generated synthetic monitoring report JSON,
- run report,
- dashboard update,
- Step 1 KPI dictionary and monitoring report spec,
- Step 2 artifact-health report JSON.

Reviewer must not train, eval, gate, download datasets, ingest private data, or
mutate protected artifacts.

Future reviewer verdict exactly one:

- `STEP3_MONITORING_REPORT_GENERATOR_APPROVED`
- `STEP3_MONITORING_REPORT_GENERATOR_NEEDS_FIXES`
- `STEP3_MONITORING_REPORT_GENERATOR_BLOCKED`

## Non-Goals

- No live telemetry ingestion.
- No private company data read or write.
- No offline eval.
- No long-run gate.
- No model training.
- No dataset download.
- No registry mutation.
- No `active_models.json` or `models.jsonl` mutation.
- No production mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No existing eval-output edit.
- No training config creation.
- No 1.5M/2M/3M/5M/10M/100M.
- No public-route-only validation claim for secondary fleet or reorder
  economics.

## Protected No-Mutation Proof

This planning task only writes:

- `docs/plans/20260611_step3_monitoring_report_generator_plan.md`
- `docs/goals/20260611_execute_step3_monitoring_report_generator_goal.txt`
- `docs/00_PROJECT_DASHBOARD.md` links/status only

Pre-write protected hashes:

- `models/registry/active_models.json`:
  `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`:
  `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- production joint checkpoint:
  `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- production manifest:
  `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Pre-write protected path profiles:

| Path | File count | Total bytes |
| --- | ---: | ---: |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/production` | `8` | `328265276` |
| `models/eval` | `128` | `388267659` |

Pre-write process scan:

```text
NO_TRAIN_EVAL_GATE_AWS_OR_PRIVATE_DATA_PROCESS
```

Post-write verification must confirm protected hashes and profiles remain
unchanged.

## Exact Goal Command

```text
/goal Read docs/goals/20260611_execute_step3_monitoring_report_generator_goal.txt and execute it exactly.
```

## Final Classification

`STEP3_MONITORING_REPORT_GENERATOR_PLAN_READY`
