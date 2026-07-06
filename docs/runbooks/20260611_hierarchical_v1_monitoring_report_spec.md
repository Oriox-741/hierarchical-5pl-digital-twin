# Hierarchical V1 Monitoring Report Spec

Date: 2026-06-11

Final classification: `STEP1_MONITORING_REPORT_SPEC_READY`

## Report Purpose

This spec defines a future read-only monitoring report for the current
hierarchical v1 1M production model. The report should summarize production
identity, registry/manifest consistency, operating KPIs, residual watches, and
alert decisions.

This spec does not implement a report generator and does not authorize
training, offline eval, long-run gates, dataset download, registry mutation,
production mutation, baseline mutation, DB mutation, checkpoint mutation,
existing eval-output edits, or training configs.

## Intended Read-Only Scope

The future report may read:

- production telemetry exported for monitoring,
- `models/registry/active_models.json`,
- `models/registry/models.jsonl`,
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`,
- current production artifact hashes,
- current dashboard and release handoff,
- previously approved eval summaries as static references.

The future report must not write protected artifacts.

## Required Metadata

Each report instance must include:

- report timestamp and monitoring window,
- logical model id,
- runtime family,
- DQN architecture,
- hierarchical init method,
- production checkpoint path,
- active PPO and DQN registry ids,
- candidate PPO and DQN registry ids,
- contract,
- observation dimension,
- continuous action dimension,
- external discrete action count,
- source data extracts and row counts,
- denominator definitions,
- report generator version if one later exists,
- protected-state check status.

## JSON Schema

The report should be serializable as JSON using this top-level shape:

```json
{
  "report_metadata": {
    "report_timestamp_utc": "string",
    "monitoring_window_start_utc": "string",
    "monitoring_window_end_utc": "string",
    "logical_model_id": "joint_torch_v5_prod_hierarchical_v1_1m_20260611",
    "runtime_family": "torch_joint",
    "dqn_architecture": "hierarchical_v1",
    "hierarchical_init_method": "flat_teacher_distillation_v1",
    "contract": "physical_reality_v5_route_candidate_visibility",
    "observation_dim": 73,
    "continuous_action_dim": 5,
    "external_discrete_action_count": 48,
    "production_checkpoint": "models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt",
    "active_ppo_id": "17ba1d28-0054-4f7c-ae9a-34cd305ebb89",
    "active_dqn_id": "f87e10d6-479f-44fc-99d1-6925bc9cb346"
  },
  "protected_state": {},
  "scenario_metrics": [],
  "action_metrics": [],
  "residual_watches": [],
  "alerts": [],
  "decision": {
    "classification": "string",
    "reason": "string"
  }
}
```

## Per-Scenario Metrics Block

Each scenario or operating-regime row must include:

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

Required scenario metrics:

- service by scenario,
- lateness,
- dispatch success,
- no-work per step,
- delivered per step,
- route failure per step,
- no-vehicle per step,
- already-assigned per step,
- top action concentration.

## Per-Action Metrics Block

Each action row must include:

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

Required action metrics:

- no-current by action,
- no-unassigned by action,
- failed-noop by action,
- route failure by action,
- no-vehicle by action,
- already-assigned by action,
- top action concentration,
- action `24` rate,
- action `32` rate,
- secondary_fleet rate,
- reorder none rate.

## Residual-Watch Block

The report must always include the accepted watches:

```json
{
  "watch_id": "route_action_32_concentration",
  "action_id": 32,
  "baseline_context": "route_disruption_congestion equal-budget seed42 20ep",
  "equal_budget_baseline_rate": 0.475174,
  "current_rate": 0.0,
  "paired_operational_metrics": {
    "service_level": 0.0,
    "lateness": 0.0,
    "route_failure": 0,
    "no_current": 0,
    "no_unassigned": 0,
    "failed_noop": 0
  },
  "decision": "info|watch|warning|critical"
}
```

Required residual watches:

- route action `32` concentration,
- mixed action `24` concentration,
- top-action concentration warnings,
- mixed-success route-failure warnings,
- secondary_fleet economics watch,
- reorder none economics watch.

## Protected State Block

The report must include current read-only artifact checks:

```json
{
  "active_models_hash": "B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A",
  "models_jsonl_hash": "935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1",
  "production_joint_hash": "C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE",
  "production_manifest_hash": "FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A",
  "active_registry_matches_handoff": true,
  "production_manifest_matches_handoff": true,
  "train_eval_gate_process_found": false
}
```

Protected-state mismatch is a `critical` alert and should stop any automated
interpretation of monitoring metrics until resolved.

## Alert Levels

| Level | Meaning | Required action |
| --- | --- | --- |
| `info` | Normal movement, no threshold crossed. | Record only. |
| `watch` | Residual watch moved materially without operational degradation. | Inspect next window. |
| `warning` | Watch plus service, lateness, route, fleet, or cost degradation. | Open investigation; do not train automatically. |
| `critical` | Hard blocker, protected-state drift, invalid runtime, or forbidden process. | Stop and perform incident/read-only audit. |

## Example Report Skeleton

```json
{
  "report_metadata": {
    "logical_model_id": "joint_torch_v5_prod_hierarchical_v1_1m_20260611",
    "monitoring_window_start_utc": "2026-06-11T00:00:00Z",
    "monitoring_window_end_utc": "2026-06-11T23:59:59Z"
  },
  "protected_state": {
    "status": "clean"
  },
  "scenario_metrics": [
    {
      "scenario_or_regime": "route_disruption_like",
      "service_level": 0.0,
      "lateness_mean": 0.0,
      "dispatch_success": 0.0,
      "top_action_id": 32,
      "top_action_share": 0.0
    }
  ],
  "action_metrics": [
    {
      "action_id": 32,
      "rate_per_step": 0.0,
      "no_current": 0,
      "no_unassigned": 0,
      "failed_noop": 0
    }
  ],
  "residual_watches": [
    {
      "watch_id": "route_action_32_concentration",
      "decision": "info"
    }
  ],
  "alerts": [],
  "decision": {
    "classification": "MONITORING_REPORT_CLEAN",
    "reason": "No warning or critical thresholds crossed."
  }
}
```

## Final Classifications

Future report classifications:

- `MONITORING_REPORT_CLEAN`
- `MONITORING_REPORT_WARNINGS_ONLY`
- `MONITORING_REPORT_NEEDS_INVESTIGATION`
- `MONITORING_REPORT_PRODUCTION_RISK_FOUND`
- `MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY`
- `MONITORING_REPORT_PROTECTED_STATE_DRIFT`

No classification should authorize training or protected mutation by itself.

## Source Documents

- `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`

## Final Classification

`STEP1_MONITORING_REPORT_SPEC_READY`
