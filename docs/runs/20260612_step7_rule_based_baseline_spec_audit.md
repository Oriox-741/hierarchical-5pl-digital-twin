# Step 7 Rule-Based Baseline Spec Audit

Date: 2026-06-13

Final classification: `RULE_BASED_BASELINE_SPEC_READY`

## Scope

This task prepared a no-run rule-based baseline benchmark design package. It did not implement code and did not run any benchmark/eval/gate/training.

Allowed writes:

- `docs/plans/20260612_rule_based_baseline_benchmark_spec.md`
- `docs/plans/20260612_rule_based_baseline_implementation_plan_no_run.md`
- `docs/reports/20260612_tez_danismani_rule_based_baseline_aciklamasi.md`
- `docs/goals/20260612_execute_rule_based_baseline_skeleton_goal.txt`
- `docs/runs/20260612_step7_rule_based_baseline_spec_audit.md`
- `docs/00_PROJECT_DASHBOARD.md` link/status update only

## Files Read

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/plans/20260612_benchmark_protocol_v1_no_run.md`
- `docs/plans/20260612_first_benchmark_recommendation_plan.md`
- `docs/plans/20260612_project_development_master_plan_no_us_application.md`
- `docs/reports/20260612_tez_danismani_tek_sayfa_proje_el_notu.md`
- `docs/reports/20260612_tez_danismani_diyagram_paketi.md`
- `docs/reports/20260612_tez_danismani_operasyon_walkthrough_detayli.md`
- `docs/runs/20260612_1m_learning_efficiency_and_sufficiency_audit.md`
- `docs/plans/20260612_autonomous_simulation_capability_audit_and_demo_plan.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary.md`
- `docs/runbooks/20260611_hierarchical_v1_monitoring_report_spec.md`
- `docs/runbooks/20260611_codex_model_lifecycle_governance_runbook.md`
- `src/act/env_5pl.py`
- `src/act/discrete_action_mapper.py`
- `src/eval/real_world_scenario_arena.py`
- `configs/eval_scenarios/*.json`

## Why Rule-Based Baselines Are Next

The Step 6 benchmark protocol and first-benchmark recommendation both identify rule-based baseline design/spec as the safest first benchmark path. It is advisor-readable, requires no dependency install or data download, does not need company data, and can be specified without running simulator eval.

This design package prepares transparent baselines for:

- FIFO + shortest + primary + none
- Earliest due date + shortest + primary + none
- Premium-first + shortest + primary/secondary + none
- Low-congestion under disruption
- Conservative stock-threshold reorder
- Emergency stockout-prevention reorder
- Vehicle-scarcity primary-first/secondary-fallback
- High-holding no-overstock

The package maps those rules to the existing 48-action contract and explicitly preserves the current production contract: `physical_reality_v5_route_candidate_visibility`, observation `73`, external action `48`.

## What Was Not Run

No training was run.

No offline eval was run.

No long-run gate was run.

No benchmark was run.

No dependency was installed.

No dataset was downloaded.

No registry, production, baseline, DB, checkpoint, or existing eval-output mutation was performed.

No training config was created.

No private company data was requested or ingested.

No firm-facing data request email was prepared or sent.

No 1.5M/2M/3M/5M/10M/100M run was started.

## Pre-Write Protected State

- `models/registry/active_models.json`: size `314`, SHA256 `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: size `14142`, SHA256 `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: size `303413903`, SHA256 `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: size `4084`, SHA256 `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Protected tree profiles:

| Path | Files | Bytes |
| --- | ---: | ---: |
| `models/production` | `8` | `328265276` |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/eval` | `128` | `388267659` |

Pre-write process scan:

```text
NO_TRAIN_EVAL_GATE_AWS_PRIVATE_DATA_PROCESS
```

## Files Written

- `docs/plans/20260612_rule_based_baseline_benchmark_spec.md`
- `docs/plans/20260612_rule_based_baseline_implementation_plan_no_run.md`
- `docs/reports/20260612_tez_danismani_rule_based_baseline_aciklamasi.md`
- `docs/goals/20260612_execute_rule_based_baseline_skeleton_goal.txt`
- `docs/runs/20260612_step7_rule_based_baseline_spec_audit.md`
- `docs/00_PROJECT_DASHBOARD.md`

## Verification

Final verification completed after writing:

- all allowed files exist;
- required sections are present;
- dashboard links point to the new package;
- protected hashes/tree profiles remain unchanged;
- no train/eval/gate/AWS/private-data process is running.

Post-write protected state:

- `models/registry/active_models.json`: size `314`, SHA256 `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: size `14142`, SHA256 `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: size `303413903`, SHA256 `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: size `4084`, SHA256 `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Post-write protected tree profiles:

| Path | Files | Bytes |
| --- | ---: | ---: |
| `models/production` | `8` | `328265276` |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/eval` | `128` | `388267659` |

Post-write process scan:

```text
NO_TRAIN_EVAL_GATE_AWS_PRIVATE_DATA_PROCESS
```

## Independent Review

Reviewer verdict:

```text
RULE_BASED_BASELINE_SPEC_APPROVED
```

Allowed reviewer verdicts:

- `RULE_BASED_BASELINE_SPEC_APPROVED`
- `RULE_BASED_BASELINE_SPEC_NEEDS_FIXES`
- `RULE_BASED_BASELINE_SPEC_BLOCKED`

## Final Classification

`RULE_BASED_BASELINE_SPEC_READY`
