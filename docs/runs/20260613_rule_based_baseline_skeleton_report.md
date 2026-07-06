# Rule-Based Baseline Skeleton Report

Date: 2026-06-13

Final classification: `RULE_BASED_BASELINE_SKELETON_READY`

## Scope

Implemented the skeleton-only rule-based baseline package requested by:

`docs/goals/20260612_execute_rule_based_baseline_skeleton_goal.txt`

This is not a benchmark runner. It does not instantiate `FivePLDigitalTwinEnv`, does not plug into `real_world_scenario_arena`, does not load checkpoints, and does not write eval outputs.

## Files Changed

- `src/eval/rule_based_baselines.py`
- `tests/eval/test_rule_based_baselines.py`
- `docs/runs/20260613_rule_based_baseline_skeleton_report.md`
- `docs/00_PROJECT_DASHBOARD.md`

## Implementation Summary

Added a pure deterministic rule policy module with:

- action constants for the relevant existing 48-action ids;
- `encode_action_id` and `decode_action_id` helpers backed by `DiscreteActionMapper`;
- synthetic `OrderView` and `RuleBasedDecisionContext` dataclasses;
- `RuleBasedDecision` result object with selected order id and action id;
- eight named rule-based baselines:
  - `fifo_shortest_primary_none`
  - `earliest_due_shortest_primary_none`
  - `premium_first_shortest_primary_secondary_none`
  - `low_congestion_under_disruption`
  - `conservative_stock_threshold_reorder`
  - `emergency_stockout_prevention_reorder`
  - `vehicle_scarcity_primary_first_secondary_fallback`
  - `high_holding_no_overstock`

The module is intentionally side-effect free and uses synthetic decision context objects only.

## TDD Evidence

Red phase:

```text
python -m unittest tests.eval.test_rule_based_baselines -v
ModuleNotFoundError: No module named 'src.eval.rule_based_baselines'
FAILED (errors=1)
```

Green phase:

```text
python -m unittest tests.eval.test_rule_based_baselines -v
Ran 11 tests in 0.001s
OK
```

## Tests Covered

- action encode/decode table covers action `24` and action `32`;
- FIFO picks oldest eligible order;
- earliest due date picks earliest due order;
- premium-first picks premium/urgent work before normal work;
- low-congestion rule emits action `32` or action `36` under route pressure;
- conservative stock-threshold reorder emits action `29`;
- emergency stockout prevention emits action `31`;
- vehicle scarcity rule emits action `28` when primary is available and action `24` when only secondary is useful;
- high-holding no-overstock keeps reorder `none`;
- invalid/no-work state returns canonical hold action `0`;
- same synthetic context returns deterministic action.

## Verification Commands

Focused tests:

```text
python -m unittest tests.eval.test_rule_based_baselines -v
```

Result: `PASS`, 11 tests.

Compile check:

```text
python -m py_compile src/eval/rule_based_baselines.py tests/eval/test_rule_based_baselines.py
```

Result: `PASS`, exit code `0`.

## What Was Not Run

- No training.
- No offline eval.
- No long-run gate.
- No benchmark.
- No dependency install.
- No dataset download.
- No checkpoint load.
- No environment rollout.
- No registry mutation.
- No production mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No existing eval-output edit.
- No training config creation.
- No private company-data ingest.
- No firm-facing data request.
- No 1.5M/2M/3M/5M/10M/100M run.

## Protected No-Mutation Proof

Pre-edit and post-code protected hashes matched:

- `models/registry/active_models.json`: size `314`, SHA256 `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: size `14142`, SHA256 `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: size `303413903`, SHA256 `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: size `4084`, SHA256 `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Protected tree profiles remained unchanged:

| Path | Files | Bytes |
| --- | ---: | ---: |
| `models/production` | `8` | `328265276` |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/eval` | `128` | `388267659` |

Process scan:

```text
NO_TRAIN_EVAL_GATE_AWS_PRIVATE_DATA_PROCESS
```

## Independent Review

Reviewer verdict:

```text
RULE_BASED_BASELINE_SKELETON_APPROVED
```

Allowed reviewer verdicts:

- `RULE_BASED_BASELINE_SKELETON_APPROVED`
- `RULE_BASED_BASELINE_SKELETON_NEEDS_FIXES`
- `RULE_BASED_BASELINE_SKELETON_BLOCKED`

## Final Classification

`RULE_BASED_BASELINE_SKELETON_READY`
