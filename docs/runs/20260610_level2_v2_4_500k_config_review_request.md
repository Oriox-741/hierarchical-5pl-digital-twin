# Level 2 V2.4 500k Config Review Request

Date: 2026-06-10

## Review Scope

Read-only independent review of the V2.4 500k ladder config before training.

Inspect:

- `configs/training_joint_curriculum_v5_prod_stability_v2_4_500k_20260610.json`
- `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`
- `tests/learn/test_curriculum_config.py`
- `docs/runs/20260610_level2_v2_4_semantics_gate_result.json`
- `docs/runs/20260610_level2_v2_4_semantics_patch_review_verdict.md`

## Required Checks

1. The config is a true 500k ladder continuation target, not a blind restart.
2. The intended run command should use `--resume models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610/joint_torch_latest.pt` so `global_step` advances from 250k to 500k.
3. The config does not use invalid parent lineages such as failed V2/V2.1/V2.2/V2.3, nextgen 1M, failed routepremium, aborted after_rewardfix, invalid 700k, V3, V4, probes, production mutation, baselines, registry, DB, or checkpoints outside the approved V2.4 250k parent.
4. The output directory and final artifact paths are fresh and isolated under `models/checkpoints/joint_torch_v5_prod_stability_v2_4_500k_20260610`.
5. Manual run safety requires `--skip-registry`, `--disable-trace-logging`, torch threads 2/1.
6. The stage schedule sums to 500k and preserves the approved V2.4 250k schedule twice.
7. It is safe to begin the 500k training run after this config.

## Verification Already Run Locally

```powershell
python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_stability_v2_4_500k_config_loads_and_is_manual_run_safe -v
python -m json.tool configs\training_joint_curriculum_v5_prod_stability_v2_4_500k_20260610.json > $null
```

Both commands exited 0.

## Constraints

Do not train.
Do not run offline eval.
Do not update registry.
Do not mutate production.
Do not mutate baselines.
Do not mutate DB.
Do not mutate checkpoints.

Return exactly one verdict line:

```text
PATCH_APPROVED_FOR_NEXT_GATE
PATCH_NEEDS_FIXES_BEFORE_NEXT_GATE
AUTOPILOT_NEEDS_ARCHITECTURE_DECISION
```
