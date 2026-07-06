# Level 2 V2.4 Pre-Review Implementation Plan

Status: plan only. Do not create the V2.4 config, start training, run offline eval,
update registry, mutate production, mutate baselines, mutate DB, or mutate
checkpoints before independent Phase 3 review approval.

Delegation boundary: the current thread has not received explicit authorization
for a Phase 3 reviewer. Do not spawn a new independent reviewer until the user
explicitly authorizes it.

## Goal

Run one controlled 250k V-cycle after V2.3 failed the long-run route action-quality
gate. The cycle should isolate the Phase 3 low-congestion no-useful-work reward
patch rather than changing schedule and reward at the same time.

## Proposed V2.4 Identity

- Experiment id: `joint_curriculum_v5_prod_stability_v2_4_250k_20260610`
- Config path, after approval only:
  `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`
- Checkpoint dir, after approval only:
  `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610`
- Eval dir, after approval only:
  `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios`
- Parent checkpoint:
  `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`

## Design Principle

V2.3 passed all scenario thresholds and failed only the long-run route
action-quality gate. The next candidate should keep the V2.3 curriculum schedule
and baseline-anchor policy unchanged unless review rejects this isolation strategy.
That makes the reward change the main experimental variable.

Carry forward from V2.3:

- v5 contract `physical_reality_v5_route_candidate_visibility`
- observation dimension `73`
- action count `48`
- production parent only
- `--skip-registry`
- `--disable-trace-logging`
- torch threads `2/1`
- baseline anchor probability `0.12`
- total timesteps `250000`

Proposed stage schedule after approval:

| Stage | Steps |
| --- | ---: |
| `normal_operation` | 20000 |
| `premium_sla_pressure` | 20000 |
| `normal_operation` | 15000 |
| `lead_time_volatility` | 25000 |
| `high_holding_cost` | 20000 |
| `vehicle_scarcity_capacity_shock` | 15000 |
| `route_disruption_congestion` | 20000 |
| `normal_operation` | 15000 |
| `premium_sla_pressure` | 20000 |
| `lead_time_volatility` | 25000 |
| `high_holding_cost` | 15000 |
| `vehicle_scarcity_capacity_shock` | 10000 |
| `demand_spike_volatility` | 10000 |
| `mixed_stress` | 10000 |
| `normal_operation` | 10000 |

## Task 1: Independent Review Gate

Required files:

- `docs/runs/20260610_level2_phase3_review_request.md`
- `docs/runs/20260610_level2_phase3_review_evidence_manifest.md`
- `docs/runs/20260610_level2_phase3_review_verdict_template.md`

Allowed verdicts:

- `PHASE3_PATCH_APPROVED_FOR_V2_4_CONFIG`
- `PHASE3_PATCH_NEEDS_FIXES_BEFORE_CONFIG`
- `AUTOPILOT_NEEDS_ARCHITECTURE_DECISION`

Do not proceed beyond this task without `PHASE3_PATCH_APPROVED_FOR_V2_4_CONFIG`.

## Task 2: Config TDD After Approval

Add a focused config test before creating the config:

- Assert the V2.4 config path exists.
- Assert the experiment identity and output paths are fresh and V2.4-specific.
- Assert the parent is exactly the production checkpoint.
- Assert no failed V2/V2.1/V2.2/nextgen/routepremium/invalid/probe parent appears.
- Assert v5 contract, observation dimension `73`, action count `48`, and
  timesteps `250000`.
- Assert `skip_registry`, trace-disabled safety, and torch thread safety.
- Assert the stage schedule above and `baseline_anchor_probability == 0.12`.

RED command after approval:

```powershell
python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_stability_v2_4_250k_config_loads_and_is_manual_run_safe -v
```

Expected RED:

- The V2.4 config does not exist yet.

Only after RED, create:

- `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`

Then rerun the focused config test and full config suite.

## Task 3: Pre-Training Verification After Approval

Run before any training:

```powershell
python -m unittest tests.act.test_reward_physics_contract -v
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
python -m unittest tests.learn.test_train_joint_curriculum -v
python -m unittest tests.learn.test_curriculum_config -v
python -m unittest tests.eval.test_long_run_gate -v
python -m py_compile src\act\env_5pl.py src\eval\scenario_metrics.py src\learn\joint_metrics.py tests\act\test_reward_physics_contract.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py
```

Also verify:

- V2.4 checkpoint dir does not exist.
- V2.4 eval dir does not exist.
- No `train_joint_torch` or `evaluate_real_world_scenarios` process is running.
- Registry, production checkpoint, baselines, and DB protected hashes/profiles are
  unchanged before launch.

## Task 4: Train V2.4 250k After Approval

Run only after Tasks 1-3 pass:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; python -m src.learn.train_joint_torch `
  --config configs\training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json `
  --output-dir models\checkpoints\joint_torch_v5_prod_stability_v2_4_250k_20260610 `
  --init-from-joint-checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt `
  --final-ppo-path models\checkpoints\joint_torch_v5_prod_stability_v2_4_250k_20260610\ppo_torch_joint_final_stability_v2_4_250k_20260610.pt `
  --final-dqn-path models\checkpoints\joint_torch_v5_prod_stability_v2_4_250k_20260610\dqn_torch_joint_final_stability_v2_4_250k_20260610.pt `
  --device cpu `
  --torch-num-threads 2 `
  --torch-num-interop-threads 1 `
  --skip-registry `
  --disable-trace-logging
```

Expected:

- Training exits `0`.
- Final checkpoint exists at the V2.4 checkpoint dir.
- Registry, production, baselines, and DB are unchanged.

## Task 5: Evaluate And Gate V2.4

Run only after training exits `0`:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; python -m src.eval.evaluate_real_world_scenarios `
  --checkpoint models\checkpoints\joint_torch_v5_prod_stability_v2_4_250k_20260610\joint_torch_latest.pt `
  --scenario-dir configs\eval_scenarios `
  --output-dir models\eval\joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios `
  --episodes 20 `
  --seed 42 `
  --device cpu `
  --deterministic
```

Then:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios\scenario_summary.json `
  --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json `
  --candidate-label stability_v2_4_250k
```

Decision rule:

- If V2.4 has `8/8 PASS`, all hard blockers zero, and long-run gate exit `0`,
  create a separate 500k plan.
- If V2.4 fails, do not tweak schedule blindly. Perform deeper causal analysis.
- If the failure points to vector env, algorithm redesign, reward objective rewrite,
  evaluator threshold policy, or a new model class, stop with
  `AUTOPILOT_BLOCKED_NEEDS_ARCHITECTURE_DECISION`.

## Non-Goals

- No config creation before independent Phase 3 review approval.
- No training before independent Phase 3 review approval and config TDD.
- No offline eval before V2.4 training exits `0`.
- No 500k before a 250k candidate has `8/8 PASS` and long-run gate exit `0`.
- No registry update.
- No production update.
- No baseline update.
- No DB mutation.
- No failed V2/V2.1/V2.2/V2.3 checkpoint as parent.
