# Level 2 V2.3 Pre-Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the exact V2.3 250k path that can be executed only after independent review approves the Phase 2 reward patch.

**Architecture:** Keep V2.3 parented from the production checkpoint only, preserve the v5 contract and 73/48 observation/action contract, and reuse V2.2 baseline/premium retention while shifting exposure toward the failed timing and vehicle scenarios. This plan deliberately does not create a config or start training before independent review.

**Tech Stack:** Python `unittest`, JSON curriculum configs, `src.learn.train_joint_torch`, `src.eval.evaluate_real_world_scenarios`, `src.eval.check_long_run_gate`, production checkpoint `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`.

---

## Current Gate

Do not execute Tasks 1-5 until independent review returns `PHASE2_PATCH_APPROVED_FOR_V2_3_CONFIG`.

Current review package:

- `docs/runs/20260610_level2_phase2_review_request.md`

Allowed before review:

- Read-only artifact inspection.
- Review-package edits.
- Protected-path hash checks.

Not allowed before review:

- Creating `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`.
- Creating `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610`.
- Starting `src.learn.train_joint_torch`.
- Registering or promoting anything.

## Planned V2.3 Identity

Config path after review:

- `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`

Output directory after review:

- `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610`

Eval directory after review:

- `models/eval/joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios`

Experiment/environment/team id:

- `joint_curriculum_v5_prod_stability_v2_3_250k_20260610`

Parent checkpoint:

- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`

Forbidden parent or reference fragments:

- `joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609`
- `joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609`
- `joint_curriculum_v5_prod_stability_v2_250k_20260609`
- `joint_torch_v5_prod_stability_v2_250k_20260609`
- `joint_curriculum_v5_prod_stability_v2_1_250k_20260609`
- `joint_torch_v5_prod_stability_v2_1_250k_20260609`
- `joint_curriculum_v5_prod_stability_v2_2_250k_20260610`
- `joint_torch_v5_prod_stability_v2_2_250k_20260610`
- `joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k`
- `joint_torch_v5_clean_cold_start_1m_after_rewardfix/`
- `joint_torch_v5_clean_cold_start_1m/`
- `joint_curriculum_v4`
- `physical_reality_v4`
- `physical_reality_v3`
- `dqn1024`
- `dqn512`
- `700k`

## Planned V2.3 Curriculum Direction

Use the same model/training hyperparameters as V2.2. The intended curriculum differs only in the stage schedule and objective labels after review.

Rationale:

- V2.2 restored baseline and premium, so keep nonzero baseline anchoring and regular `normal_v4` / `premium_sla` retention.
- V2.2 failed `high_holding_cost`, `lead_time_volatility`, and `vehicle_scarcity_capacity_shock`, so increase those exposures after the Phase 2 reward patch.
- Route scenario service passed but route action-quality failed, so keep route exposure stable instead of increasing it blindly.

Stage schedule after review:

```json
[
  {"name": "normal_v4", "steps": 20000},
  {"name": "premium_sla", "steps": 20000},
  {"name": "normal_v4", "steps": 15000},
  {"name": "lead_time_delay", "steps": 25000},
  {"name": "high_holding_cost", "steps": 20000},
  {"name": "vehicle_scarcity", "steps": 15000},
  {"name": "route_disruption", "steps": 20000},
  {"name": "normal_v4", "steps": 15000},
  {"name": "premium_sla", "steps": 20000},
  {"name": "lead_time_delay", "steps": 25000},
  {"name": "high_holding_cost", "steps": 15000},
  {"name": "vehicle_scarcity", "steps": 10000},
  {"name": "demand_spike", "steps": 10000},
  {"name": "mixed_stress", "steps": 10000},
  {"name": "normal_v4", "steps": 10000}
]
```

Schedule totals:

- `normal_v4`: 60k
- `premium_sla`: 40k
- `lead_time_delay`: 50k
- `high_holding_cost`: 35k
- `vehicle_scarcity`: 25k
- `route_disruption`: 20k
- `demand_spike`: 10k
- `mixed_stress`: 10k
- Total: 250k

Baseline anchor after review:

- `baseline_anchor_probability`: `0.12`

Reason: V2.2 passed baseline/premium with `0.10`, but V2.3 reduces explicit normal-stage steps while increasing failed-scenario pressure. The small anchor increase is a retention hedge, not the main fix.

## Task 1: Independent Review Gate

**Files:**

- Read: `docs/runs/20260610_level2_phase2_review_request.md`
- Read: `docs/runs/20260610_level2_postmortem_v2_cycle_design.md`

- [ ] **Step 1: Confirm review verdict**

Accept only one of these review verdicts:

```text
PHASE2_PATCH_APPROVED_FOR_V2_3_CONFIG
PHASE2_PATCH_NEEDS_FIXES_BEFORE_CONFIG
AUTOPILOT_NEEDS_ARCHITECTURE_DECISION
```

Expected:

- Continue only on `PHASE2_PATCH_APPROVED_FOR_V2_3_CONFIG`.
- Stop and patch/test again on `PHASE2_PATCH_NEEDS_FIXES_BEFORE_CONFIG`.
- Stop and report architecture block on `AUTOPILOT_NEEDS_ARCHITECTURE_DECISION`.

## Task 2: Add Config Test With TDD

**Files:**

- Modify after review: `tests/learn/test_curriculum_config.py`
- Future create after review: `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`

- [ ] **Step 1: Write the failing config test**

Add this test after `test_prod_stability_v2_2_250k_config_loads_and_is_manual_run_safe`:

```python
    def test_prod_stability_v2_3_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        v2_2 = load_config(Path("configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json"))
        config_path = Path("configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_3_250k_20260610"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        future_output_dir = Path("models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610")
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, v2_2["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k",
        )
        self.assertEqual(fine_tune["source_global_step"], 200_000)
        self.assertEqual(fine_tune["source_parent_global_step"], 1_000_000)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_continuous_action_dim"], 5)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], str(future_output_dir).replace("\\", "/"))
        self.assertFalse(future_output_dir.exists())

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            stage_schedule,
            [
                ("normal_v4", 20_000),
                ("premium_sla", 20_000),
                ("normal_v4", 15_000),
                ("lead_time_delay", 25_000),
                ("high_holding_cost", 20_000),
                ("vehicle_scarcity", 15_000),
                ("route_disruption", 20_000),
                ("normal_v4", 15_000),
                ("premium_sla", 20_000),
                ("lead_time_delay", 25_000),
                ("high_holding_cost", 15_000),
                ("vehicle_scarcity", 10_000),
                ("demand_spike", 10_000),
                ("mixed_stress", 10_000),
                ("normal_v4", 10_000),
            ],
        )
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(stage_schedule[-1][0], "normal_v4")
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)
```

- [ ] **Step 2: Run RED verification**

Run:

```powershell
python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_stability_v2_3_250k_config_loads_and_is_manual_run_safe -v
```

Expected:

- FAIL because `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json` does not exist.

## Task 3: Create V2.3 Config After Review

**Files:**

- Create after review: `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`

- [ ] **Step 1: Copy V2.2 config structure**

Start from:

```powershell
Copy-Item configs\training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json configs\training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json
```

- [ ] **Step 2: Replace identity and safety paths**

Set these JSON values exactly:

```json
{
  "shared_global_parameters": {
    "experiment_name": "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
    "environment_id": "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
    "team_id": "joint_curriculum_v5_prod_stability_v2_3_250k_20260610"
  },
  "manual_run_safety": {
    "requires_skip_registry": true,
    "requires_disable_trace_logging": true,
    "torch_num_threads": 2,
    "torch_num_interop_threads": 1,
    "output_dir": "models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610",
    "final_ppo_path": "models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/ppo_torch_joint_final_stability_v2_3_250k_20260610.pt",
    "final_dqn_path": "models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/dqn_torch_joint_final_stability_v2_3_250k_20260610.pt"
  }
}
```

- [ ] **Step 3: Preserve production parent**

Confirm the `fine_tune_initialization.source_checkpoint` remains exactly:

```text
models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt
```

- [ ] **Step 4: Replace curriculum schedule**

Set `curriculum.baseline_anchor_probability` to `0.12`.

Set `curriculum.stage_schedule` to the exact schedule in the "Planned V2.3 Curriculum Direction" section.

- [ ] **Step 5: Run GREEN verification**

Run:

```powershell
python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_stability_v2_3_250k_config_loads_and_is_manual_run_safe -v
python -m unittest tests.learn.test_curriculum_config -v
python -m py_compile tests\learn\test_curriculum_config.py
```

Expected:

- Focused V2.3 config test passes.
- Full curriculum config suite passes.
- `py_compile` exits `0`.

## Task 4: Train V2.3 250k Only After Config Tests Pass

**Files:**

- Read: `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`
- Write after review: `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610`

- [ ] **Step 1: Confirm output directory is fresh**

Run:

```powershell
Test-Path models\checkpoints\joint_torch_v5_prod_stability_v2_3_250k_20260610
```

Expected:

- `False`

- [ ] **Step 2: Launch 250k training**

Run:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; python -m src.learn.train_joint_torch `
  --config configs\training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json `
  --output-dir models\checkpoints\joint_torch_v5_prod_stability_v2_3_250k_20260610 `
  --init-from-joint-checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt `
  --final-ppo-path models\checkpoints\joint_torch_v5_prod_stability_v2_3_250k_20260610\ppo_torch_joint_final_stability_v2_3_250k_20260610.pt `
  --final-dqn-path models\checkpoints\joint_torch_v5_prod_stability_v2_3_250k_20260610\dqn_torch_joint_final_stability_v2_3_250k_20260610.pt `
  --device cpu `
  --torch-num-threads 2 `
  --torch-num-interop-threads 1 `
  --skip-registry `
  --disable-trace-logging
```

Expected:

- Training exits `0`.
- Final checkpoint exists at `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/joint_torch_latest.pt`.
- Registry files are unchanged.

## Task 5: Evaluate And Gate V2.3

**Files:**

- Read after review: `models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/joint_torch_latest.pt`
- Write after review: `models/eval/joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios`

- [ ] **Step 1: Confirm eval directory is fresh**

Run:

```powershell
Test-Path models\eval\joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios
```

Expected:

- `False`

- [ ] **Step 2: Run offline scenarios**

Run:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; python -m src.eval.evaluate_real_world_scenarios `
  --checkpoint models\checkpoints\joint_torch_v5_prod_stability_v2_3_250k_20260610\joint_torch_latest.pt `
  --scenario-dir configs\eval_scenarios `
  --output-dir models\eval\joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios `
  --episodes 20 `
  --seed 42 `
  --device cpu `
  --deterministic
```

Expected:

- Eval exits `0`.
- `scenario_summary.json` exists in the V2.3 eval directory.

- [ ] **Step 3: Run long-run gate**

Run:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_3_250k_20260610_offline_scenarios\scenario_summary.json `
  --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json `
  --candidate-label stability_v2_3_250k
```

Expected:

- Exit `0` and `decision` `PASS` before any 500k plan can begin.
- Exit nonzero or `decision` `FAIL` means stop and perform deeper causal analysis.

## Task 6: Report And Horizon Decision

**Files:**

- Modify after review and run: `docs/runs/20260610_level2_postmortem_v2_cycle_design.md`

- [ ] **Step 1: Update report**

Record:

- Files changed.
- Tests.
- Training command and output dir.
- Eval command and output dir.
- Gate decision.
- Scenario comparison versus production.
- Protected-path hashes.
- Clean 250k status.

- [ ] **Step 2: Decide next horizon**

Use this exact decision rule:

- If V2.3 250k has 8/8 PASS and long-run gate exit `0`, proceed to a separate 500k plan.
- If V2.3 250k fails, do not tweak schedule blindly; perform deeper causal analysis.
- If failure suggests vector env, algorithm redesign, reward objective rewrite, evaluator threshold policy, or new model class, report `AUTOPILOT_BLOCKED_NEEDS_ARCHITECTURE_DECISION`.

## Final Verification Before Any Training

Run before Task 4:

```powershell
python -m unittest tests.act.test_reward_physics_contract -v
python -m unittest tests.learn.test_train_joint_curriculum -v
python -m unittest tests.learn.test_curriculum_config -v
python -m unittest tests.eval.test_long_run_gate -v
python -m py_compile src\act\env_5pl.py src\eval\scenario_metrics.py src\learn\joint_metrics.py tests\act\test_reward_physics_contract.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py
```

Expected:

- All listed tests pass.
- Compile exits `0`.

## Non-Goals

- No training before independent review.
- No config creation before independent review.
- No 500k before a 250k candidate has 8/8 PASS and long-run gate exit `0`.
- No 1M before a 500k candidate has 8/8 PASS and long-run gate exit `0`.
- No 3M, 5M, 10M, or 100M in this goal.
- No registry update.
- No production update.
- No baseline update.
- No DB mutation.
- No failed V2/V2.1/V2.2 checkpoint as parent.
