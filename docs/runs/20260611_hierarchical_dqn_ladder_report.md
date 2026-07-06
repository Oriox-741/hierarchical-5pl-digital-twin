# Hierarchical DQN Ladder Report

Date: 2026-06-11

## Result

Final classification: `CLEAN_1M_STABILITY_CANDIDATE_READY_FOR_REVIEW`

Candidate review package only. No registry update, production promotion, baseline mutation, DB mutation, source production checkpoint mutation, or model activation was performed.

The hierarchical ladder reached 1M and passed the long-run gate:

- 250k: PASS, gate exit code `0`.
- 500k: PASS, gate exit code `0`.
- 1M: PASS, gate exit code `0`.

## Architecture Used

- DQN architecture: `hierarchical_v1`.
- Initialization: `flat_teacher_distillation_v1`.
- External contract preserved: `physical_reality_v5_route_candidate_visibility`.
- Observation dimension preserved: `73`.
- External discrete action count preserved: `48`.

Flat production was used only as the read-only teacher/init source for the first 250k rung. It was never used as a hierarchical exact-resume parent.

## Files Changed

- `configs/training_joint_curriculum_v5_prod_hierarchical_v1_500k_20260611.json`
- `configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json`
- `tests/learn/test_curriculum_config.py`
- `docs/runs/20260611_hierarchical_dqn_ladder_report.md`
- `docs/00_PROJECT_DASHBOARD.md`

Existing readiness config used:

- `configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json`

## Config Review

500k config reviewer verdict:

```text
PATCH_APPROVED_FOR_NEXT_GATE
```

1M config reviewer verdict:

```text
PATCH_APPROVED_FOR_NEXT_GATE
```

Both reviewers confirmed exact-resume lineage, `hierarchical_v1`, v5/73/48 contract preservation, replay/RNG requirements, absence of `--allow-empty-replay-resume`, no failed parent checkpoints, and protected output paths.

## Tests And Preflight

Before 250k:

- `python -m unittest tests.learn.test_hierarchical_dqn_initialization tests.think.test_hierarchical_dqn_policy tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator tests.learn.test_train_joint_curriculum tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_250k_config_loads_and_is_manual_run_safe -v`
- Result: PASS, 110 tests.

- `python -m py_compile src\learn\hierarchical_dqn_initialization.py src\learn\train_joint_torch.py src\think\joint_policies.py src\orchestration\torch_joint_runtime.py src\eval\real_world_scenario_arena.py src\eval\check_long_run_gate.py tests\learn\test_hierarchical_dqn_initialization.py tests\think\test_hierarchical_dqn_policy.py tests\orchestration\test_torch_joint_runtime.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py`
- Result: PASS.

500k config TDD:

- RED: `python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_500k_config_loads_and_requires_exact_resume -v`
- Result before config: FAIL because config did not exist.
- GREEN: 250k and 500k config tests passed, 2 tests.
- Relevant preflight after 500k config: PASS, 71 tests.

1M config TDD:

- RED: `python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_1m_config_loads_and_requires_exact_resume -v`
- Result before config: FAIL because config did not exist.
- GREEN: 250k, 500k, and 1M config tests passed, 3 tests.
- Relevant preflight after 1M config: PASS, 72 tests.

Final verification after report/dashboard update:

- `python -m unittest tests.learn.test_hierarchical_dqn_initialization tests.think.test_hierarchical_dqn_policy tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator tests.learn.test_train_joint_curriculum tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_250k_config_loads_and_is_manual_run_safe tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_500k_config_loads_and_requires_exact_resume tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_1m_config_loads_and_requires_exact_resume -v`
- Result: PASS, 112 tests.
- `python -m py_compile src\learn\hierarchical_dqn_initialization.py src\learn\train_joint_torch.py src\think\joint_policies.py src\orchestration\torch_joint_runtime.py src\eval\real_world_scenario_arena.py src\eval\check_long_run_gate.py tests\learn\test_hierarchical_dqn_initialization.py tests\think\test_hierarchical_dqn_policy.py tests\orchestration\test_torch_joint_runtime.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py`
- Result: PASS.

## Run Artifacts

250k:

- Config: `configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json`
- Run dir: `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_250k_20260611`
- Eval dir: `models/eval/joint_torch_v5_prod_hierarchical_v1_250k_20260611_offline_scenarios`
- Gate result: `docs/runs/20260611_hierarchical_dqn_250k_gate_result.json`
- Train log: `docs/runs/20260611_hierarchical_dqn_250k_train.log`
- Eval log: `docs/runs/20260611_hierarchical_dqn_250k_eval.log`

500k:

- Config: `configs/training_joint_curriculum_v5_prod_hierarchical_v1_500k_20260611.json`
- Run dir: `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_500k_20260611`
- Eval dir: `models/eval/joint_torch_v5_prod_hierarchical_v1_500k_20260611_offline_scenarios`
- Gate result: `docs/runs/20260611_hierarchical_dqn_500k_gate_result.json`
- Train log: `docs/runs/20260611_hierarchical_dqn_500k_train.log`
- Eval log: `docs/runs/20260611_hierarchical_dqn_500k_eval.log`

1M:

- Config: `configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json`
- Run dir: `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Eval dir: `models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`
- Gate result: `docs/runs/20260611_hierarchical_dqn_1m_gate_result.json`
- Train log: `docs/runs/20260611_hierarchical_dqn_1m_train.log`
- Eval log: `docs/runs/20260611_hierarchical_dqn_1m_eval.log`

## Training Commands

250k used `--init-from-joint-checkpoint` from production as a read-only hierarchical teacher/init source:

```powershell
python -m src.learn.train_joint_torch --config configs\training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json --output-dir models\checkpoints\joint_torch_v5_prod_hierarchical_v1_250k_20260611 --init-from-joint-checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt --seed 42 --device cpu --final-ppo-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_250k_20260611\ppo_torch_joint_final_hierarchical_v1_250k_20260611.pt --final-dqn-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_250k_20260611\dqn_torch_joint_final_hierarchical_v1_250k_20260611.pt --skip-registry --disable-trace-logging --torch-num-threads 2 --torch-num-interop-threads 1
```

500k used normal exact resume from 250k:

```powershell
python -m src.learn.train_joint_torch --config configs\training_joint_curriculum_v5_prod_hierarchical_v1_500k_20260611.json --output-dir models\checkpoints\joint_torch_v5_prod_hierarchical_v1_500k_20260611 --resume models\checkpoints\joint_torch_v5_prod_hierarchical_v1_250k_20260611\joint_torch_latest.pt --seed 42 --device cpu --final-ppo-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_500k_20260611\ppo_torch_joint_final_hierarchical_v1_500k_20260611.pt --final-dqn-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_500k_20260611\dqn_torch_joint_final_hierarchical_v1_500k_20260611.pt --skip-registry --disable-trace-logging --torch-num-threads 2 --torch-num-interop-threads 1
```

1M used normal exact resume from 500k:

```powershell
python -m src.learn.train_joint_torch --config configs\training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json --output-dir models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611 --resume models\checkpoints\joint_torch_v5_prod_hierarchical_v1_500k_20260611\joint_torch_latest.pt --seed 42 --device cpu --final-ppo-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt --final-dqn-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt --skip-registry --disable-trace-logging --torch-num-threads 2 --torch-num-interop-threads 1
```

`--allow-empty-replay-resume` was never used.

## Exact-Resume Proof

250k final checkpoint:

- `global_step`: `250000`
- `dqn_architecture`: `hierarchical_v1`
- `external_discrete_action_count`: `48`
- `observation_dim`: `73`
- `hierarchical_init_method`: `flat_teacher_distillation_v1`
- replay state present: yes
- RNG state present: yes
- optimizer state present: yes
- `resume_state.exact_resume_capable`: yes
- replay size/position/total added: `250000` / `250000` / `250000`

500k final checkpoint:

- `global_step`: `500000`
- `dqn_architecture`: `hierarchical_v1`
- `external_discrete_action_count`: `48`
- `observation_dim`: `73`
- replay state present: yes
- RNG state present: yes
- optimizer state present: yes
- `resume_state.exact_resume_capable`: yes
- replay size/position/total added: `500000` / `0` / `500000`
- trainer startup log: `resume=exact replay_buffer=restored replay_size=250000 rng_state=restored`

1M final checkpoint:

- `global_step`: `1000000`
- `dqn_architecture`: `hierarchical_v1`
- `external_discrete_action_count`: `48`
- `observation_dim`: `73`
- replay state present: yes
- RNG state present: yes
- optimizer state present: yes
- `resume_state.exact_resume_capable`: yes
- replay size/position/total added: `500000` / `0` / `1000000`
- trainer startup log: `resume=exact replay_buffer=restored replay_size=500000 rng_state=restored`

## Distillation Telemetry

250k warm-start metadata:

- `hierarchical_distillation_steps`: `512`
- `hierarchical_distillation_observation_sample_count`: `4096`
- `hierarchical_distillation_batch_size`: `128`
- `hierarchical_distillation_initial_loss`: `0.997825`
- `hierarchical_distillation_loss`: `0.997805`
- `hierarchical_distillation_raw_loss`: `454.483337`
- `hierarchical_distillation_q_loss`: `441.715912`
- `hierarchical_distillation_action_ce`: `12.767425`
- `hierarchical_distillation_action_match_rate`: `0.000977`
- `hierarchical_distillation_teacher_entropy`: `0.117629`

The distillation loss was near the configured bound and teacher action-match was low, but the downstream 250k, 500k, and 1M scenario gates all passed.

## Gate Decisions

250k:

- Eval exit code: `0`
- Gate decision: PASS
- Gate exit code: `0`
- Fatal failures: none
- Warnings: `20`

500k:

- Eval exit code: `0`
- Gate decision: PASS
- Gate exit code: `0`
- Fatal failures: none
- Warnings: `24`

1M:

- Eval exit code: `0`
- Gate decision: PASS
- Gate exit code: `0`
- Fatal failures: none
- Warnings: `18`

## Scenario Table

Values are `service / true_lateness_pressure / dispatch_rate / dispatch_success / no-current+no-unassigned`.

| Scenario | Production | H250k | H500k | H1M | H1M verdict |
|---|---:|---:|---:|---:|---|
| baseline_normal | 0.938 / 0.000 / 0.487 / 1.000 / 0 | 0.952 / 0.002 / 0.469 / 1.000 / 1 | 0.945 / 0.011 / 0.455 / 1.000 / 1 | 0.990 / 0.000 / 0.372 / 1.000 / 0 | PASS |
| demand_spike_volatility | 0.843 / 0.177 / 0.728 / 1.000 / 0 | 0.955 / 0.001 / 0.986 / 1.000 / 1 | 0.934 / 0.001 / 0.858 / 1.000 / 0 | 0.859 / 0.027 / 0.793 / 1.000 / 0 | PASS |
| high_holding_cost | 0.943 / 0.000 / 0.387 / 1.000 / 0 | 0.923 / 0.001 / 0.439 / 1.000 / 0 | 0.980 / 0.002 / 0.418 / 0.999 / 3 | 0.966 / 0.000 / 0.449 / 0.999 / 2 | PASS |
| lead_time_volatility | 0.936 / 0.000 / 0.455 / 1.000 / 0 | 0.946 / 0.004 / 0.450 / 1.000 / 1 | 0.982 / 0.027 / 0.355 / 0.999 / 1 | 1.000 / 0.000 / 0.360 / 0.999 / 1 | PASS |
| mixed_stress | 0.951 / 0.000 / 0.911 / 0.996 / 6 | 0.940 / 0.003 / 0.980 / 1.000 / 0 | 0.794 / 0.010 / 0.789 / 1.000 / 0 | 0.942 / 0.000 / 0.987 / 0.999 / 8 | PASS |
| premium_sla_pressure | 0.980 / 0.000 / 0.427 / 0.769 / 178 | 0.943 / 0.002 / 0.381 / 0.999 / 3 | 0.993 / 0.000 / 0.357 / 0.999 / 1 | 1.000 / 0.000 / 0.357 / 0.999 / 1 | PASS |
| route_disruption_congestion | 0.929 / 0.000 / 0.575 / 0.822 / 171 | 0.948 / 0.003 / 0.489 / 1.000 / 1 | 0.892 / 0.018 / 0.500 / 1.000 / 0 | 0.901 / 0.008 / 0.499 / 0.999 / 2 | PASS |
| vehicle_scarcity_capacity_shock | 0.914 / 0.000 / 0.731 / 1.000 / 0 | 0.945 / 0.002 / 0.812 / 1.000 / 2 | 0.946 / 0.004 / 0.811 / 1.000 / 0 | 0.949 / 0.001 / 0.806 / 1.000 / 0 | PASS |

## 1M Review Notes

The candidate passed all hard gates. Residual warnings worth human review before any promotion decision:

- `mixed_stress` service remains below production: `0.942` vs `0.951`.
- `route_disruption_congestion` service remains below production: `0.901` vs `0.929`, with lateness `0.008` vs production `0.000`.
- Demand-spike service regressed from the 500k gate: `0.934` to `0.859`, but remains above production `0.843` and passes scenario thresholds.
- Several scenarios show increased top-action concentration and mixed-success route-failure steps.

These are warnings, not fatal gate failures.

## Protected No-Mutation Proof

Protected hashes after all runs, evals, and gates:

- `models/registry/active_models.json`: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl`: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles:

- `models/production`: 4 files, 15040339 bytes.
- `models/baselines`: 2692 files, 7392576274 bytes.
- `db`: 7 files, 28330 bytes.

No active `train_joint_torch` or `evaluate_real_world_scenarios` process remained after final verification.

## Stop Condition

The ladder stops here because the 1M candidate passed. No 3M, 5M, 10M, or 100M run was created or started.

Next action is human candidate review only. Do not register or promote without explicit approval.
