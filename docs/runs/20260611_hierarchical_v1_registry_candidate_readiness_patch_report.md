# Hierarchical V1 Registry Candidate Readiness Patch Report

Date: 2026-06-11

## Result

Patch status: implemented, locally verified, and independently approved for hierarchical candidate dry-run.

Registry status: dry-run only. No real candidate rows were appended and no active registry mapping was changed.

No training, offline eval, registry mutation, activation, production promotion, production copy, baseline mutation, DB mutation, or checkpoint mutation was performed.

## Root Cause

`src.learn.register_torch_joint_candidate` was built around the historical 200k balanced-retention production handoff. It required existing candidate ids, validated against hardcoded `joint_torch_v5_balanced_retention_ft_200k_20260608` / `training_step=200000` assumptions, and had no artifact-path candidate dry-run path for the hierarchical 1M checkpoint.

That blocked safe candidate-row readiness for:

- `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `training_step=1000000`
- exact-resume-capable replay/RNG checkpoint metadata

## Patch Summary

Changed:

- `src/learn/register_torch_joint_candidate.py`
  - Keeps existing candidate-id activation flow.
  - Adds artifact-path candidate registration planning mode:
    - `--joint-checkpoint`
    - `--ppo-path`
    - `--dqn-path`
    - `--eval-output-dir`
    - `--gate-result`
    - `--dqn-architecture`
    - `--hierarchical-init-method`
  - Adds explicit `--register-candidate` for real candidate row appends.
  - Keeps dry-run as the default when neither `--activate` nor `--register-candidate` is supplied.
  - Validates hierarchical checkpoint metadata, eval evidence, and gate evidence before planning candidate rows.
  - Prints planned PPO/DQN candidate rows and planned metadata in dry-run mode.
  - Does not write `active_models.json` in candidate-row mode.

- `tests/learn/test_register_torch_joint_candidate.py`
  - Preserves old 200k activation/dry-run tests.
  - Adds hierarchical candidate dry-run tests for:
    - metadata validation,
    - no registry writes,
    - missing/wrong `dqn_architecture`,
    - wrong `hierarchical_init_method`,
    - failed eval verdict,
    - nonzero hard blocker,
    - missing `exact_resume_capable`,
    - planned metadata including hierarchical fields.

- `tests/orchestration/test_policy_service_torch_joint.py`
  - Adds active-style hierarchical metadata coverage proving `PolicyService` still resolves through `load_torch_joint_policy` and does not call SB3 loaders.

## Validated Hierarchical Metadata

The artifact-path dry-run validates:

- joint checkpoint exists,
- split PPO/DQN artifacts exist,
- eval dir exists,
- gate JSON exists,
- `checkpoint_version=torch_joint_policy_v1`,
- `artifact_kind=joint_final`,
- `global_step=1000000`,
- `dqn_architecture=hierarchical_v1`,
- `hierarchical_init_method=flat_teacher_distillation_v1`,
- `resume_state.exact_resume_capable=true`,
- replay state present,
- RNG state present,
- contract `physical_reality_v5_route_candidate_visibility`,
- observation dim `73`,
- discrete action count `48`,
- `external_discrete_action_count=48`,
- `internal_heads` present,
- `8` scenarios present,
- all scenario verdicts PASS,
- expected episode rows present,
- global hard blockers zero,
- long-run gate verdict PASS and no fatal failures.

## Candidate Metadata Planned

The dry-run plans both PPO and DQN candidate rows with:

- `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- `framework=torch_joint`
- `runtime_loader=joint_torch`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `joint_checkpoint=models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- `contract=physical_reality_v5_route_candidate_visibility`
- `obs_dim=73`
- `action_count=48`
- `external_discrete_action_count=48`
- `training_step=1000000`
- `exact_resume_capable=true`
- `eval_output_dir=models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`
- `eval_verdict=PASS`
- `hard_blocker_status=zero`
- `long_run_gate_verdict=PASS`
- `production_ready=false`
- `baseline_update=false`
- `registry_scope=candidate_only`

## TDD Evidence

RED run:

- `python -m unittest tests.learn.test_register_torch_joint_candidate -v`
- Result before implementation: FAIL, 6 errors.
- Expected failure: `argparse` still required `--ppo-candidate-id` / `--dqn-candidate-id` and did not accept artifact-path candidate registration flags.

GREEN focused runs:

- `python -m unittest tests.learn.test_register_torch_joint_candidate -v`
  - Result: PASS, `26` tests.
- `python -m unittest tests.orchestration.test_policy_service_torch_joint -v`
  - Result: PASS, `12` tests.

## Verification

Requested verification:

- `python -m unittest tests.learn.test_register_torch_joint_candidate -v`
  - Result: PASS, `26` tests.
- `python -m unittest tests.learn.test_model_registry -v`
  - Result: PASS, `7` tests.
- `python -m unittest tests.orchestration.test_policy_service_torch_joint -v`
  - Result: PASS, `12` tests.
- `python -m unittest tests.orchestration.test_torch_joint_runtime -v`
  - Result: PASS, `9` tests.
- `python -m py_compile src\learn\register_torch_joint_candidate.py tests\learn\test_register_torch_joint_candidate.py tests\orchestration\test_policy_service_torch_joint.py`
  - Result: PASS.

## Real Hierarchical Candidate Dry-Run

Command run against the real registry in dry-run mode only:

```powershell
python -m src.learn.register_torch_joint_candidate --logical-model-id joint_torch_v5_prod_hierarchical_v1_1m_20260611 --registry-dir models\registry --joint-checkpoint models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt --ppo-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt --dqn-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt --eval-output-dir models\eval\joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios --gate-result docs\runs\20260611_hierarchical_dqn_1m_gate_result.json --dqn-architecture hierarchical_v1 --hierarchical-init-method flat_teacher_distillation_v1 --dry-run
```

Result:

```text
DRY_RUN_OK
```

The dry-run printed planned PPO/DQN candidate rows and planned metadata, including `dqn_architecture`, `hierarchical_init_method`, `training_step=1000000`, `exact_resume_capable=true`, `eval_verdict=PASS`, `hard_blocker_status=zero`, `long_run_gate_verdict=PASS`, and `registry_scope=candidate_only`.

## Future Real Candidate Registration Command

Do not run without explicit human approval:

```powershell
python -m src.learn.register_torch_joint_candidate --logical-model-id joint_torch_v5_prod_hierarchical_v1_1m_20260611 --registry-dir models\registry --joint-checkpoint models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt --ppo-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt --dqn-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt --eval-output-dir models\eval\joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios --gate-result docs\runs\20260611_hierarchical_dqn_1m_gate_result.json --dqn-architecture hierarchical_v1 --hierarchical-init-method flat_teacher_distillation_v1 --register-candidate
```

This command would append candidate rows to `models/registry/models.jsonl`. It would not activate the model.

## Future Activation Dry-Run Shape

After candidate rows are actually registered and reviewed, activation review should still start with dry-run only:

```powershell
python -m src.learn.register_torch_joint_candidate --logical-model-id joint_torch_v5_prod_hierarchical_v1_1m_20260611 --ppo-candidate-id <hierarchical-ppo-candidate-id> --dqn-candidate-id <hierarchical-dqn-candidate-id> --registry-dir models\registry --dry-run
```

Do not run `--activate` without a separate explicit approval.

## Protected No-Mutation Proof

Protected hashes before and after the real dry-run stayed unchanged:

- `models/registry/active_models.json`: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl`: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles after verification:

- `models/production`: `4` files, `15040339` bytes.
- `models/baselines`: `2692` files, `7392576274` bytes.
- `db`: `7` files, `28330` bytes.

Process check:

- `NO_TRAIN_EVAL_GATE_PROCESS`

## Independent Review

Reviewer verdict:

```text
REGISTRY_PATCH_APPROVED_FOR_HIERARCHICAL_CANDIDATE_DRY_RUN
```

## Final Classification

`HIERARCHICAL_V1_REGISTRY_CANDIDATE_DRY_RUN_READY`
