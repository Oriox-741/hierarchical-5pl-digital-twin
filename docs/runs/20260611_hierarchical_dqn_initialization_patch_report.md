# Hierarchical DQN Initialization Patch Report

Date: 2026-06-11

## Result

Patch status: implemented, locally verified, and independently approved.

Config status: readiness config created after independent approval. No training run was started.

No training, offline eval, registry update, production mutation, baseline mutation, DB mutation, source production checkpoint mutation, existing checkpoint mutation, 250k, 500k, or 1M run was performed.

## Initialization Decision

Chosen method: `flat_teacher_distillation_v1`.

The flat production checkpoint remains incompatible with `hierarchical_v1` exact resume. The patch adds a separate warm-start path:

- load the flat checkpoint as a frozen read-only teacher,
- load compatible PPO state into the hierarchical student,
- copy compatible flat DQN trunk tensors into the hierarchical DQN trunk,
- run bounded supervised distillation from flat external 48-Q teacher values to hierarchical external 48-Q student values on a finite sampled observation bank,
- start future RL with fresh optimizer, empty replay, and fresh RNG,
- store explicit initialization metadata.

## Files Changed

- `src/learn/hierarchical_dqn_initialization.py`
  - Added flat teacher checkpoint loader.
  - Added hierarchical initialization config parser.
  - Added deterministic sampled observation bank helper.
  - Added finite bounded flat-to-hierarchical distillation loss.
  - Added supervised distillation warm-start loop.
  - Added PPO and compatible DQN trunk warm-start helper.

- `src/learn/train_joint_torch.py`
  - Added `hierarchical_initialization` config parsing.
  - Added explicit `hierarchical_v1` init branch under `--init-from-joint-checkpoint`.
  - Kept `--resume` unchanged.
  - Kept flat `--init-from-joint-checkpoint` unchanged.
  - Added initialization metadata keys to checkpoint payload export.

- `tests/learn/test_hierarchical_dqn_initialization.py`
  - Added TDD coverage for exact-resume rejection, read-only flat teacher loading, sampled observation bank validity, bounded loss, tiny supervised action match, warm-start metadata, and checkpoint payload metadata.

- `tests/learn/test_curriculum_config.py`
  - Added config-readiness test for the first production-parent hierarchical 250k config.

- `configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json`
  - Added readiness-only 250k config using production as the only read-only teacher/init source.
  - Added `dqn_architecture=hierarchical_v1`.
  - Added explicit `hierarchical_initialization.method=flat_teacher_distillation_v1`.
  - Marked `manual_run_safety.training_authorized=false`.

- `docs/plans/20260611_hierarchical_dqn_initialization_plan.md`
  - Added implementation plan.

- `docs/runs/20260611_hierarchical_dqn_initialization_decision.md`
  - Added initialization decision.

## Safety Semantics

Flat production checkpoint use is explicitly read-only. The trainer validates the existing approved parent path before hierarchical initialization.

This is not an exact resume:

- `resume_mode`: `init_from_joint_checkpoint`
- `replay_restore_status`: `fresh_replay_by_design`
- future optimizer state: fresh
- future replay state: empty until new training fills it
- future RNG state: fresh until the new hierarchical checkpoint is saved

Normal `JointPolicyBundle.load_checkpoint_state()` still rejects `flat_v1` checkpoints loaded into `hierarchical_v1`.

## TDD Evidence

RED run:

- `python -m unittest tests.learn.test_hierarchical_dqn_initialization -v`
- Result before implementation: FAIL.
- Failure reason: `ModuleNotFoundError: No module named 'src.learn.hierarchical_dqn_initialization'`.

GREEN run:

- `python -m unittest tests.learn.test_hierarchical_dqn_initialization -v`
- Result: PASS, 7 tests.

Focused verification:

- `python -m unittest tests.learn.test_hierarchical_dqn_initialization tests.think.test_hierarchical_dqn_policy tests.orchestration.test_torch_joint_runtime -v`
- Result: PASS, 23 tests.

- `python -m unittest tests.learn.test_train_joint_curriculum -v`
- Result: PASS, 46 tests.

- `python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_250k_config_loads_and_is_manual_run_safe -v`
- Result: PASS, 1 test.

- `python -m unittest tests.learn.test_hierarchical_dqn_initialization tests.think.test_hierarchical_dqn_policy tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_250k_config_loads_and_is_manual_run_safe -v`
- Result: PASS, 64 tests.

- `python -m py_compile src\learn\hierarchical_dqn_initialization.py src\learn\train_joint_torch.py tests\learn\test_hierarchical_dqn_initialization.py`
- Result: PASS.

- `python -m py_compile src\learn\hierarchical_dqn_initialization.py src\learn\train_joint_torch.py tests\learn\test_hierarchical_dqn_initialization.py tests\learn\test_curriculum_config.py`
- Result: PASS.

Final combined verification after config readiness:

- `python -m unittest tests.learn.test_hierarchical_dqn_initialization tests.think.test_hierarchical_dqn_policy tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_hierarchical_v1_250k_config_loads_and_is_manual_run_safe tests.learn.test_train_joint_curriculum -v`
- Result: PASS, 110 tests.

Final syntax verification:

- `python -m py_compile src\learn\hierarchical_dqn_initialization.py src\learn\train_joint_torch.py tests\learn\test_hierarchical_dqn_initialization.py tests\learn\test_curriculum_config.py`
- Result: PASS.

## Independent Review

Reviewer verdict:

```text
PATCH_APPROVED_FOR_NEXT_GATE
```

Reviewer summary:

- Flat checkpoints still reject hierarchical exact resume.
- Flat teacher loading is separate, read-only, frozen, contract checked, and architecture checked as `flat_v1`.
- Distillation trains hierarchical external `(batch, 48)` Q output against flat external `(batch, 48)` teacher output using sampled observations only.
- Metadata marks `hierarchical_init_method=flat_teacher_distillation_v1`, `resume_mode=init_from_joint_checkpoint`, and `replay_restore_status=fresh_replay_by_design`.
- Safe to create the first hierarchical 250k config next, without running training.

## Config Readiness

`configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json`

Config properties:

- production checkpoint is the only read-only init/teacher source,
- `dqn_architecture=hierarchical_v1`,
- `hierarchical_initialization.enabled=true`,
- `hierarchical_initialization.method=flat_teacher_distillation_v1`,
- no failed checkpoints as parents,
- `manual_run_safety.training_authorized=false`,
- no training run.

Fresh artifact check:

- `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_250k_20260611`: absent.
- `models/eval/joint_torch_v5_prod_hierarchical_v1_250k_20260611`: absent.
- `models/eval/joint_torch_v5_prod_hierarchical_v1_250k_20260611_offline_scenarios`: absent.

Protected no-mutation proof:

- `models/registry/active_models.json`: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl`: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
- `models/production`: 4 files, 15040339 bytes.
- `models/baselines`: 2692 files, 7392576274 bytes.
- `db`: 7 files, 28330 bytes.
- No Python/Torch training or eval process remained after verification.
