# Hierarchical DQN Action Architecture Patch Report

Date: 2026-06-11

## Result

Final classification: `HIERARCHICAL_DQN_ARCHITECTURE_PATCH_READY_FOR_CONFIG`

Patch status: approved for the next gate.

Config/training status: still blocked in this task. The code can represent and load `hierarchical_v1` checkpoints, but no training config was created because flat production-parent initialization into hierarchical heads needs a separate conversion/distillation or partial-initialization decision.

No training, offline eval, registry update, production mutation, baseline mutation, DB mutation, source production checkpoint mutation, existing checkpoint mutation, or new model config was performed.

## Chosen Architecture

Chosen internal DQN architecture: `hierarchical_v1`.

Implementation:

- Preserve external MDP contract: `physical_reality_v5_route_candidate_visibility`.
- Preserve observation dimension: `73`.
- Preserve external discrete action count: `48`.
- Preserve `DiscreteActionMapper` external encode/decode semantics.
- Keep the existing trainer, replay, Bellman loss, runtime, evaluator, and env API expecting an external `(batch, 48)` Q tensor.
- Replace the internal flat 48-logit option with an opt-in additive factorized composer:

```text
Q_external(action) =
  Q_dispatch[dispatch]
  + Q_reorder[reorder]
  + I(dispatch == DISPATCH) * (Q_route[route] + Q_mode[mode])
```

Why this architecture:

- It directly addresses flat composite-action drift while preserving the existing DQN loss surface.
- It makes HOLD actions invariant to route/fleet labels, targeting the action-16 hold label pocket observed after teacher retention.
- It avoids a broad conditional-action replay/loss rewrite in the first patch.

## Files Changed

- `src/think/joint_policies.py`
  - Added `FLAT_DQN_ARCHITECTURE = "flat_v1"`.
  - Added `HIERARCHICAL_DQN_ARCHITECTURE = "hierarchical_v1"`.
  - Added `HierarchicalDQNQNetwork`.
  - Added DQN network factory.
  - Added DQN architecture metadata to policy checkpoints.
  - Added architecture compatibility checks in checkpoint loading.

- `src/orchestration/torch_joint_runtime.py`
  - Runtime loader now instantiates `JointPolicyBundle` using checkpoint `dqn_architecture`, defaulting missing metadata to legacy `flat_v1`.

- `src/eval/real_world_scenario_arena.py`
  - Offline eval loader now instantiates `JointPolicyBundle` using checkpoint `dqn_architecture`, defaulting missing metadata to legacy `flat_v1`.

- `src/learn/train_joint_torch.py`
  - Trainer now reads optional `dqn_architecture` from `torch_joint_training` or `torch_training`.
  - Default remains `flat_v1`.

- `tests/act/test_discrete_action_mapper_contract.py`
  - Added all-48 action encode/decode round-trip coverage.

- `tests/think/__init__.py`
- `tests/think/test_hierarchical_dqn_policy.py`
  - Added hierarchical Q-shape, action composition, HOLD route/mode masking, dispatch-only gradient, checkpoint metadata, and architecture mismatch tests.

- `tests/orchestration/test_torch_joint_runtime.py`
  - Added runtime loading/smoke coverage for hierarchical checkpoints.

- `tests/eval/test_real_world_scenario_evaluator.py`
  - Added offline evaluator loader coverage for hierarchical checkpoints.

- `tests/learn/test_train_joint_curriculum.py`
  - Added trainer config parsing, exact-resume checkpoint metadata, and flat/hierarchical mismatch rejection coverage.

- `docs/plans/20260611_hierarchical_dqn_action_architecture_plan.md`
  - Added architecture implementation plan.

## Checkpoint Compatibility Decision

Rules implemented:

- Legacy checkpoints missing `dqn_architecture` are treated as `flat_v1`.
- `flat_v1` checkpoints load only into `flat_v1` bundles.
- `hierarchical_v1` checkpoints load only into `hierarchical_v1` bundles.
- Mismatches raise `ValueError` with `dqn_architecture mismatch`.
- New checkpoints include:
  - `dqn_architecture`
  - `external_discrete_action_count`
  - `internal_heads`

Decision:

- No implicit flat-to-hierarchical conversion.
- No exact resume from flat production into hierarchical DQN.
- A later config task must choose either fresh hierarchical DQN initialization, explicit partial PPO/flat-to-head initialization, or teacher/distillation conversion.

## Runtime Compatibility Proof

Runtime and eval both load the architecture from checkpoint metadata:

- `load_torch_joint_policy()` returns legal external discrete actions in `[0, 47]`.
- `load_policy_checkpoint()` preserves hierarchical architecture and sets policies to eval mode.
- Env action semantics are unchanged because env still receives the same external integer action id.
- Replay semantics are unchanged because replay still stores external action ids in `[0, 47]`.
- DQN loss semantics are unchanged because both flat and hierarchical networks return `(batch, 48)` Q values.

## TDD Evidence

RED run:

- `python -m unittest tests.act.test_discrete_action_mapper_contract tests.think.test_hierarchical_dqn_policy tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator tests.learn.test_train_joint_curriculum -v`
- Result before implementation: FAIL as expected.
- Failure reason: missing `FLAT_DQN_ARCHITECTURE`, `HIERARCHICAL_DQN_ARCHITECTURE`, `HierarchicalDQNQNetwork`, and trainer architecture helper.

GREEN focused run:

- `python -m unittest tests.act.test_discrete_action_mapper_contract tests.think.test_hierarchical_dqn_policy tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator tests.learn.test_train_joint_curriculum -v`
- Result: PASS, 104 tests.

Additional verification:

- `python -m py_compile src\think\joint_policies.py src\orchestration\torch_joint_runtime.py src\eval\real_world_scenario_arena.py src\learn\train_joint_torch.py tests\act\test_discrete_action_mapper_contract.py tests\think\test_hierarchical_dqn_policy.py tests\orchestration\test_torch_joint_runtime.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py`
- Result: PASS.

- `python -m unittest discover tests\orchestration -v`
- Result: PASS, 20 tests.

- `python -m unittest tests.learn.test_train_joint_curriculum tests.learn.test_promote_torch_joint_production tests.learn.test_register_torch_joint_candidate -v`
- Result: PASS, 83 tests.

## Independent Reviews

Design reviewer verdict:

```text
PATCH_APPROVED_FOR_NEXT_GATE
```

Patch reviewer verdict:

```text
PATCH_APPROVED_FOR_NEXT_GATE
```

Reviewer finding summary:

- `hierarchical_v1` preserves the external `(batch, 48)` DQN contract.
- HOLD route/mode masking is implemented correctly for the stated design.
- Checkpoint compatibility is safe.
- Runtime, eval, and trainer wiring are consistent.
- No configs contain `dqn_architecture`, so no training ladder was started.

## Protected No-Mutation Proof

Protected hashes after patch:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt` SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles after patch:

- `models/production`: 4 files, 15040339 bytes, max write UTC `2026-06-09 02:38:51Z`
- `models/baselines`: 2692 files, 7392576274 bytes, max write UTC `2026-05-24 23:55:21Z`
- `db`: 7 files, 28330 bytes, max write UTC `2026-05-19 14:13:31Z`

Process check:

- No running `python.exe` / `pythonw.exe` training or eval process was found at final verification.

## Next Gate

The architecture patch is ready for a follow-up config-design task, but the follow-up must not treat the flat production checkpoint as an exact-resume-compatible hierarchical parent.

Required next decision before config/training:

- choose fresh hierarchical DQN heads with PPO/feature initialization only, or
- define explicit flat-to-hierarchical distillation/conversion, or
- define another reviewed initialization strategy.

No 250k/500k/1M run is authorized by this patch report alone.
