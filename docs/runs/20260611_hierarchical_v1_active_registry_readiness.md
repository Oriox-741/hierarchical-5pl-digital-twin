# Hierarchical V1 Active Registry Readiness

Date: 2026-06-11

## Result

Active readiness verdict: `ACTIVE_REGISTRY_DRY_RUN_READY`

Final classification: `HIERARCHICAL_V1_ACTIVE_REGISTRY_DRY_RUN_READY`

No real activation was run. `active_models.json` was not written. `models.jsonl` was not appended during this task. No production promotion, baseline mutation, DB mutation, checkpoint mutation, training, or offline eval was performed.

## Candidate Pair

Validated registered candidate rows:

- PPO candidate id: `3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- DQN candidate id: `71af6701-ac3a-40f2-bdd9-cc80868d5a1c`
- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Both rows are `status=candidate`, share the same `joint_checkpoint`, and preserve:

- `framework=torch_joint`
- `runtime_loader=joint_torch`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `training_step=1000000`
- `exact_resume_capable=true`
- `eval_verdict=PASS`
- `hard_blocker_status=zero`
- `long_run_gate_verdict=PASS`
- `production_ready=false`
- `baseline_update=false`

## Tooling Patch

The first real-registry activation dry-run without replacement approval failed safely:

```text
ERROR: active registry mapping already active for ppo:continuous_control, dqn:tactical_dispatch.
```

Root cause: the activation CLI correctly protected existing active production mappings, but it had no explicit handoff mode for a reviewed replacement candidate. The patch adds `--replace-active` as an explicit gate:

- Without `--replace-active`, occupied active mappings still fail.
- With `--replace-active`, dry-run validates the candidate pair and prints the current active mappings that would be replaced.
- Future real activation still requires separate `--activate` plus `--replace-active`.
- Candidate registration mode rejects `--replace-active`.

## Dry-Run Activation Command

This command was run and returned `DRY_RUN_OK`:

```powershell
python -m src.learn.register_torch_joint_candidate --logical-model-id joint_torch_v5_prod_hierarchical_v1_1m_20260611 --ppo-candidate-id 3befa11c-e853-4d0f-a951-c2fbbb2a9898 --dqn-candidate-id 71af6701-ac3a-40f2-bdd9-cc80868d5a1c --registry-dir models\registry --dry-run --replace-active
```

Dry-run output showed it would activate:

- `ppo:continuous_control -> models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `dqn:tactical_dispatch -> models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- joint checkpoint: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`

Dry-run output also showed it would replace the current old production active mappings:

- `ppo:continuous_control -> models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\ppo_torch_joint_final_balanced_retention_ft_200k.pt`
- `dqn:tactical_dispatch -> models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\dqn_torch_joint_final_balanced_retention_ft_200k.pt`

Planned future writes, if separately approved:

- `models\registry\models.jsonl`
- `models\registry\active_models.json`

Dry-run confirmed:

- `production_promotion: false`
- `baseline_update: false`

## Future Real Activation Command

Do not run without separate explicit human approval:

```powershell
python -m src.learn.register_torch_joint_candidate --logical-model-id joint_torch_v5_prod_hierarchical_v1_1m_20260611 --ppo-candidate-id 3befa11c-e853-4d0f-a951-c2fbbb2a9898 --dqn-candidate-id 71af6701-ac3a-40f2-bdd9-cc80868d5a1c --registry-dir models\registry --activate --replace-active
```

## Expected Future Mutation Scope

If the real activation command is later approved, expected mutation is registry-active only:

- Append exactly two `status=active` rows to `models/registry/models.jsonl`.
- Update only these active mapping keys in `models/registry/active_models.json`:
  - `ppo:continuous_control`
  - `dqn:tactical_dispatch`
- Preserve existing candidate rows.
- Do not copy or mutate production files.
- Do not mutate baselines.
- Do not mutate DB.
- Do not mutate checkpoints.
- Do not run training or offline eval.

## Runtime Readiness

Read-only runtime checks:

- `load_torch_joint_policy` loaded the candidate joint checkpoint on CPU.
- Deterministic zero observation returned continuous action length `5`.
- Continuous action values were finite.
- Discrete action was `1`, inside `0..47`.

Temporary-registry activation smoke:

- Copied real registry files to a temporary registry.
- Ran activation with `--activate --replace-active` only inside that temporary registry.
- Temp activation appended two active rows.
- `PolicyService.predict_joint(..., fallback_to_heuristic=False)` returned `algorithm=torch_joint`.
- Temp PolicyService discrete action was `1`, inside `0..47`.

No real registry activation was run.

## Tests

TDD RED:

- `python -m unittest tests.learn.test_register_torch_joint_candidate.RegisterTorchJointCandidateTests.test_activation_dry_run_refuses_existing_active_mapping_without_replace_flag tests.learn.test_register_torch_joint_candidate.RegisterTorchJointCandidateTests.test_hierarchical_activation_dry_run_replace_active_writes_nothing tests.learn.test_register_torch_joint_candidate.RegisterTorchJointCandidateTests.test_activate_with_replace_active_updates_temp_active_mapping_only -v`
- Result before implementation: FAIL. `--replace-active` was unrecognized and the default refusal did not point to the explicit handoff path.

Focused GREEN:

- Same command after implementation: PASS, `3` tests.

Required verification:

- `python -m unittest tests.learn.test_register_torch_joint_candidate -v`
  - PASS, `29` tests.
- `python -m unittest tests.learn.test_model_registry -v`
  - PASS, `7` tests.
- `python -m unittest tests.orchestration.test_policy_service_torch_joint -v`
  - PASS, `12` tests.
- `python -m unittest tests.orchestration.test_torch_joint_runtime -v`
  - PASS, `9` tests.
- `python -m py_compile src\learn\register_torch_joint_candidate.py tests\learn\test_register_torch_joint_candidate.py`
  - PASS.

## Rollback Plan

If future activation is approved and later needs rollback:

1. Restore `models/registry/active_models.json` to the old production mapping recorded in this report.
2. Leave candidate and active rows in `models/registry/models.jsonl` for audit unless a separate cleanup plan is explicitly approved.
3. Do not delete checkpoints.
4. Do not mutate production directories.
5. Do not mutate baselines.
6. Do not mutate DB rows.
7. Rerun a read-only registry/runtime audit after rollback.

Current old production mapping:

```json
{
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\\ppo_torch_joint_final_balanced_retention_ft_200k.pt",
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\\dqn_torch_joint_final_balanced_retention_ft_200k.pt"
}
```

## Protected No-Mutation Proof

Protected hashes after dry-run and tests:

- `models/registry/active_models.json`: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl`: `3BD9424A2536E49CE311EC8F994737D20E7494375E3B68398DF0BF90F866E09E`
- source production checkpoint `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles:

- `models/production`: `4` files, `15040339` bytes
- `models/baselines`: `2692` files, `7392576274` bytes
- `db`: `7` files, `28330` bytes

Process check:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

## Independent Review

Reviewer verdict:

```text
ACTIVE_REGISTRY_DRY_RUN_READY
```

No blocking findings. The reviewer confirmed the registered candidate rows, explicit `--replace-active` gate, default refusal on occupied active mappings, non-mutating replacement dry-run output, registry-only future activation scope, runtime readiness evidence, and test coverage.
