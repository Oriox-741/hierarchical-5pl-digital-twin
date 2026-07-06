# Hierarchical V1 Candidate Registration Audit

Date: 2026-06-11

## Result

Final classification: `HIERARCHICAL_V1_CANDIDATE_REGISTRATION_COMPLETE_SAFE`

The explicitly approved real candidate registration command was run once. It appended exactly two candidate rows to `models/registry/models.jsonl`.

No activation, production promotion, baseline mutation, DB mutation, checkpoint mutation, training, or offline eval was performed.

## Command Run

```powershell
python -m src.learn.register_torch_joint_candidate --logical-model-id joint_torch_v5_prod_hierarchical_v1_1m_20260611 --registry-dir models\registry --joint-checkpoint models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt --ppo-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt --dqn-path models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt --eval-output-dir models\eval\joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios --gate-result docs\runs\20260611_hierarchical_dqn_1m_gate_result.json --dqn-architecture hierarchical_v1 --hierarchical-init-method flat_teacher_distillation_v1 --register-candidate
```

Result:

```text
CANDIDATE_REGISTRATION_COMPLETE
```

## Registered Candidate Rows

PPO candidate:

- `model_id`: `3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- `algorithm`: `ppo`
- `agent_role`: `continuous_control`
- `artifact_kind`: `ppo_final`
- `status`: `candidate`
- `path`: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`

DQN candidate:

- `model_id`: `71af6701-ac3a-40f2-bdd9-cc80868d5a1c`
- `algorithm`: `dqn`
- `agent_role`: `tactical_dispatch`
- `artifact_kind`: `dqn_final`
- `status`: `candidate`
- `path`: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`

## Registry Audit

Before registration:

- `models/registry/models.jsonl` rows: `6`
- hierarchical candidate rows for `joint_torch_v5_prod_hierarchical_v1_1m_20260611`: `0`
- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`

After registration:

- `models/registry/models.jsonl` rows: `8`
- hierarchical candidate rows for `joint_torch_v5_prod_hierarchical_v1_1m_20260611`: `2`
- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `3BD9424A2536E49CE311EC8F994737D20E7494375E3B68398DF0BF90F866E09E`

`active_models.json` stayed byte/hash identical. Its active paths still point to the current production balanced-retention PPO/DQN artifacts.

## Metadata Audit

Both new candidate rows include:

- `logical_model_id`: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- `framework`: `torch_joint`
- `runtime_loader`: `joint_torch`
- `dqn_architecture`: `hierarchical_v1`
- `hierarchical_init_method`: `flat_teacher_distillation_v1`
- `training_step`: `1000000`
- `exact_resume_capable`: `true`
- `eval_verdict`: `PASS`
- `hard_blocker_status`: `zero`
- `long_run_gate_verdict`: `PASS`
- `production_ready`: `false`
- `baseline_update`: `false`
- `registry_scope`: `candidate_only`
- `contract`: `physical_reality_v5_route_candidate_visibility`
- `obs_dim`: `73`
- `action_count`: `48`
- `external_discrete_action_count`: `48`

## Protected No-Mutation Proof

Protected hashes after registration:

- `models/registry/active_models.json`: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- source production checkpoint `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles after registration:

- `models/production`: `4` files, `15040339` bytes
- `models/baselines`: `2692` files, `7392576274` bytes
- `db`: `7` files, `28330` bytes

Process check:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

## Next Safe Step

The candidate is registered for review only. Activation remains a separate step and must begin with a separate explicit approval and dry-run review using the two candidate ids above.
