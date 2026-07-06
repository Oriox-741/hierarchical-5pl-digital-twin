# Hierarchical V1 Active Registry Activation Audit

Date: 2026-06-11

## Result

Final classification: `HIERARCHICAL_V1_ACTIVE_REGISTRY_WRITE_COMPLETE_SAFE`

The explicitly approved real active-registry activation command was run once. It appended exactly two `status=active` rows to `models/registry/models.jsonl` and updated only the PPO/DQN role mappings in `models/registry/active_models.json`.

No production promotion, production directory mutation, baseline mutation, DB mutation, checkpoint mutation, training, or offline eval was performed.

## Command Run

```powershell
python -m src.learn.register_torch_joint_candidate --logical-model-id joint_torch_v5_prod_hierarchical_v1_1m_20260611 --ppo-candidate-id 3befa11c-e853-4d0f-a951-c2fbbb2a9898 --dqn-candidate-id 71af6701-ac3a-40f2-bdd9-cc80868d5a1c --registry-dir models\registry --activate --replace-active
```

Result:

```text
ACTIVATION_COMPLETE
```

## Active Registry IDs

New PPO active row:

- `model_id`: `17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- `algorithm`: `ppo`
- `agent_role`: `continuous_control`
- `status`: `active`
- `promoted_from_candidate_id`: `3befa11c-e853-4d0f-a951-c2fbbb2a9898`

New DQN active row:

- `model_id`: `f87e10d6-479f-44fc-99d1-6925bc9cb346`
- `algorithm`: `dqn`
- `agent_role`: `tactical_dispatch`
- `status`: `active`
- `promoted_from_candidate_id`: `71af6701-ac3a-40f2-bdd9-cc80868d5a1c`

## Active Models Mapping

`models/registry/active_models.json` now points to the hierarchical v1 1M final artifacts:

```json
{
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt",
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt"
}
```

## Registry Audit

Before activation:

- `models/registry/models.jsonl` rows: `8`
- old production active rows in `models.jsonl`: `2`
- hierarchical candidate rows: `2`
- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `3BD9424A2536E49CE311EC8F994737D20E7494375E3B68398DF0BF90F866E09E`

After activation:

- `models/registry/models.jsonl` rows: `10`
- old production active rows in `models.jsonl`: `2`
- hierarchical candidate rows still `status=candidate`: `2`
- hierarchical active rows: `2`
- `models/registry/active_models.json` SHA256: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl` SHA256: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`

The row count increased by exactly `2`.

## Active Metadata Audit

Both new active rows include:

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
- `production_promotion`: `false`
- `baseline_update`: `false`
- `activation_scope`: `registry_active_only`
- `contract`: `physical_reality_v5_route_candidate_visibility`
- `obs_dim`: `73`
- `action_count`: `48`
- `external_discrete_action_count`: `48`

## Runtime Audit

`PolicyService.predict_joint(..., fallback_to_heuristic=False)` now serves the active hierarchical pair:

- `algorithm`: `torch_joint`
- joint checkpoint: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`
- PPO artifact: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- DQN artifact: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- continuous action length: `5`
- discrete action: `1`, valid in `0..47`
- SB3 PPO loader calls: `0`
- SB3 DQN loader calls: `0`

## Protected No-Mutation Proof

Protected root profiles before and after activation stayed unchanged:

- `models/production`: `4` files, `15040339` bytes
- `models/baselines`: `2692` files, `7392576274` bytes
- `db`: `7` files, `28330` bytes
- `models/checkpoints`: `787` files, `4920830666` bytes

Protected checkpoint hashes stayed unchanged:

- source production checkpoint `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
- hierarchical joint checkpoint `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- hierarchical PPO final artifact `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611/ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`: `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996`
- hierarchical DQN final artifact `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611/dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`: `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D`

Process check:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

## Rollback Note

If active-registry rollback is required, restore `models/registry/active_models.json` to the previous old-production mapping:

```json
{
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\\ppo_torch_joint_final_balanced_retention_ft_200k.pt",
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\\dqn_torch_joint_final_balanced_retention_ft_200k.pt"
}
```

Leave candidate and active registry rows in `models.jsonl` for audit unless a separate cleanup plan is explicitly approved. Do not delete checkpoints, mutate production, mutate baselines, or mutate DB rows.
