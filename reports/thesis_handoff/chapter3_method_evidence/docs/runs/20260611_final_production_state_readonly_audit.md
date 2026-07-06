# Final Production State Read-Only Audit

Date: 2026-06-11

Final classification: `FINAL_PRODUCTION_STATE_AUDIT_CLEAN`

## Scope

This was a final read-only production state audit after the hierarchical v1 1M production handoff.

No training, offline evaluation, registry mutation, production mutation, baseline mutation, DB mutation, or checkpoint mutation was performed. The only write was this audit report.

## Source Documents Read

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_hierarchical_v1_production_promotion_audit.md`
- `docs/runs/20260611_hierarchical_v1_active_registry_activation_audit.md`
- `docs/runs/20260611_hierarchical_dqn_ladder_report.md`

## Active Registry Verification

`models/registry/active_models.json` points to the hierarchical v1 1M PPO/DQN artifacts:

```json
{
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt",
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt"
}
```

Registry hashes:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`

Result: active registry state matches the final release handoff.

## Production Directory Verification

Production directory exists:

`models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611`

It contains exactly these files:

- `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `joint_torch_latest.pt`
- `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `production_manifest.json`

No missing or extra files were found.

Production file hashes:

| File | SHA256 | Bytes |
| --- | --- | ---: |
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` | `303413903` |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` | `4903475` |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` | `4903475` |
| `production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` | `4084` |

## Manifest Handoff Verification

`production_manifest.json` matches the final handoff on all audited fields:

- `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- `active_ppo_id=17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- `active_dqn_id=f87e10d6-479f-44fc-99d1-6925bc9cb346`
- `candidate_ppo_id=3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- `candidate_dqn_id=71af6701-ac3a-40f2-bdd9-cc80868d5a1c`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `contract=physical_reality_v5_route_candidate_visibility`
- `observation_dim=73`
- `action_dim=48`
- `training_step=1000000`
- `eval_verdict=PASS`
- `hard_blocker_status=zero`
- `long_run_gate_verdict=PASS`
- `residual_watches_accepted=true`
- `baseline_update=false`
- `db_mutation=false`
- `registry_mutation=false`
- `human_approver=egeme`

Result: manifest-handoff consistency checks passed.

## Runtime Verification

Production checkpoint loaded read-only on CPU with `load_torch_joint_policy`:

`models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`

Loaded checkpoint metadata:

- `checkpoint_version=torch_joint_policy_v1`
- `artifact_kind=joint_final`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `global_step=1000000`
- `resume_state.exact_resume_capable=true`
- `contract=physical_reality_v5_route_candidate_visibility`
- `observation_dim=73`
- `discrete_action_count=48`

Deterministic zero-observation prediction:

- continuous length: `5`
- continuous finite: `true`
- continuous bounded in `[-1, 1]`: `true`
- continuous min/max: `-0.22180308401584625` / `0.9940987825393677`
- discrete action: `1`
- discrete action legal in `0..47`: `true`

## PolicyService Verification

`PolicyService.predict_joint(..., fallback_to_heuristic=False)` served the active hierarchical model:

- algorithm: `torch_joint`
- active joint path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`
- active PPO path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- active DQN path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- continuous length: `5`
- continuous finite and bounded: `true`
- discrete action: `1`
- discrete action legal in `0..47`: `true`
- SB3 PPO loader calls: `0`
- SB3 DQN loader calls: `0`

Result: runtime route is the Torch joint hierarchical path, not SB3 fallback.

## Protected State Verification

Protected path profiles:

| Path | File count | Total bytes | Status |
| --- | ---: | ---: | --- |
| `models\baselines` | `2692` | `7392576274` | unchanged |
| `db` | `7` | `28330` | unchanged |
| `models\checkpoints` | `787` | `4920830666` | unchanged |
| `models\eval` | `120` | `354070196` | unchanged |

Source and production artifact hashes checked:

- production joint: `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- production PPO: `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996`
- production DQN: `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D`
- source checkpoint joint: `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- source checkpoint PPO: `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996`
- source checkpoint DQN: `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D`

No protected drift was found.

## Process Verification

Narrow process scan result:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

No training, offline evaluation, or gate process was running.

## Dashboard Verification

`docs/00_PROJECT_DASHBOARD.md` points to:

- final release handoff: `docs/releases/20260611_hierarchical_v1_1m_production_handoff`
- current production checkpoint: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`

Result: dashboard points to the final release handoff and current production checkpoint.

## Result

All requested read-only checks passed. The current production state matches the hierarchical v1 1M production handoff.

Final classification:

`FINAL_PRODUCTION_STATE_AUDIT_CLEAN`
