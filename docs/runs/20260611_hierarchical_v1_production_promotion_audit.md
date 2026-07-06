# Hierarchical V1 Production Promotion Audit

Date: 2026-06-11

Final classification: `HIERARCHICAL_V1_PRODUCTION_PROMOTION_COMPLETE_SAFE`

## Scope

This was the explicitly approved copy-only production promotion for active logical model:

`joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Allowed mutation was limited to creating:

`models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611`

and writing exactly these files:

- `joint_torch_latest.pt`
- `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `production_manifest.json`

No registry update, baseline update, DB mutation, checkpoint mutation, training, or offline evaluation was performed.

## Human Residual Watch Acceptance

Human approver: `egeme`

Accepted residual watches:

- `route_disruption_congestion` service below production accepted because scenario threshold and long-run gate passed.
- `mixed_stress` service below production accepted because scenario threshold and long-run gate passed.
- top-action concentration warnings accepted as nonfatal.
- mixed-success route-failure warnings accepted as nonfatal.

The manifest records `residual_watches_accepted=true`.

## Pre-Copy Evidence

Promotion plan read:

`docs\runbooks\20260611_hierarchical_v1_production_promotion_plan.md`

Target production directory preflight:

- `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- existed before copy: `false`

Active registry before copy pointed to hierarchical v1 1M:

- `ppo:continuous_control` -> `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `dqn:tactical_dispatch` -> `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`

Registry IDs:

| Role | Status | ID |
| --- | --- | --- |
| PPO continuous control | candidate | `3befa11c-e853-4d0f-a951-c2fbbb2a9898` |
| DQN tactical dispatch | candidate | `71af6701-ac3a-40f2-bdd9-cc80868d5a1c` |
| PPO continuous control | active | `17ba1d28-0054-4f7c-ae9a-34cd305ebb89` |
| DQN tactical dispatch | active | `f87e10d6-479f-44fc-99d1-6925bc9cb346` |

Checkpoint metadata validated before copy:

- `checkpoint_version=torch_joint_policy_v1`
- `artifact_kind=joint_final`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `global_step=1000000`
- `resume_state.exact_resume_capable=true`
- `resume_state.dqn_replay_buffer_included=true`
- `resume_state.rng_state_included=true`
- contract `physical_reality_v5_route_candidate_visibility`
- observation dimension `73`
- discrete action count `48`

Evaluation and gate evidence:

- scenarios: `8`
- scenario verdicts: `8/8 PASS`
- episode rows: `160`
- hard blockers: `zero`
- long-run gate: `PASS`
- long-run gate exit code: `0` in `docs\runs\20260611_hierarchical_dqn_ladder_report.md`
- gate JSON fatal failures: `[]`

## Copied Files

Production directory created:

`models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Copied artifacts:

| Artifact | Source SHA256 | Production SHA256 |
| --- | --- | --- |
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` |

Manifest:

- path: `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\production_manifest.json`
- SHA256: `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`
- promotion timestamp: `2026-06-11T11:53:41.6830950Z`

Manifest fields validated:

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

The production directory contains exactly:

- `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `joint_torch_latest.pt`
- `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `production_manifest.json`

No extra or missing files were found.

## Runtime Smoke

Copied production joint checkpoint loaded read-only on CPU with:

`load_torch_joint_policy("models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt", device="cpu")`

Deterministic zero-observation prediction:

- continuous length: `5`
- continuous finite: `true`
- continuous min/max: `-0.22180308401584625` / `0.9940987825393677`
- continuous bounded in `[-1, 1]`: `true`
- discrete action: `1`
- discrete legal in `0..47`: `true`

Active `PolicyService.predict_joint(..., fallback_to_heuristic=False)` still resolves the active hierarchical checkpoint through the registry:

- algorithm: `torch_joint`
- active joint path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`
- active PPO path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- active DQN path: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- SB3 PPO loader calls: `0`
- SB3 DQN loader calls: `0`

## Scenario Table

| Scenario | Production service | H1M service | Production lateness | H1M lateness | Production dispatch success | H1M dispatch success | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_normal` | `0.938` | `0.990` | `0.000` | `0.000` | `1.000` | `1.000` | PASS |
| `demand_spike_volatility` | `0.843` | `0.859` | `0.177` | `0.027` | `1.000` | `1.000` | PASS |
| `high_holding_cost` | `0.943` | `0.966` | `0.000` | `0.000` | `1.000` | `0.999` | PASS |
| `lead_time_volatility` | `0.936` | `1.000` | `0.000` | `0.000` | `1.000` | `0.999` | PASS |
| `mixed_stress` | `0.951` | `0.942` | `0.000` | `0.000` | `0.996` | `0.999` | PASS |
| `premium_sla_pressure` | `0.980` | `1.000` | `0.000` | `0.000` | `0.769` | `0.999` | PASS |
| `route_disruption_congestion` | `0.929` | `0.901` | `0.000` | `0.008` | `0.822` | `0.999` | PASS |
| `vehicle_scarcity_capacity_shock` | `0.914` | `0.949` | `0.000` | `0.001` | `1.000` | `1.000` | PASS |

## Protected No-Mutation Proof

Registry hashes after copy:

- `models\registry\active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models\registry\models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`

These match the pre-copy hashes, so registry was unchanged.

Protected path profiles after copy:

- `models\production`: `8` files, `328265276` bytes
- `models\baselines`: `2692` files, `7392576274` bytes
- `db`: `7` files, `28330` bytes
- `models\checkpoints`: `787` files, `4920830666` bytes
- `models\eval`: `120` files, `354070196` bytes

The expected production profile changed only because the new production directory and four approved files were added. Baselines, DB, checkpoints, and eval profiles match their pre-copy values.

Protected hashes after copy:

- prior production source checkpoint: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
- hierarchical source joint checkpoint: `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- hierarchical source PPO final: `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996`
- hierarchical source DQN final: `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D`

These match the pre-copy hashes. Source checkpoints were unchanged.

Process check after copy:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

## Result

The hierarchical v1 1M model has been promoted by copy-only production artifact creation. Active registry already routes to this model and was not modified by this task. Production artifacts are now available under:

`models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Final classification:

`HIERARCHICAL_V1_PRODUCTION_PROMOTION_COMPLETE_SAFE`
