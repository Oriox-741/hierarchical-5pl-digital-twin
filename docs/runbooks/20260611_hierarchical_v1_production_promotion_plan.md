# Hierarchical V1 Production Promotion Plan

Date: 2026-06-11

## Scope

This is a production promotion readiness plan only. It does not promote production, create or copy production files, update baselines, mutate DB rows, mutate checkpoints, mutate registry files, train, or rerun offline evaluation.

Recommendation: `PRODUCTION_PROMOTION_PLAN_READY_FOR_MANUAL_APPROVAL`

## Active Registry State

Active logical model:

`joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Active registry mappings:

- `ppo:continuous_control` -> `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `dqn:tactical_dispatch` -> `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`

Registry IDs:

| Role | Algorithm | Status | Registry ID |
| --- | --- | --- | --- |
| `continuous_control` | PPO | candidate | `3befa11c-e853-4d0f-a951-c2fbbb2a9898` |
| `tactical_dispatch` | DQN | candidate | `71af6701-ac3a-40f2-bdd9-cc80868d5a1c` |
| `continuous_control` | PPO | active | `17ba1d28-0054-4f7c-ae9a-34cd305ebb89` |
| `tactical_dispatch` | DQN | active | `f87e10d6-479f-44fc-99d1-6925bc9cb346` |

The old production active rows remain in `models/registry/models.jsonl` for audit.

## Active Checkpoint Contract

Source artifacts:

- joint checkpoint: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt`
- PPO final: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- DQN final: `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`

Validated metadata:

- `checkpoint_version=torch_joint_policy_v1`
- `artifact_kind=joint_final`
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `global_step=1000000`
- `resume_state.exact_resume_capable=true`
- contract `physical_reality_v5_route_candidate_visibility`
- observation dimension `73`
- discrete action count `48`

File hashes:

| Artifact | SHA256 |
| --- | --- |
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` |

## Evaluation And Gate Evidence

Evaluation output:

`models\eval\joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`

Evidence:

- scenarios: `8`
- scenario verdicts: `8/8 PASS`
- hard-blocker verdicts: `8/8 PASS`
- episode rows: `160`
- long-run gate: `PASS`
- fatal gate failures: `0`

Evaluation/gate evidence hashes:

- `scenario_summary.json`: `9655E531F80728F5F3EB90BF53FA8BF5E5EECAE88F61868D17DB51781781B238`
- `episode_metrics.jsonl`: `F453A82E4B3CE67265AE266308A1D90B875B182B4EA8A70DBA6EB108CB9A383D`
- `docs\runs\20260611_hierarchical_dqn_1m_gate_result.json`: `01DDF3FDA564623B91C8A8CBD79DABB18B2DBE977C19C953AFEAF765FD0BEEC9`

Residual watches requiring explicit human acceptance before production copy:

- `residual_watches_accepted=false`
- `route_disruption_congestion` service below production: `0.901` vs `0.929`
- `mixed_stress` service below production: `0.942` vs `0.951`
- top-action concentration warnings remain
- mixed-success route-failure warnings remain

## Runtime Evidence

Read-only runtime smoke:

- `load_torch_joint_policy` loaded the active joint checkpoint on CPU.
- Deterministic zero observation returned continuous action length `5`.
- Continuous values were finite and bounded in `[-1, 1]`.
- Discrete action was `1`, valid in `0..47`.
- `PolicyService.predict_joint(..., fallback_to_heuristic=False)` returned `algorithm=torch_joint`.
- SB3 PPO loader calls: `0`.
- SB3 DQN loader calls: `0`.

## Production Promotion Target

Do not create this directory without separate explicit approval:

`models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Preflight finding:

- target directory exists now: `false`

Future copy targets:

| Source | Target |
| --- | --- |
| `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt` | `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt` |
| `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` |
| `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` |

Future manifest target:

`models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\production_manifest.json`

## Required Manifest Fields

The future `production_manifest.json` should include:

- `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- `active_ppo_model_id=17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- `active_dqn_model_id=f87e10d6-479f-44fc-99d1-6925bc9cb346`
- `candidate_ppo_model_id=3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- `candidate_dqn_model_id=71af6701-ac3a-40f2-bdd9-cc80868d5a1c`
- source paths for joint, PPO, and DQN artifacts
- target paths for joint, PPO, DQN, and manifest artifacts
- file SHA256 hashes for copied artifacts
- `dqn_architecture=hierarchical_v1`
- `hierarchical_init_method=flat_teacher_distillation_v1`
- `contract=physical_reality_v5_route_candidate_visibility`
- `observation_dim=73`
- `discrete_action_count=48`
- `training_step=1000000`
- `eval_verdict=PASS`
- `hard_blocker_status=zero`
- `long_run_gate_verdict=PASS`
- `residual_watches_accepted=false` unless explicitly accepted by the human approver
- `baseline_update=false`
- `db_mutation=false`
- `registry_mutation=false`
- promotion timestamp
- human approver

## Future Production Copy Command Shape

This is a plan shape only, not an executed command.

1. Create `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611`.
2. Copy exactly the three `.pt` source artifacts listed above.
3. Write `production_manifest.json` with the required fields.
4. Rehash copied files and confirm they match source hashes.
5. Load the copied `joint_torch_latest.pt` read-only on CPU.
6. Run one deterministic zero-observation runtime smoke against the copied checkpoint.

Do not mutate registry as part of this copy-only promotion because active registry already points to this model. Any future registry metadata change must be a separate explicit registry-write task.

## Protected No-Mutation Proof

Protected state during this planning task:

- `models\production`: `4` files, `15040339` bytes
- `models\baselines`: `2692` files, `7392576274` bytes
- `db`: `7` files, `28330` bytes
- `models\checkpoints`: `787` files, `4920830666` bytes
- `models\eval`: `120` files, `354070196` bytes
- source production checkpoint hash: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
- active registry hash: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- model registry hash: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`

Process check:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

## Rollback Plan

If a future production promotion is approved and then must be rolled back:

1. If the production directory was created, quarantine `models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611` only with explicit approval.
2. Keep source checkpoint artifacts under `models\checkpoints` intact.
3. Do not mutate baselines.
4. Do not mutate DB rows.
5. If active-registry rollback is required, restore `models\registry\active_models.json` to the old production active pair:
   - `ppo:continuous_control -> models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\ppo_torch_joint_final_balanced_retention_ft_200k.pt`
   - `dqn:tactical_dispatch -> models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\dqn_torch_joint_final_balanced_retention_ft_200k.pt`
6. Leave candidate and active rows in `models\registry\models.jsonl` for audit unless a separate cleanup plan is explicitly approved.
7. Rerun a read-only registry/runtime/production audit after rollback.

## Manual Approval Gate

Production promotion can proceed only after separate explicit approval that accepts or rejects the residual watches above and names the human approver for the manifest.

## Independent Review

Reviewer verdict:

```text
PRODUCTION_PROMOTION_PLAN_READY_FOR_MANUAL_APPROVAL
```

No blocking findings. The reviewer confirmed active/candidate/old active registry state, checkpoint metadata, eval/gate evidence, residual watch handling, runtime path evidence, copy-only production target, manifest fields, rollback plan, and protected-path no-mutation proof.

Final classification: `HIERARCHICAL_V1_PRODUCTION_PROMOTION_PLAN_READY`
