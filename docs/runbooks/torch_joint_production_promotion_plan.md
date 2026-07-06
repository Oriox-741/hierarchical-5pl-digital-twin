# Torch Joint Production Promotion Plan

Date: 2026-06-09

## Scope

This runbook documents the next manual production-promotion decision for the active Torch joint runtime model. It is a planning artifact only. Creating or updating this file does not promote production, update baselines, mutate DB rows, mutate checkpoints, train, or rerun evaluation.

Final recommendation: `PRODUCTION_PROMOTION_READY_FOR_MANUAL_APPROVAL`.

## Current Active Registry State

Active logical model:
`joint_torch_v5_balanced_retention_ft_200k_20260608`

Active registry mappings in `models/registry/active_models.json`:
- `ppo:continuous_control` -> `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/ppo_torch_joint_final_balanced_retention_ft_200k.pt`
- `dqn:tactical_dispatch` -> `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/dqn_torch_joint_final_balanced_retention_ft_200k.pt`

Joint checkpoint used by both active records:
`models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`

Registry rows:
- Original candidate PPO id: `556b047a-7178-43bc-832c-0f9b49e6ef3c`
- Original candidate DQN id: `9fe92e2b-3ac2-46c3-9ade-3b6e3d8d5686`
- Active PPO id: `1a5b1fd0-a1be-4818-9b5b-fab84aca3c71`
- Active DQN id: `fbf7b25f-c083-4246-aac9-3f88993b43db`

Active metadata that must remain true:
- `framework=torch_joint`
- `runtime_loader=joint_torch`
- `contract=physical_reality_v5_route_candidate_visibility`
- `obs_dim=73`
- `action_count=48`
- `training_step=200000`
- `parent_step=1000000`
- `eval_verdict=PASS`
- `hard_blocker_status=zero`
- `production_promotion=false`
- `baseline_update=false`
- `activation_scope=registry_active_only`

## Runtime Evidence

`PolicyService.predict_joint(..., fallback_to_heuristic=False)` must resolve the active pair through the Torch joint runtime path and return:
- `algorithm=torch_joint`
- a finite `float32` continuous action vector of length `5`
- a discrete action in `[0, 47]`
- no SB3 `.zip` loader usage for the Torch `.pt` artifacts

## Offline Evaluation Evidence

Evaluation output:
`models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios`

Required evidence before production promotion:
- all `8` scenarios are present
- `24` episode rows are present
- all scenario threshold verdicts are `PASS`
- all hard-blocker verdicts are `PASS`
- final scenario verdicts are `PASS`
- all global hard-blocker counters are zero:
  - `fake_dispatch_credit`
  - `customer_revisited`
  - `route_failure_positive_dispatch_credit`
  - `nan_inf_detected`
  - `dqn_local_negative_positive_train_rows`
  - `no_current_work_dqn_delivery_credit`
  - `hold_delivery_credit_leak`
  - `action8_route_or_delivery_credit_leak`
  - `unsafe_24_25_candidate_credit`
  - `no_work_positive_dqn_local`
  - `emergency_zero_useful_positive_credit`

Residual watches that require explicit human acceptance:
- `route_disruption_congestion.high_resilience_selected_when_not_best_count=411`
- `demand_spike_volatility.true_lateness_pressure=0.17686320800109925`
- `mixed_success_route_failure_steps=1` in `baseline_normal`, `lead_time_volatility`, and `mixed_stress`

## Production Promotion Prerequisites

Before any production write:
- Confirm the active registry still resolves only the intended PPO/DQN role keys.
- Confirm `models/registry/models.jsonl` still contains the original candidate rows and exactly one active PPO/DQN pair for this logical model.
- Confirm the joint checkpoint exists and has:
  - `checkpoint_version=torch_joint_policy_v1`
  - `artifact_kind=joint_final`
  - `global_step=200000`
  - `observation_dim=73`
  - `continuous_action_dim=5`
  - `discrete_action_count=48`
  - v5 contract in the saved config: `physical_reality_v5_route_candidate_visibility`
- Confirm the parent checkpoint remains:
  `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt`
- Confirm `models/production` is absent or has been explicitly reviewed before creating a new production directory.
- Confirm no train or evaluation process is running.
- Confirm the human reviewer accepts the residual watches above.

## Future Production Files

Future manual production promotion should create this directory:
`models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/`

Copy exactly these checkpoint artifacts:
- from `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`
  to `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`
- from `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/ppo_torch_joint_final_balanced_retention_ft_200k.pt`
  to `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/ppo_torch_joint_final_balanced_retention_ft_200k.pt`
- from `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/dqn_torch_joint_final_balanced_retention_ft_200k.pt`
  to `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/dqn_torch_joint_final_balanced_retention_ft_200k.pt`

Write exactly this production manifest:
`models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/production_manifest.json`

The manifest should include:
- logical model id
- active PPO and DQN registry ids
- source checkpoint paths
- copied production paths
- contract
- observation/action dimensions
- training step and parent step
- eval output path
- eval verdict
- hard-blocker status
- residual watch acceptance note
- promotion timestamp
- human approver

No registry files are required for a copy-only production promotion because the active registry is already set. Any future registry metadata change to mark `production_promotion=true` must be a separate, explicit registry-write task.

## Baseline Separation

Baseline update remains separate later work.

Do not copy anything into `models/baselines`.
Do not overwrite historical baselines.
Do not mark this model as a baseline as part of production promotion.

## Post-Promotion Smoke Plan

After the future production copy only:
- Verify the three copied `.pt` files exist under the production directory.
- Load `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt` read-only on CPU with `load_torch_joint_policy`.
- Run one deterministic zero-observation prediction and verify:
  - continuous action shape `(5,)`
  - finite continuous action values
  - continuous action values bounded in `[-1, 1]`
  - discrete action in `[0, 47]`
- Verify `PolicyService` still resolves the active registry pair, unless a later approved production-pointer mechanism replaces registry-active routing.
- Confirm no SB3 loader is used for Torch joint `.pt` artifacts.
- Confirm production files have no unexpected size or timestamp mismatch against the source artifacts.

This smoke plan is not offline evaluation and must not launch scenario evaluation.

## Rollback Plan

If production promotion must be rolled back:
- Restore `models/registry/active_models.json` to the approved pre-production state if the future promotion also changed active routing.
- If the future promotion is copy-only, leave active registry unchanged unless a separate active rollback is explicitly approved.
- Do not delete candidate rows from `models/registry/models.jsonl`.
- Do not delete source checkpoints under `models/checkpoints`.
- Move or quarantine the production directory only with explicit human approval.
- Do not modify baselines.
- Do not mutate DB rows.
- Rerun a read-only registry/runtime audit immediately after rollback.

## Non-Goals

This production promotion must not:
- no baseline overwrite
- overwrite baselines
- mutate DB rows
- mutate source checkpoints
- train
- rerun offline evaluation
- rerun registry activation
- append registry rows without a separate approved registry task

## Final Decision Gate

Production promotion is ready for manual approval only if:
- active registry and runtime checks remain clean
- offline evaluation evidence remains complete
- residual watches are explicitly accepted by the human reviewer
- production copy targets are reviewed
- rollback owner and approver are named before the copy

Decision: `PRODUCTION_PROMOTION_READY_FOR_MANUAL_APPROVAL`.
