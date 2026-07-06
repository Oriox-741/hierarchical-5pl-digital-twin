# Torch Joint Runtime Promotion Checklist

Use this checklist before any active registry write for the reviewed Torch joint candidate.

## Candidate Identity

- [ ] PPO candidate id matches `556b047a-7178-43bc-832c-0f9b49e6ef3c`.
- [ ] DQN candidate id matches `9fe92e2b-3ac2-46c3-9ade-3b6e3d8d5686`.
- [ ] Logical model id is `joint_torch_v5_balanced_retention_ft_200k_20260608`.
- [ ] Candidate env id is `joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k`.
- [ ] Candidate remains inactive until the approved manual active write.

## Checkpoint Contract

- [ ] Candidate joint checkpoint exists:
  `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`.
- [ ] Checkpoint metadata has `checkpoint_version=torch_joint_policy_v1`.
- [ ] Checkpoint metadata has `artifact_kind=joint_final`.
- [ ] Contract is `physical_reality_v5_route_candidate_visibility`.
- [ ] Observation/action metadata is `73/48`.
- [ ] `global_step=200000`.

## Offline Evaluation Evidence

- [ ] Offline scenario evaluation has 8/8 scenarios PASS.
- [ ] Offline scenario evaluation has 24 episode rows.
- [ ] Global hard blockers are zero.
- [ ] Residual watches are explicitly accepted by a human reviewer:
  - [ ] `route_disruption high_resilience_selected_when_not_best_count=411`.
  - [ ] `demand_spike lateness=0.177`.
  - [ ] `mixed_success_route_failure_steps` present in baseline/lead_time/mixed.

## Runtime And CLI Gates

- [ ] Task 1 Torch joint loader tests pass.
- [ ] Task 2 `PolicyService` paired-active Torch joint tests pass.
- [ ] Task 3 real candidate read-only runtime test passes.
- [ ] Task 4 activation CLI tests pass.
- [ ] Dry-run output is reviewed and shows `DRY_RUN_OK`.
- [ ] Dry-run planned only these active mappings:
  - [ ] `ppo:continuous_control -> models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/ppo_torch_joint_final_balanced_retention_ft_200k.pt`.
  - [ ] `dqn:tactical_dispatch -> models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/dqn_torch_joint_final_balanced_retention_ft_200k.pt`.
- [ ] Dry-run planned joint checkpoint:
  `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`.
- [ ] Dry-run planned writes only to:
  - [ ] `models/registry/models.jsonl`.
  - [ ] `models/registry/active_models.json`.
- [ ] Dry-run confirms `production_promotion=false`.
- [ ] Dry-run confirms `baseline_update=false`.

## Manual Active Write Gate

- [ ] Real active write is performed manually.
- [ ] Real active write is run once only.
- [ ] `--activate` is not run until the dry-run output and this checklist are reviewed.
- [ ] Active write appends active records separately; candidate rows are not deleted or rewritten.
- [ ] `active_models.json` is written only with:
  - [ ] `ppo:continuous_control`.
  - [ ] `dqn:tactical_dispatch`.

## Post-Active Audit

- [ ] Post-active audit is run immediately after activation.
- [ ] `active_models.json` contains only the intended PPO/DQN role keys.
- [ ] `models.jsonl` has exactly two new active records for the expected logical model id.
- [ ] Original two candidate records remain present with status `candidate`.
- [ ] `PolicyService.predict_joint(..., fallback_to_heuristic=False)` returns `algorithm="torch_joint"`.
- [ ] No SB3 loader is called for Torch joint artifacts.
- [ ] No production directory is created or modified.
- [ ] No baseline directory is overwritten.
- [ ] No checkpoint file is changed.
- [ ] No DB row is inserted, updated, or deleted.

## Separate Later Work

- [ ] Production promotion remains separate later work.
- [ ] Baseline update remains separate later work.
- [ ] A separate production promotion plan must be created and reviewed before any production copy or deployment step.
- [ ] A separate baseline update decision must be reviewed before any baseline overwrite.

## Rollback Plan

If activation causes a runtime problem:

- [ ] Restore `models/registry/active_models.json` to `{}`.
- [ ] Do not delete candidate rows.
- [ ] Do not delete active rows; leave them for audit unless a separate registry cleanup plan is approved.
- [ ] Do not delete checkpoints.
- [ ] Do not modify baselines.
- [ ] Do not modify production.
- [ ] Do not mutate DB rows.
- [ ] Rerun registry/runtime audit after rollback.
- [ ] Confirm `PolicyService` no longer resolves an active Torch joint pair after `active_models.json` is restored to `{}`.

## Non-Goals

- No production promotion.
- No baseline overwrite.
- No checkpoint mutation.
- No DB mutation.
- No training.
- No evaluation rerun.
- No smoke run.
