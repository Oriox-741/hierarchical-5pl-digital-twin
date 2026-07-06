# Torch Joint V5 Balanced Retention Production Handoff

Date: 2026-06-09

Final classification: `TORCH_JOINT_V5_BALANCED_RETENTION_PRODUCTION_HANDOFF_READY`

## Final Production Model

- Logical model id: `joint_torch_v5_balanced_retention_ft_200k_20260608`
- Runtime family: `torch_joint`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`
- Continuous action dimension: `5`
- Discrete action count: `48`
- Training step: `200000`
- Parent step: `1000000`

## Production Directory And Files

Production directory:

`models/production/joint_torch_v5_balanced_retention_ft_200k_20260608`

Production files:

| File | SHA256 | Source Match |
| --- | --- | --- |
| `joint_torch_latest.pt` | `7a6e4eaac7cb9cdbf5926a544cf1cab7812940586c26e4c01cc757f3fd81b52e` | yes |
| `ppo_torch_joint_final_balanced_retention_ft_200k.pt` | `d759a2ef7a69d519beb8c4e4a50021b25546173bb73d087051320af8bbed4a6a` | yes |
| `dqn_torch_joint_final_balanced_retention_ft_200k.pt` | `a837b5152918419546ee48ede0b5d829ac97b0a657822fb009e918bffb2b3a77` | yes |
| `production_manifest.json` | manifest file present | n/a |

## Active Registry State

`models/registry/active_models.json` contains only the intended active mappings:

```json
{
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\\dqn_torch_joint_final_balanced_retention_ft_200k.pt",
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\\ppo_torch_joint_final_balanced_retention_ft_200k.pt"
}
```

`models/registry/models.jsonl` has 6 rows after the active write: the original candidate PPO/DQN rows, the active PPO/DQN rows, and prior registry history.

## Candidate And Active Registry IDs

| Role | Algorithm | Status | Registry ID |
| --- | --- | --- | --- |
| `continuous_control` | PPO | candidate | `556b047a-7178-43bc-832c-0f9b49e6ef3c` |
| `tactical_dispatch` | DQN | candidate | `9fe92e2b-3ac2-46c3-9ade-3b6e3d8d5686` |
| `continuous_control` | PPO | active | `1a5b1fd0-a1be-4818-9b5b-fab84aca3c71` |
| `tactical_dispatch` | DQN | active | `fbf7b25f-c083-4246-aac9-3f88993b43db` |

The active rows preserve `framework=torch_joint`, `runtime_loader=joint_torch`, `eval_verdict=PASS`, `hard_blocker_status=zero`, `production_promotion=false`, and `baseline_update=false`.

## Parent Checkpoint

Parent checkpoint:

`models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt`

Parent environment id:

`joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`

## Training And Fine-Tune Lineage

- Clean 1M perfclean parent: `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`, trained to `1000000` steps and used as the parent checkpoint.
- Routepremium mixed 200k lesson: the targeted route/premium/mixed fine-tune was not selected as the final candidate. The lesson was that narrow targeting risked retention/regression, so the release path moved to a balanced retention fine-tune.
- Balanced retention 200k final: `joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k`, initialized from the clean 1M perfclean parent and trained for `200000` fine-tune steps with the v5 route candidate visibility contract.

## Offline Evaluation Summary

Evaluation output:

`models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios`

Summary:

- Scenarios: `8/8 PASS`
- Episode rows: `24`
- Global hard blockers: `zero`
- Threshold verdicts: `8/8 PASS`
- Hard-blocker verdicts: `8/8 PASS`
- Final scenario verdicts: `8/8 PASS`
- Checkpoint evaluated: `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`

Accepted residual watches:

- `route_disruption_congestion.high_resilience_selected_when_not_best_count=411`
- `demand_spike_volatility.true_lateness_pressure=0.17686320800109925`
- `mixed_success_route_failure_steps=1` in `baseline_normal`, `lead_time_volatility`, and `mixed_stress`

## Runtime Evidence

Production runtime checks passed read-only:

- `load_torch_joint_policy` loaded the production `joint_torch_latest.pt` checkpoint on CPU.
- Loaded checkpoint metadata matched `checkpoint_version=torch_joint_policy_v1`, `artifact_kind=joint_final`, `global_step=200000`, contract `physical_reality_v5_route_candidate_visibility`, observation dimension `73`, continuous action dimension `5`, and discrete action count `48`.
- Deterministic zero-observation prediction returned a `float32` continuous action with length `5`, finite values, and all values bounded in `[-1, 1]`.
- Deterministic zero-observation prediction returned discrete action `42`, which is in the valid range `0..47`.
- `PolicyService.predict_joint(..., fallback_to_heuristic=False)` returned `algorithm=torch_joint`.
- The Torch joint path did not call the SB3 loader for these `.pt` artifacts (`sb3_loader_calls=0`).

## Production Promotion Evidence

Production promotion was copy-only:

- Copied exactly three `.pt` files into `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608`.
- SHA256 hashes for all three copied `.pt` files match their source checkpoint artifacts.
- `production_manifest.json` is present.
- Manifest records `human_approver=egeme`.
- Manifest records `residual_watches_accepted=true`.
- Manifest records `baseline_update=false` and `db_mutation=false`.
- Manifest records `eval_verdict=PASS` and `hard_blocker_status=zero`.

## Non-Mutated Areas

The release path intentionally left these areas untouched:

- Baselines were not overwritten or updated.
- DB rows were not inserted, updated, deleted, or manually mutated.
- Source checkpoints under `models/checkpoints` were not mutated.
- Registry rows were not changed after the intended active registry write; production promotion did not append registry rows or alter `active_models.json`.
- Production promotion copied artifacts only into the production directory and did not mutate checkpoint sources.

## Rollback Summary

If production rollback is required:

1. Restore `models/registry/active_models.json` to `{}` or to the previous approved active mapping, depending on the desired runtime fallback.
2. Do not delete the candidate or active records from `models/registry/models.jsonl`; preserve them for audit history.
3. Do not delete source checkpoint artifacts.
4. Treat the production directory as a copy-only artifact; quarantine or remove it only after explicit approval.
5. Rerun the read-only registry/runtime audit after rollback.
6. Keep baseline updates separate from runtime rollback.

## Remaining Later Work

- Decide separately whether baselines should be updated.
- Handle external deployment or service rollout separately, if applicable.
- Monitor accepted residual watches in future evaluations.
- Keep future offline evaluations watching demand-spike lateness, high-resilience route preference under disruption, and mixed-success route-failure steps.

## Final Classification

`TORCH_JOINT_V5_BALANCED_RETENTION_PRODUCTION_HANDOFF_READY`
