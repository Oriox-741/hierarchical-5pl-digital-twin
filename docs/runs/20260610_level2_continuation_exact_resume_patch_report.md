# Level 2 Continuation Exact Resume Patch Report

Date: 2026-06-10

## Decision

Implemented the exact-resume architecture fix for future Level 2 continuation rungs.

Classification: `EXACT_RESUME_REPLAY_STATE_FIX_READY_FOR_REVIEW`

Independent review verdict: `PATCH_APPROVED_FOR_NEXT_GATE`

## Architecture Finding

The V2.4 500k run was launched with `--resume` from the passing V2.4 250k checkpoint, but the trainer only restored:

- PPO/DQN model state
- PPO/DQN optimizer state
- `TrainingState.global_step`
- episode/update/loss counters stored in checkpoint `extra`

It did not restore:

- DQN replay-buffer contents
- DQN replay cursor/size/total-added state
- Python/NumPy/Torch RNG state

Artifact metadata confirms the gap:

- `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610/joint_torch_latest.pt`
  - `global_step`: `250000`
  - optimizer keys: `ppo`, `dqn`
  - replay keys: none
  - RNG keys: none
- `models/checkpoints/joint_torch_v5_prod_stability_v2_4_500k_20260610/joint_torch_latest.pt`
  - `global_step`: `500000`
  - optimizer keys: `ppo`, `dqn`
  - replay keys: none
  - RNG keys: none

The trainer also printed:

```text
resume=warm_start replay_buffer=empty dqn_learning_starts=10000 metrics_state=reinitialized
```

Therefore V2.4 500k was not an exact continuation. It was a model/optimizer/global-step warm start with an empty DQN replay buffer and fresh RNG state.

## Required Questions

1. Does `--resume` restore DQN replay buffer?
   - Before patch: no.
   - After patch: yes, when the resume checkpoint contains `dqn_replay_buffer_state`.

2. Does `--resume` restore PPO/DQN optimizer state?
   - Yes. `optimizer_state_dicts["ppo"]` and `optimizer_state_dicts["dqn"]` are loaded when present.

3. Does `--resume` restore RNG states?
   - Before patch: no.
   - After patch: yes, when the resume checkpoint contains `rng_state` with Python, NumPy, Torch CPU, and optional Torch CUDA RNG state.

4. Does `--resume` restore curriculum/global-step schedule position?
   - It restores `global_step`; curriculum sampling receives `global_step`, so schedule position resumes from the checkpoint step.
   - V2.4 500k also repeated the 250k curriculum schedule, and exploration epsilon was already at final value by 250k.

5. Does `--init-from-joint-checkpoint` intentionally start with empty replay and fresh run state?
   - Yes. It remains weights-only/fresh-state by design. The patch adds metadata `replay_restore_status=fresh_replay_by_design`.

6. Was V2.4 500k launched as exact resume or weights/optimizer continuation with empty replay?
   - It was weights/optimizer/global-step continuation with empty replay and fresh RNG, not exact resume.

7. Is `replay_buffer=empty` a confirmed architectural bug for gated 250k->500k continuation?
   - Yes as an exact-resume contract bug. The V2.4 evidence strongly supports it as the primary continuation-instability explanation, but the existing V2.4 250k checkpoint cannot be retrofitted because it never stored replay/RNG state.

8. What exact state must be persisted for stable DQN continuation?
   - DQN replay tensors: observations, actions, rewards, next observations, dones.
   - Replay metadata: capacity, learning_starts, observation_dim, action_count, cursor position, size, total_added.
   - PPO/DQN optimizer state.
   - Trainer counters: global_step, episode_count, dqn_update_count, latest losses.
   - RNG state: Python, NumPy, Torch CPU, optional Torch CUDA.

## Patch Summary

Changed files:

- `src/learn/joint_buffers.py`
  - Added DQN replay `state_dict`, `load_state_dict`, and `state_summary`.
  - Persisted valid ring-storage slots plus replay cursor/size/total-added metadata.
  - Validated replay version, shape, capacity, learning_starts, observation dimension, action count, finite tensors, and action bounds on restore.

- `src/learn/train_joint_torch.py`
  - Added checkpoint keys:
    - `dqn_replay_buffer_state`
    - `rng_state`
    - `resume_state`
  - Added RNG capture/restore for Python, NumPy, Torch CPU, and optional Torch CUDA.
  - Added `--allow-empty-replay-resume` for explicit diagnostic warm starts from legacy checkpoints.
  - Normal `--resume` now rejects checkpoints missing replay/RNG state.
  - `joint_torch_latest.pt` now carries exact resume state.
  - PPO/DQN eval artifacts remain lightweight and are marked `exact_resume_capable=false`.
  - `--init-from-joint-checkpoint` remains weights-only/fresh replay/fresh optimizer state.

- `tests/learn/test_joint_reward_blending.py`
  - Added replay ring-state round-trip coverage.

- `tests/learn/test_train_joint_curriculum.py`
  - Added checkpoint payload replay/RNG metadata coverage.
  - Added periodic save coverage proving `joint_torch_latest.pt` gets replay/RNG and PPO eval artifacts do not.
  - Added exact resume restore coverage for replay data, cursor/size/total-added, RNG, and global_step.
  - Added unsafe missing-replay rejection coverage with explicit warm-start override.
  - Added init-from metadata coverage proving fresh replay remains intentional.

## Verification

No training was run.
No offline eval was run.
No config for V2.5 was created.

Commands run:

```powershell
python -m unittest tests.learn.test_joint_reward_blending.DQNReplayBufferStorageTests.test_replay_state_dict_round_trips_ring_storage_and_cursor tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_checkpoint_payload_includes_replay_state_rng_state_and_resume_metadata tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_load_training_checkpoint_restores_replay_rng_and_global_step tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_load_training_checkpoint_rejects_missing_replay_unless_explicitly_allowed
```

Result: PASS, 4 tests.

```powershell
python -m unittest tests.learn.test_joint_reward_blending tests.learn.test_train_joint_curriculum
```

Result: PASS, 45 tests.

```powershell
python -m py_compile src\learn\joint_buffers.py src\learn\train_joint_torch.py tests\learn\test_joint_reward_blending.py tests\learn\test_train_joint_curriculum.py
```

Result: PASS.

```powershell
python -m unittest discover tests\learn
```

Result: one unrelated artifact-state failure. `test_curriculum_config.CurriculumConfigTests.test_prod_stability_v2_4_250k_config_loads_and_is_manual_run_safe` expects the V2.4 250k output directory not to exist, but this workspace already contains completed V2.4 250k artifacts. The artifacts were not deleted or moved.

```powershell
python -m unittest tests.learn.test_cold_start_contract tests.learn.test_joint_reward_blending tests.learn.test_model_registry tests.learn.test_pretrain_mdp_sanity_check tests.learn.test_promote_torch_joint_production tests.learn.test_register_torch_joint_candidate tests.learn.test_trace_sampling_contract tests.learn.test_train_joint_curriculum
```

Result: PASS, 103 tests.

## Independent Review

One GPT-5.5 read-only reviewer inspected the patch and returned exactly:

```text
PATCH_APPROVED_FOR_NEXT_GATE
```

## Protected-Path Proof

No registry, production, baseline, DB, source production checkpoint, or existing checkpoint mutation was performed.

Protected hashes after patch:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Source production checkpoint SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles:

- `models/production`: 4 files, 15040339 bytes, max_mtime_ns `1780972731823548200`
- `models/baselines`: 2692 files, 7392576274 bytes, max_mtime_ns `1779666921467665500`
- `db`: 7 files, 28330 bytes, max_mtime_ns `1779200011801591300`

## Operational Consequence

The exact-resume mechanism is fixed for future checkpoints produced by the patched trainer.

The existing V2.4 250k checkpoint is still not exact-resume capable because it lacks replay/RNG state. It should not be used for a new exact 500k continuation without `--allow-empty-replay-resume`, and that override should be considered diagnostic warm-start mode, not a gated ladder continuation.

Next safe gate path:

- Produce the next approved short rung with the patched trainer so `joint_torch_latest.pt` includes replay/RNG state.
- Only then use normal `--resume` for an exact replay-state continuation.
- Do not use failed 300k/400k/500k checkpoints as parents.
