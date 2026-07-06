# Chapter 3 Training and Resume Truth Table

Generated UTC: `2026-06-23T00:16:34.600186+00:00`

This package did not run training. The table records what the final configs and trainer source say.

## Ladder Stages

| Stage | Target/global timesteps | Seed | Device | Output dir | Initialization/resume mode | Source | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 250k | `250000` | `42` | `auto` | `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_250k_20260611` | `init_from_joint_checkpoint` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json $.shared_global_parameters/$.manual_run_safety/$.ladder_continuation | current final hierarchical ladder config |
| 500k | `500000` | `42` | `auto` | `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_500k_20260611` | `exact_resume_from_passed_checkpoint` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_500k_20260611.json $.shared_global_parameters/$.manual_run_safety/$.ladder_continuation | current final hierarchical ladder config |
| 1M | `1000000` | `42` | `auto` | `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611` | `exact_resume_from_passed_checkpoint` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json $.shared_global_parameters/$.manual_run_safety/$.ladder_continuation | current final hierarchical ladder config |

## Exact-Resume Contents and Checks

| Resume/checkpoint fact | Exact value/rule | Source | Status |
| --- | --- | --- | --- |
| Checkpoint payload includes optimizer states | `ppo and dqn optimizer_state_dicts` | src/learn/train_joint_torch.py:1553 | explicit source |
| DQN replay payload key | `DQN_REPLAY_STATE_KEY = "dqn_replay_buffer_state"` | src/learn/train_joint_torch.py:75 | explicit source |
| DQN replay state persisted | `payload[DQN_REPLAY_STATE_KEY] = dqn_replay.state_dict()` | src/learn/train_joint_torch.py:1569 | explicit source |
| RNG state persisted | `payload[RNG_STATE_KEY] = capture_rng_state()` | src/learn/train_joint_torch.py:1572 | explicit source |
| Resume metadata exact_resume_capable | `replay_state_included and rng_state_included` | src/learn/train_joint_torch.py:1625 | explicit source |
| Resume restores optimizer states | `ppo_optimizer.load_state_dict and dqn_optimizer.load_state_dict` | src/learn/train_joint_torch.py:1648 | explicit source |
| Resume restores global step | `state.global_step = checkpoint.global_step` | src/learn/train_joint_torch.py:1654 | explicit source |
| Resume restores replay | `dqn_replay.load_state_dict(replay_state) else reject unless allow_empty_replay_resume` | src/learn/train_joint_torch.py:1661 | explicit source |
| Missing replay rejection | `exact continuation requires 'dqn_replay_buffer_state'` | src/learn/train_joint_torch.py:1671 | explicit source |
| Resume restores RNG | `restore_rng_state(rng_state) else reject unless allow_empty_replay_resume` | src/learn/train_joint_torch.py:1676 | explicit source |
| 1M parent checkpoint | `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_500k_20260611/joint_torch_latest.pt` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:113 $.ladder_continuation.parent_checkpoint | explicit config |
| Legacy warm resume allowed | `False` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:122 $.ladder_continuation.legacy_warm_resume_allowed | explicit config |

## Curriculum Schedule

| Stage | Enabled | Initial stage | Baseline-anchor probability | Schedule entries | Schedule step sum | Source | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 250k | `True` | `normal_v4` | `0.12` | `15` | `250000` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json:160 $.curriculum.stage_schedule | fixed stage schedule; dynamic transition criteria not found |
| 500k | `True` | `normal_v4` | `0.12` | `30` | `500000` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_500k_20260611.json:153 $.curriculum.stage_schedule | fixed stage schedule; dynamic transition criteria not found |
| 1M | `True` | `normal_v4` | `0.12` | `60` | `1000000` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:154 $.curriculum.stage_schedule | fixed stage schedule; dynamic transition criteria not found |

## Registry and Run-Safety Settings

| Setting | Exact value | Source | Status/limitation |
| --- | --- | --- | --- |
| --skip-registry | `True` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:125 $.manual_run_safety.requires_skip_registry | explicit config |
| --disable-trace-logging | `True` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:126 $.manual_run_safety.requires_disable_trace_logging | explicit config |
| Registry mutation setting in production manifest | `False` | models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json $.registry_mutation | explicit manifest |
| Config schema file | NOT_FOUND_IN_REPOSITORY | NOT_FOUND_IN_REPOSITORY | No dedicated JSON schema file found for training configs. |
