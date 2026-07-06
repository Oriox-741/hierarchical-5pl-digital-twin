# Chapter 3 Hyperparameter Truth Table

Generated UTC: `2026-06-23T00:16:34.548050+00:00`

Values come from final 1M config unless noted; teacher-distillation values come from the initial 250k config.

## PPO Hyperparameters

| Field | Exact value | Source | Status/limitation |
| --- | --- | --- | --- |
| Input dimension | `73` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:13 $.shared_global_parameters.observation_dim | explicit config value |
| Output dimension | `5` | src/act/action_projector.py:14 | explicit source constant |
| Feature extractor hidden layers | `[256, 256]` | src/think/joint_policies.py:67 | source default |
| Feature extractor hidden dim | `256` | src/think/joint_policies.py:68 | source default |
| Actor hidden layers | `[256]` | src/think/joint_policies.py:110 | source default |
| Critic hidden layers | `[256]` | src/think/joint_policies.py:169 | source default |
| Activation functions | `LayerNorm + SiLU in feature extractor; SiLU in MLP` | src/think/joint_policies.py:84 | explicit source modules |
| Policy distribution/output transform | `Normal(mean, exp(log_std)) with tanh-squashed action; deterministic uses mean before tanh` | src/think/joint_policies.py:137 | explicit source behavior |
| Learning rate | `0.0003` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:24 $.ppo_hyperparameters.learning_rate | explicit config value |
| Discount factor gamma | `0.99` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:25 $.ppo_hyperparameters.gamma | explicit config value |
| GAE lambda | `0.95` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:26 $.ppo_hyperparameters.gae_lambda | explicit config value |
| PPO clip range | `0.2` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:48 $.torch_joint_training.ppo_clip_range | explicit config value |
| PPO epochs | `4` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:46 $.torch_joint_training.ppo_epochs | explicit config value |
| PPO minibatch size | `256` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:47 $.torch_joint_training.ppo_minibatch_size | explicit config value |
| Rollout length | `2048` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:45 $.torch_joint_training.rollout_steps | explicit config value |
| Entropy coefficient | `0.01` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:50 $.torch_joint_training.ppo_entropy_coef | explicit config value |
| Value coefficient | `0.5` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:49 $.torch_joint_training.ppo_value_coef | explicit config value |
| Max gradient norm | `0.5` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:51 $.torch_joint_training.ppo_max_grad_norm | explicit config value |
| Optimizer | `torch.optim.Adam` | src/learn/train_joint_torch.py:637 | explicit source constructor |
| Learning-rate schedule | NOT_FOUND_IN_REPOSITORY | NOT_FOUND_IN_REPOSITORY | No schedule field or scheduler use found for PPO. |
| Action bounds | `continuous output bounded to [-1, 1] by tanh and runtime validation` | src/think/joint_policies.py:140 | source behavior |
| Deterministic inference | `PPO actor uses pre_tanh=mean when deterministic=True; runtime calls deterministic=True by default` | src/think/joint_policies.py:137 | source behavior |

## Hierarchical DQN Hyperparameters

| Field | Exact value | Source | Status/limitation |
| --- | --- | --- | --- |
| Input dimension | `73` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:13 $.shared_global_parameters.observation_dim | explicit config value |
| External discrete action count | `48` | src/act/discrete_action_mapper.py:40 | derived from enum cardinalities |
| Shared trunk layer sizes | `[256, 256]` | src/think/joint_policies.py:230 | source default |
| Dispatch head structure | `Linear(256, len(DispatchDecision)=2)` | src/think/joint_policies.py:243 | explicit source |
| Route head structure | `Linear(256, len(RouteDecision)=3)` | src/think/joint_policies.py:244 | explicit source |
| Fleet/mode head structure | `Linear(256, len(ModeDecision)=2)` | src/think/joint_policies.py:245 | explicit source |
| Reorder head structure | `Linear(256, len(ReorderDecision)=4)` | src/think/joint_policies.py:246 | explicit source |
| Activation functions | `SiLU in _mlp trunk` | src/think/joint_policies.py:468 | explicit source |
| Value-combination rule | `dispatch + reorder + dispatch_mask(route + mode)` | src/think/joint_policies.py:261 | explicit source formula |
| Learning rate | `0.0001` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:35 $.dqn_hyperparameters.learning_rate | explicit config value |
| Discount factor gamma | `0.99` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:36 $.dqn_hyperparameters.gamma | explicit config value |
| Replay-buffer capacity | `500000` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:52 $.torch_joint_training.dqn_replay_buffer_size | explicit config value |
| Minimum replay size / learning starts | `10000` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:53 $.torch_joint_training.dqn_learning_starts | explicit config value |
| Batch size | `256` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:54 $.torch_joint_training.dqn_batch_size | explicit config value |
| Epsilon start | `1.0` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:58 $.torch_joint_training.dqn_exploration_initial_eps | explicit config value |
| Epsilon end | `0.05` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:59 $.torch_joint_training.dqn_exploration_final_eps | explicit config value |
| Epsilon decay fraction | `0.25` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:60 $.torch_joint_training.dqn_exploration_fraction | explicit config value |
| Epsilon decay formula | `initial + ((final - initial) * progress), progress=step/(total_timesteps*fraction) clipped to [0,1]` | src/learn/train_joint_torch.py:2057 | explicit source formula |
| Target-network update interval | `1000` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:56 $.torch_joint_training.dqn_target_update_interval | explicit config value |
| Target-network tau | `1.0` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:57 $.torch_joint_training.dqn_tau | explicit config value |
| Target-network mechanism | `tau=1.0 hard_update_dqn_target else soft_update_dqn_target on interval` | src/learn/train_joint_torch.py:1346 | explicit source behavior |
| Optimizer | `torch.optim.Adam` | src/learn/train_joint_torch.py:645 | explicit source constructor |
| Gradient clipping | `clip_grad_norm_ max_norm=10.0 error_if_nonfinite=True` | src/learn/train_joint_torch.py:1335 | explicit source behavior |
| DQN gradient steps per rollout | `2048` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:55 $.torch_joint_training.dqn_gradient_steps_per_rollout | explicit config value |
| Head-to-action encoding formula | `(((dispatch * 3) + route) * 2 + mode) * 4 + reorder` | src/act/discrete_action_mapper.py:87 | explicit source encode formula |

## Teacher Distillation and Initialization Settings

| Field | Exact value | Source | Status/limitation |
| --- | --- | --- | --- |
| Teacher model identity | `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json:115 $.hierarchical_initialization.teacher_checkpoint | explicit 250k config |
| Initialization method | `flat_teacher_distillation_v1` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json:114 $.hierarchical_initialization.method | explicit 250k config |
| Observation sample count | `4096` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json:116 $.hierarchical_initialization.observation_sample_count | explicit 250k config |
| Distillation update count | `512` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json:117 $.hierarchical_initialization.distillation_steps | explicit 250k config |
| Batch size | `128` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json:118 $.hierarchical_initialization.batch_size | explicit 250k config |
| Optimizer and learning rate | `Adam lr=0.0001` | src/learn/hierarchical_dqn_initialization.py:227 | optimizer source plus config LR |
| Loss functions | `smooth_l1 Q loss + cross entropy on teacher argmax; soft bounded by max_loss` | src/learn/hierarchical_dqn_initialization.py:197 | explicit source |
| Loss weights | `{"action_ce_weight": 1.0, "max_loss": 1.0, "q_loss_weight": 1.0, "temperature": 1.0}` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json:121 $.hierarchical_initialization.* | explicit config values |
| Observation bank construction | `torch.rand(sample_count, 73) with seed; first rows set to 0, 1, 0.5` | src/learn/hierarchical_dqn_initialization.py:157 | explicit source |
| PPO layers copied from teacher | `ppo_feature_extractor, ppo_actor, ppo_critic` | src/learn/hierarchical_dqn_initialization.py:361 | explicit source |
| DQN trunk transfer | `True` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json:125 $.hierarchical_initialization.transfer_dqn_trunk | explicit config |
| DQN trunk copy rule | `copy compatible flat DQN trunk tensors with matching key and shape into hierarchical trunk` | src/learn/hierarchical_dqn_initialization.py:340 | explicit source |
| Independent head initialization | NOT_FOUND_IN_REPOSITORY | NOT_FOUND_IN_REPOSITORY | No explicit independent-init policy beyond normal module initialization was found. |

## Shared Training Settings

| Field | Exact value | Source | Status/limitation |
| --- | --- | --- | --- |
| Seed | `42` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:8 $.shared_global_parameters.seed | explicit config value |
| Device | `auto` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:70 $.torch_joint_training.device | explicit config value |
| Max steps per episode | `288` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:11 $.shared_global_parameters.max_steps | explicit config value |
| Decision interval seconds | `300` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:12 $.shared_global_parameters.decision_interval_seconds | explicit config value |
| Checkpoint cadence | `100000` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:68 $.torch_joint_training.checkpoint_interval_steps | explicit config value |
| Metrics interval | `4096` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:69 $.torch_joint_training.metrics_interval_steps | explicit config value |
| Trace flush interval | `5000` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:67 $.torch_joint_training.trace_flush_interval | explicit config value |
| Trace sample interval | `100` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:16 $.shared_global_parameters.trace_sample_interval | explicit config value |
| Skip registry required | `True` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:125 $.manual_run_safety.requires_skip_registry | explicit config value |
| Disable trace logging required | `True` | configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json:126 $.manual_run_safety.requires_disable_trace_logging | explicit config value |
| Number of environments | NOT_FOUND_IN_REPOSITORY | NOT_FOUND_IN_REPOSITORY | No vectorized environment count field found; trainer constructs one FivePLDigitalTwinEnv instance. |
