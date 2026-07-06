# Clean Rewardfix Perfclean 1M Autopilot Status

- Updated: 2026-06-08T10:33:43.286761+03:00
- Status: CLEAN_REWARDFIX_PERFCLEAN_1M_TRAINING_COMPLETE_READY_FOR_OFFLINE_EVAL_READINESS

## Milestones
- PREFLIGHT_STARTED: 2026-06-08T08:14:06.2714353+03:00

## Notes
- Bounded autopilot mode active.
- Allowed mutation scope: training output dir and this status file only.

## PREFLIGHT_PASSED
- Timestamp: 2026-06-08T08:15:33.6518492+03:00
- No train_joint_torch/evaluate_real_world_scenarios process detected.
- Output dir absent: models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608
- Config exists: configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json
- Environment id: joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean
- Contract: physical_reality_v5_route_candidate_visibility
- Observation/action: 73/48
- Resume/checkpoint-load path: none detected; only checkpoint_interval_steps and allow_resume_contract fields present.
- Command flags confirmed: --skip-registry and --disable-trace-logging
- Quarantined after_rewardfix output dir not referenced.

## TRAINING_STARTED
- Timestamp: 2026-06-08T08:15:33.6518492+03:00
- Command: python -m src.learn.train_joint_torch --config configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json --output-dir models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608 --final-ppo-path models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/ppo_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt --final-dqn-path models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/dqn_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt --device cpu --torch-num-threads 2 --torch-num-interop-threads 1 --skip-registry --disable-trace-logging


## TRAINING_COMPLETED
- Timestamp: 2026-06-08T10:29:59.5824558+03:00
- Exit status: 0
- Elapsed seconds: 8053.9183143


## POST_TRAINING_AUDIT_STARTED
- Timestamp: 2026-06-08T10:30:33.8827226+03:00
- Mode: read-only audit of training artifacts and process state.

## POST_TRAINING_AUDIT_COMPLETED
- Timestamp: 2026-06-08T10:33:43.286761+03:00
- Training exit status: 0
- Training elapsed seconds: 8053.9183143
- Output dir exists: True
- Metrics file exists: True
- Metrics row count: 245
- Highest metrics global_step: 1000000
- Last metrics global_step: 1000000
- Checkpoint/metrics step consistency: True
- Contract: physical_reality_v5_route_candidate_visibility
- Observation/action metadata: 73/48
- Curriculum stages observed: demand_spike, high_holding_cost, lead_time_delay, mixed_stress, normal_v4, premium_sla, route_disruption, vehicle_scarcity

### Expected Checkpoints
- models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/dqn_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt: exists=True, size=5019963, global_step=1000000, artifact_kind=dqn_final
- models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt: exists=True, size=5007035, global_step=1000000, artifact_kind=joint_final
- models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/ppo_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt: exists=True, size=5019963, global_step=1000000, artifact_kind=ppo_final

### Final/FPS Telemetry From Last Metrics Row
- fps_total: 124.22187460449035
- fps_window: 50.734159399377646
- torch_num_threads: 2.0
- torch_num_interop_threads: 1.0
- seconds_env_step_window: 2.0846939006005414
- seconds_dqn_update_window: 8.238832800008822
- dqn_total_avg_ms_per_grad_step_window: 4.022867578129308
- dqn_forward_avg_ms_per_grad_step_window: 1.3394544435527678
- dqn_backward_avg_ms_per_grad_step_window: 1.6436980468199636
- dqn_optimizer_avg_ms_per_grad_step_window: 0.6631058105028842
- dqn_validation_avg_ms_per_grad_step_window: 0.042613916122036244
- service_level: 0.9293911312070058
- blocked_action_rate: 0.0
- projected_action_rate: 0.011092

### NaN/Inf Audit
- NaN/Inf values detected in metrics: 0

### Hard-Blocker/Premium Guard Telemetry
- Training metrics hard-blocker/premium guard fields present in last row: none present in training metrics
- Note: hard-blocker/premium guard scenario counters are primarily offline-evaluation telemetry; offline evaluation was not run.

### Registry/Baseline/Production Audit
- models/registry/active_models.json: exists=True, last_write=2026-05-18T07:18:09.271413, size=3
- models/registry/models.jsonl: exists=True, last_write=2026-05-18T07:18:18.442251, size=830
- models/baselines: exists=True, last_write=2026-05-25T03:11:15.036299, size=dir
- models/production: exists=False
- Registry update command path: skipped by --skip-registry.
- Baselines touched by this task: no evidence from inspected paths; no baseline commands were run.
- Production touched by this task: no evidence; models/production is absent.

### Trace/DB Audit
- Trace logging flag: --disable-trace-logging used.
- Trainer path inspected: disabled trace logging uses JointTraceRecorder.disabled and does not create DatabasePool for trace persistence.
- Trace/db/log files under new output dir: none found by extension/name scan.
- Manual DB mutation performed: no.

### Process Audit
- train_joint_torch/evaluate_real_world_scenarios process after training: none detected.

## FINAL_CLASSIFICATION
- CLEAN_REWARDFIX_PERFCLEAN_1M_TRAINING_COMPLETE_READY_FOR_OFFLINE_EVAL_READINESS
- Stop point: post-training audit only. Offline evaluation and registry update were not run.
