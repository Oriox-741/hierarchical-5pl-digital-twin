# Level 2 Teacher-Retention Ladder Report

Date: 2026-06-11

## Result

Final classification: `TEACHER_RETENTION_250K_GATE_FAILED`

Architecture consequence: teacher retention did not preserve the passing V2.5 250k behavior even at the fresh 250k rung. The hybrid decision path now points to the documented hierarchical/decomposed action architecture fallback before any further 500k/1M continuation attempt.

No 500k config, 500k training run, 1M config, or 1M training run was created.

## Architecture Decision Used

Decision source: [[docs/runs/20260610_level2_teacher_retention_vs_hierarchical_architecture_decision]]

Decision: `RECOMMEND_HYBRID_TEACHER_THEN_HIERARCHICAL`

Applied path: teacher-retention / behavior anchoring was implemented first. Hierarchical action redesign remains unimplemented and is now the recommended next architecture track because the teacher-retention 250k candidate failed the long-run gate.

## Reviewer Verdict

One read-only GPT-5.5 reviewer was used after the code patch.

Verdict: `PATCH_APPROVED_FOR_NEXT_GATE`

## Config Created

- `configs/training_joint_curriculum_v5_prod_teacher_retention_250k_20260611.json`

Safety properties:

- Parent/init source is production only:
  `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`
- Passing V2.5 250k is used only as read-only teacher evidence.
- Failed V2.5 500k and older failed V-cycle checkpoints are not used as parents.
- Contract remains `physical_reality_v5_route_candidate_visibility`.
- Observation dim remains `73`.
- Discrete action count remains `48`.
- Registry updates are disabled.
- Trace logging is disabled.

## Tests Run Before Training

- `python -m unittest tests.learn.test_train_joint_curriculum -v`
  - PASS, 43 tests.
- `python -m unittest tests.learn.test_joint_reward_blending -v`
  - PASS, 10 tests.
- `python -m unittest tests.eval.test_long_run_gate -v`
  - PASS, 13 tests.
- `python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_teacher_retention_250k_config_loads_and_is_manual_run_safe -v`
  - PASS, 1 test.
- `python -m py_compile src\learn\teacher_retention.py src\learn\train_joint_torch.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py`
  - PASS.

Known non-blocking test issue:

- Full `python -m unittest tests.learn.test_curriculum_config -v` has one unrelated stale-artifact failure because this workspace already contains the historical V2.4 250k output directory that the old safety test expects to be absent. No artifact was deleted or moved.

Post-run test maintenance:

- The new teacher-retention config test was updated after the rung to follow the existing artifact-tolerant pattern used by completed config tests: before training, the fresh-dir preflight was performed with explicit shell checks; after training/eval, the unit test validates that any existing output dir contains the expected `joint_torch_latest.pt` and any existing eval dir contains `scenario_summary.json`.
- Post-run `python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_teacher_retention_250k_config_loads_and_is_manual_run_safe -v`
  - PASS, 1 test.
- Post-run `python -m unittest tests.learn.test_train_joint_curriculum -v`
  - PASS, 43 tests.
- Post-run `python -m py_compile src\learn\teacher_retention.py src\learn\train_joint_torch.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py`
  - PASS.

## 250k Training

Run dir:

- `models/checkpoints/joint_torch_v5_prod_teacher_retention_250k_20260611`

Command shape:

- `python -m src.learn.train_joint_torch`
- `--config configs\training_joint_curriculum_v5_prod_teacher_retention_250k_20260611.json`
- `--output-dir models\checkpoints\joint_torch_v5_prod_teacher_retention_250k_20260611`
- `--init-from-joint-checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt`
- `--enable-teacher-retention`
- `--teacher-production-checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt`
- `--teacher-candidate-checkpoint models\checkpoints\joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610\joint_torch_latest.pt`
- `--skip-registry`
- `--disable-trace-logging`
- torch threads `2/1`

Exit code: `0`

Startup resume mode:

- `init_from_joint_checkpoint=true`
- `replay_buffer=empty`
- `optimizers=fresh`
- `global_step=0`
- `parent_global_step=200000`

This is expected for a fresh 250k candidate initialized from the production parent. It was not a resume rung.

## 250k Exact-Resume Capability Audit

Checkpoint:

- `models/checkpoints/joint_torch_v5_prod_teacher_retention_250k_20260611/joint_torch_latest.pt`

Audit result:

- `artifact_kind`: `joint_final`
- `checkpoint_version`: `torch_joint_policy_v1`
- `global_step`: `250000`
- `mdp_contract`: `physical_reality_v5_route_candidate_visibility`
- `observation_dim`: `73`
- `discrete_action_count`: `48`
- DQN replay state present: yes
- RNG state present: yes
- Teacher-retention state bank present: yes
- `resume_state.exact_resume_capable`: yes
- Replay size: `250000`
- Replay capacity: `500000`
- Replay cursor/position: `250000`
- Replay total added: `250000`
- Teacher bank size: `4096`
- Teacher bank scenario counts:
  - baseline: `512`
  - demand_spike: `512`
  - high_holding: `512`
  - lead_time: `512`
  - mixed_stress: `512`
  - premium_sla: `512`
  - route_disruption: `512`
  - vehicle_scarcity: `512`

## Teacher-Retention Telemetry

Last training metrics row:

- `teacher_retention_updates_window`: `2048`
- `teacher_dqn_kl_window`: `0.020200035559740925`
- `teacher_dqn_ce_window`: `0.001618169221728749`
- `teacher_dqn_action_match_rate_window`: `0.39234542944315365`
- `teacher_retention_loss_window`: `0.0004341496552271451`
- `gate_critical_action_drift_window`: `0.021202087538071623`
- `bad_pocket_margin_by_action_window`:
  - action 32: `0.34806246657003026`
  - action 33: `0.3903441533762759`
  - action 41: `0.37917540482226286`
  - action 45: `0.4267239720264797`
  - action 46: `1.0025878553194616`

Interpretation:

- The retention path was active and populated all eight scenario banks.
- The loss remained small and bounded.
- The student was not tightly matching the selected teachers (`action_match_rate` about `0.392`).
- The bounded teacher loss was too weak or too indirect to preserve the passing 250k behavior, and stronger flat-action imitation now risks freezing or amplifying teacher-specific weaknesses rather than solving the decomposition problem.

## Offline Eval

Eval dir:

- `models/eval/joint_torch_v5_prod_teacher_retention_250k_20260611_offline_scenarios`

Artifacts:

- `scenario_summary.json`
- `scenario_summary.csv`
- `episode_metrics.jsonl`
- `real_world_evaluation_report.md`

## Long-Run Gate

Gate command:

- `python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_prod_teacher_retention_250k_20260611_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label teacher_retention_250k_20260611`

Exit code: `1`

Decision: `FAIL`

Fatal failures:

- `baseline_normal`: service `0.911 < 0.930`
- `baseline_normal`: true lateness pressure `0.119 > 0.050`
- `baseline_normal`: service regressed from production `0.938` to `0.911`
- `high_holding_cost`: true lateness pressure `0.098 > 0.080`
- `lead_time_volatility`: service `0.589 < 0.900`
- `lead_time_volatility`: true lateness pressure `0.369 > 0.100`
- `route_disruption_congestion`: service `0.706 < 0.880`

## Scenario Table

| Scenario | Production svc/late/dispatch/success | V2.5 250k svc/late/dispatch/success | Teacher 250k svc/late/dispatch/success | Teacher verdict |
|---|---:|---:|---:|---|
| baseline_normal | 0.938 / 0.000 / 0.487 / 1.000 | 0.961 / 0.000 / 0.446 / 1.000 | 0.911 / 0.119 / 0.398 / 1.000 | FAIL |
| demand_spike_volatility | 0.843 / 0.177 / 0.728 / 1.000 | 0.948 / 0.002 / 0.956 / 1.000 | 0.956 / 0.000 / 0.988 / 1.000 | PASS |
| high_holding_cost | 0.943 / 0.000 / 0.387 / 1.000 | 0.990 / 0.000 / 0.392 / 1.000 | 0.948 / 0.098 / 0.349 / 0.998 | FAIL |
| lead_time_volatility | 0.936 / 0.000 / 0.455 / 1.000 | 0.962 / 0.006 / 0.412 / 0.998 | 0.589 / 0.369 / 0.316 / 0.999 | FAIL |
| mixed_stress | 0.951 / 0.000 / 0.911 / 0.996 | 0.936 / 0.025 / 0.934 / 1.000 | 0.931 / 0.147 / 0.816 / 1.000 | PASS |
| premium_sla_pressure | 0.980 / 0.000 / 0.427 / 0.769 | 0.988 / 0.000 / 0.365 / 0.999 | 0.958 / 0.002 / 0.378 / 0.999 | PASS |
| route_disruption_congestion | 0.929 / 0.000 / 0.575 / 0.822 | 0.946 / 0.001 / 0.482 / 1.000 | 0.706 / 0.080 / 0.437 / 0.961 | FAIL |
| vehicle_scarcity_capacity_shock | 0.914 / 0.000 / 0.731 / 1.000 | 0.942 / 0.000 / 0.764 / 1.000 | 0.966 / 0.010 / 0.751 / 1.000 | PASS |

## Action Evidence

Teacher-retention did not simply repeat the V2.5 500k action-33 pocket. It shifted behavior into a different failure mode:

- Baseline moved toward action 16 concentration:
  - V2.5 250k action 16: `0.181`
  - teacher 250k action 16: `0.396`
  - result: service fell and lateness rose.
- High-holding moved toward action 16 concentration:
  - V2.5 250k action 16: `0.247`
  - teacher 250k action 16: `0.478`
  - result: lateness rose above threshold.
- Lead-time lost useful dispatch coverage:
  - V2.5 250k dispatch rate: `0.412`
  - teacher 250k dispatch rate: `0.316`
  - result: service collapsed to `0.589`.
- Route-disruption retained a bad composite-action surface:
  - teacher action 33: `94` attempts, `3` successes, `91` failed no-ops
  - teacher action 41: `593` attempts, `574` successes, `19` failed no-ops
  - route service fell to `0.706`.
- Mixed-stress route ranking degraded:
  - V2.5 250k selected route rank 3 count: `1152`
  - teacher 250k selected route rank 3 count: `3071`
  - selected route margin-to-best mean rose to `0.359`.

Root-cause interpretation:

- The flat 48-action DQN still treats dispatch, route, fleet, and reorder decisions as one entangled categorical choice.
- The teacher-retention loss can reduce some known bad pockets but cannot reliably preserve the useful sub-decisions across scenarios.
- Strengthening the same flat-action anchor is likely to trade one pocket for another or freeze teacher weaknesses.
- The evidence now supports moving to a factorized/hierarchical action architecture while preserving external 0..47 semantics.

## Protected No-Mutation Proof

Protected hashes after the run/eval/gate:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt` SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles after the run/eval/gate:

- `models/production`: 4 files, 15040339 bytes, max write UTC `2026-06-09 02:38:51Z`
- `models/baselines`: 2692 files, 7392576274 bytes, max write UTC `2026-05-24 23:55:21Z`
- `db`: 7 files, 28330 bytes, max write UTC `2026-05-19 14:13:31Z`

No registry, production, baseline, DB, source production checkpoint, or existing checkpoint mutation was required or observed.

## Stop Condition

The teacher-retention 250k candidate failed the long-run gate. Per the ladder constraints:

- no 500k config was created,
- no 500k training was started,
- no 1M config was created,
- no 1M training was started,
- no registry or production promotion action was taken.

Recommended next step: open a new architecture task for hierarchical/decomposed DQN action heads that preserve the external 48-action API.
