# Level 2 Teacher-Retention Patch Report

Date: 2026-06-11

## Architecture Decision Used

Decision source: [[docs/runs/20260610_level2_teacher_retention_vs_hierarchical_architecture_decision]]

Decision: `RECOMMEND_HYBRID_TEACHER_THEN_HIERARCHICAL`

Implementation status: teacher-retention / behavior anchoring patch implemented for review. Hierarchical/decomposed action redesign remains documented fallback and was not implemented.

## Scope

This patch adds an explicit, disabled-by-default DQN teacher-retention path intended to reduce 250k-to-500k action-pocket drift without changing the external action contract.

Preserved contract:

- MDP contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`
- Discrete action count: `48`
- External action ids and `0..47` semantics unchanged

No PPO retention was added. The current causal evidence points to DQN action-pocket drift across actions `32`, `33`, `41`, `45`, and `46`; adding PPO anchoring now would widen the patch beyond the observed failure mechanism.

## Files Changed

- `src/learn/teacher_retention.py`
  - New teacher-retention helper module.
  - Adds strict read-only teacher checkpoint loading.
  - Adds explicit config parsing and disabled-by-default guard.
  - Adds gate-critical teacher state bank.
  - Adds DQN CE/KL retention loss and bad-pocket Q-margin telemetry.
  - Adds runtime support for production/candidate dual teachers.

- `src/learn/train_joint_torch.py`
  - Adds CLI flags:
    - `--enable-teacher-retention`
    - `--teacher-production-checkpoint`
    - `--teacher-candidate-checkpoint`
  - Wires optional teacher-retention runtime into DQN updates only.
  - Records gate-critical observations into a scenario state bank.
  - Persists/restores teacher state-bank state in future exact-resume joint checkpoints when teacher retention is enabled.
  - Adds training metrics fields:
    - `teacher_dqn_kl_window`
    - `teacher_dqn_ce_window`
    - `teacher_dqn_action_match_rate_window`
    - `teacher_retention_loss_window`
    - `gate_critical_action_drift_window`
    - `bad_pocket_margin_by_action_window`
    - `teacher_bank_scenario_counts`

- `tests/learn/test_train_joint_curriculum.py`
  - Adds TDD coverage for explicit enable guards, read-only teacher loading, strict checkpoint contract/version checks, gate-critical state-bank sampling, bounded finite DQN retention loss, bad-pocket action drift, teacher-agreed valid action-33 behavior, telemetry aggregation/reset, disabled default behavior, and teacher state-bank exact-resume persistence.

## Design Details

Teacher loading is read-only and strict:

- Requires v5 contract.
- Requires observation dim `73`.
- Requires discrete action count `48`.
- Requires `torch_joint_policy_v1` checkpoint version.
- Freezes teacher parameters and puts teacher policies in eval mode.

Teacher retention is opt-in only:

- Config default is disabled.
- `teacher_retention.enabled=true` requires `explicit_enable=true`.
- CLI enablement requires `--enable-teacher-retention`.
- Enabled config requires at least one teacher checkpoint.

Dual-teacher selection:

- Production teacher is the default anchor for baseline states.
- Candidate teacher is preferred for lead-time, route-disruption, premium, high-holding, vehicle-scarcity, mixed-stress, and demand-spike states.
- If a preferred teacher is unavailable, the runtime falls back to the loaded teacher.

State bank:

- Records sampled gate-critical observations by canonical scenario:
  - `baseline`
  - `lead_time`
  - `route_disruption`
  - `premium_sla`
  - `high_holding`
  - `vehicle_scarcity`
  - `mixed_stress`
  - `demand_spike`
- Validates finite observation tensors and dimension `73`.
- Future exact-resume checkpoints include `teacher_retention_state_bank` when retention is enabled, so the new retention path does not create hidden non-resumed trainer state.

DQN retention loss:

- Uses teacher action CE as excess over teacher self-CE, so identical student/teacher logits do not apply unnecessary force.
- Uses KL against teacher Q-softmax distribution.
- Uses optional Q-margin against bad pockets `32`, `33`, `41`, `45`, and `46`.
- Bad-pocket margin excludes rows where the teacher itself selects that action, so useful teacher-agreed dispatch through action `33` is not overridden by the bad-pocket guard.
- Final weighted retention loss is bounded by `max_loss`.

## Tests Run

RED evidence:

- `python -m unittest tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_teacher_retention_config_is_disabled_by_default_and_requires_explicit_enable -v`
- Initial result: FAIL, missing `teacher_retention_config_from_training_config`.

Focused teacher-retention tests:

- `python -m unittest tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_teacher_retention_config_is_disabled_by_default_and_requires_explicit_enable tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_teacher_checkpoint_loads_read_only_and_rejects_contract_or_version_mismatch tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_teacher_state_bank_records_gate_critical_scenario_counts_and_samples tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_teacher_retention_loss_is_finite_bounded_and_detects_bad_action_33_drift tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_teacher_retention_does_not_penalize_teacher_agreed_valid_bad_pocket_dispatch tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_teacher_retention_telemetry_is_aggregated_in_timing_metrics_and_resets -v`
- Result: PASS, 6 tests.

Exact-resume teacher bank tests:

- `python -m unittest tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_checkpoint_payload_includes_replay_state_rng_state_and_resume_metadata tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_checkpoint_payload_includes_teacher_retention_state_bank_when_present tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_load_training_checkpoint_restores_replay_rng_and_global_step tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_load_training_checkpoint_restores_teacher_retention_state_bank_when_required -v`
- Result: PASS, 4 tests.

Full trainer plumbing:

- `python -m unittest tests.learn.test_train_joint_curriculum -v`
- Result: PASS, 43 tests.

Replay/reward blending:

- `python -m unittest tests.learn.test_joint_reward_blending -v`
- Result: PASS, 10 tests.

Long-run gate:

- `python -m unittest tests.eval.test_long_run_gate -v`
- Result: PASS, 13 tests.

Compile:

- `python -m py_compile src\learn\teacher_retention.py src\learn\train_joint_torch.py tests\learn\test_train_joint_curriculum.py`
- Result: PASS.

Additional non-blocking suite run:

- `python -m unittest tests.learn.test_curriculum_config -v`
- Result: FAIL, 1 known unrelated stale-artifact assertion:
  - `test_prod_stability_v2_4_250k_config_loads_and_is_manual_run_safe` expects `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610` not to exist, but this workspace already contains the completed V2.4 artifact.
  - No artifact was deleted or moved.

## Run, Eval, And Gate Status

Post-review ladder status is documented in [[docs/runs/20260611_level2_teacher_retention_ladder_report]].

Config created after review approval:

- `configs/training_joint_curriculum_v5_prod_teacher_retention_250k_20260611.json`

Run dir:

- `models/checkpoints/joint_torch_v5_prod_teacher_retention_250k_20260611`

Eval dir:

- `models/eval/joint_torch_v5_prod_teacher_retention_250k_20260611_offline_scenarios`

Gate decision:

- teacher-retention 250k: `FAIL`, long-run gate exit code `1`

Stop condition:

- No 500k config/run was created.
- No 1M config/run was created.
- No registry or production promotion action was taken.

## Teacher-Retention Telemetry

Telemetry fields are implemented, unit-tested, and populated in the teacher-retention 250k training run.

Last 250k training metrics row:

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
- `teacher_bank_scenario_counts`: all eight configured scenario banks reached `512`.

## Exact-Resume Proof

Existing exact replay/RNG behavior remains covered by trainer tests.

The teacher-retention 250k checkpoint audit proves:

- `models/checkpoints/joint_torch_v5_prod_teacher_retention_250k_20260611/joint_torch_latest.pt` includes DQN replay state.
- RNG state is present.
- `teacher_retention_state_bank` is present.
- `resume_state.exact_resume_capable` is true.
- Replay size is `250000`.
- Teacher bank size is `4096`, with `512` observations in each configured scenario bank.

Additional teacher-retention exact-resume tests prove:

- Joint checkpoint payloads include `teacher_retention_state_bank` when a teacher state bank is passed.
- Resume metadata records teacher bank inclusion, total size, and scenario counts.
- `load_training_checkpoint(..., require_teacher_retention_state=True)` restores the teacher bank.
- Required teacher-retention resume rejects checkpoints missing the teacher bank.

## Protected No-Mutation Proof

After the patch:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Source production checkpoint SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles:

- `models/production`: 4 files, 15040339 bytes, max write UTC `2026-06-08 23:38:51Z`
- `models/baselines`: 2692 files, 7392576274 bytes, max write UTC `2026-05-24 20:55:21Z`
- `db`: 7 files, 28330 bytes, max write UTC `2026-05-19 11:13:31Z`

Process check:

- No running `python` / `pythonw` training or eval process found.

## Independent Review

Status: complete.

Reviewer verdict:

- `PATCH_APPROVED_FOR_NEXT_GATE`

The approved patch was used for the teacher-retention 250k ladder rung. That rung failed the long-run gate; see [[docs/runs/20260611_level2_teacher_retention_ladder_report]].

Current classification: `TEACHER_RETENTION_250K_GATE_FAILED`
