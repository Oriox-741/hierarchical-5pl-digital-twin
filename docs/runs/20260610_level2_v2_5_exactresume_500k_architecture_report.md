# Level 2 V2.5 Exact-Resume 500k Architecture Report

Date: 2026-06-10

## Decision

Stop the Level 2 ladder before 1M.

Classification: `EXACT_RESUME_500K_GATE_FAILED`

Reason: the new V2.5 250k rung was produced from the protected production parent with the patched trainer, passed offline scenarios and the long-run gate, and saved replay/RNG state. The 500k rung then used normal `--resume` from that new 250k checkpoint and the trainer restored replay/RNG exactly. Despite fixing the old empty-replay continuation gap, the 500k candidate failed the long-run gate on route service and lead-time no-useful-work action quality. No 1M config, run, or eval was created.

## Artifacts

250k exact-resume-capable rung:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610.json`
- Parent checkpoint: `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`
- Output dir: `models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610`
- Final checkpoint: `models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610/joint_torch_latest.pt`
- Eval dir: `models/eval/joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610_offline_scenarios`
- Gate result: `docs/runs/20260610_level2_v2_5_exactresume_250k_gate_result.json`
- Gate decision: PASS, exit 0

500k exact-resume rung:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_5_exactresume_500k_20260610.json`
- Resume parent: `models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610/joint_torch_latest.pt`
- Output dir: `models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610`
- Final checkpoint: `models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610/joint_torch_latest.pt`
- Eval dir: `models/eval/joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610_offline_scenarios`
- Gate result: `docs/runs/20260610_level2_v2_5_exactresume_500k_gate_result.json`
- Gate decision: FAIL, exit 2

No 1M artifacts:

- No `configs/*v2_5_exactresume_1m*`
- No `models/checkpoints/*v2_5_exactresume_1m*`
- No `models/eval/*v2_5_exactresume_1m*`

## Config Safety

The 250k config uses the production parent only:

```text
models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt
```

The 500k config declares exact ladder continuation from the new V2.5 250k checkpoint:

```text
mode: exact_resume_from_passed_checkpoint
parent_checkpoint: models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610/joint_torch_latest.pt
requires_replay_state: true
requires_rng_state: true
legacy_warm_resume_allowed: false
```

Neither config references old V2.4, V2.3, V2.2, V2.1, V2, nextgen, failed 300k/400k/500k, production registry, baseline, or DB paths as mutable parents.

## Training Commands

250k used weights-only initialization from production:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:TORCH_NUM_THREADS='2'; $env:TORCH_NUM_INTEROP_THREADS='1'; python -m src.learn.train_joint_torch --config configs\training_joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610.json --output-dir models\checkpoints\joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610 --init-from-joint-checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt --seed 42 --device cpu --final-ppo-path models\checkpoints\joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610\ppo_torch_joint_final_stability_v2_5_exactresume_250k_20260610.pt --final-dqn-path models\checkpoints\joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610\dqn_torch_joint_final_stability_v2_5_exactresume_250k_20260610.pt --skip-registry --disable-trace-logging --torch-num-threads 2 --torch-num-interop-threads 1
```

Training exit code: 0.

Trainer start semantics:

```text
init_from_joint_checkpoint=true replay_buffer=empty optimizers=fresh global_step=0 parent_global_step=200000 metrics_state=fresh
```

500k used exact resume from the new 250k checkpoint:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:TORCH_NUM_THREADS='2'; $env:TORCH_NUM_INTEROP_THREADS='1'; python -m src.learn.train_joint_torch --config configs\training_joint_curriculum_v5_prod_stability_v2_5_exactresume_500k_20260610.json --output-dir models\checkpoints\joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610 --resume models\checkpoints\joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610\joint_torch_latest.pt --seed 42 --device cpu --final-ppo-path models\checkpoints\joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610\ppo_torch_joint_final_stability_v2_5_exactresume_500k_20260610.pt --final-dqn-path models\checkpoints\joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610\dqn_torch_joint_final_stability_v2_5_exactresume_500k_20260610.pt --skip-registry --disable-trace-logging --torch-num-threads 2 --torch-num-interop-threads 1
```

Training exit code: 0.

Trainer start semantics:

```text
resume=exact replay_buffer=restored replay_size=250000 dqn_learning_starts=10000 rng_state=restored metrics_state=reinitialized
```

`--allow-empty-replay-resume` was never used.

## Exact-Resume Proof

250k final checkpoint:

- `global_step`: 250000
- `artifact_kind`: `joint_final`
- Observation dimension: 73
- Discrete action count: 48
- `dqn_replay_buffer_state`: present
- `rng_state`: present
- `resume_state.exact_resume_capable`: true
- `resume_state.dqn_replay_size`: 250000
- `resume_state.dqn_replay_position`: 250000
- `resume_state.rng_state_included`: true

500k final checkpoint:

- `global_step`: 500000
- `artifact_kind`: `joint_final`
- Observation dimension: 73
- Discrete action count: 48
- `dqn_replay_buffer_state`: present
- `rng_state`: present
- `resume_state.exact_resume_capable`: true
- `resume_state.dqn_replay_size`: 500000
- `resume_state.dqn_replay_position`: 0
- `resume_state.rng_state_included`: true

1M was not reached because the 500k gate failed.

## Independent Review

One GPT-5.5 read-only reviewer inspected the V2.5 configs, config tests, 250k/500k gate evidence, and scenario summaries.

Reviewer verdict:

```text
AUTOPILOT_NEEDS_ARCHITECTURE_DECISION
```

Interpretation: no next gate should be attempted from this state. The exact-resume mechanics and ladder guardrails are documented, but the 500k behavior requires architecture-level diagnosis before any further config/training attempt.

## Gate Decisions

250k gate:

- Decision: PASS
- Exit code: 0
- Fatal failures: none

500k gate:

- Decision: FAIL
- Exit code: 2
- Fatal failures:
  - `route_disruption_congestion: scenario threshold verdict is 'FAIL'.`
  - `route_disruption_congestion: scenario threshold failure: service_level 0.861 < 0.880`
  - `lead_time_volatility: no-current-work/no-unassigned total exploded from 0 to 740.`
  - `lead_time_volatility: no-current-work/no-unassigned top-action explosion on action 33: count=228, production_count=0, top_percentage=0.236.`

## Scenario Comparison

| Scenario | Run | Verdict | Service | Dispatch success | Dispatch rate | No-work total | Top action | Top action % | Premium missed useful |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | production | PASS | 0.938 | 1.000 | 0.487 | 0 | 33 | 0.175 | n/a |
| baseline_normal | v2.5_250k | PASS | 0.961 | 1.000 | 0.446 | 1 | 16 | 0.181 | n/a |
| baseline_normal | v2.5_500k | PASS | 0.978 | 0.998 | 0.416 | 4 | 1 | 0.236 | n/a |
| demand_spike_volatility | production | PASS | 0.843 | 1.000 | 0.728 | 0 | 29 | 0.203 | n/a |
| demand_spike_volatility | v2.5_250k | PASS | 0.948 | 1.000 | 0.956 | 0 | 29 | 0.265 | n/a |
| demand_spike_volatility | v2.5_500k | PASS | 0.934 | 1.000 | 0.856 | 1 | 24 | 0.388 | n/a |
| high_holding_cost | production | PASS | 0.943 | 1.000 | 0.387 | 0 | 9 | 0.238 | n/a |
| high_holding_cost | v2.5_250k | PASS | 0.990 | 1.000 | 0.392 | 0 | 16 | 0.247 | n/a |
| high_holding_cost | v2.5_500k | PASS | 0.983 | 1.000 | 0.387 | 0 | 8 | 0.212 | n/a |
| lead_time_volatility | production | PASS | 0.936 | 1.000 | 0.455 | 0 | 4 | 0.214 | n/a |
| lead_time_volatility | v2.5_250k | PASS | 0.962 | 0.998 | 0.412 | 7 | 16 | 0.268 | n/a |
| lead_time_volatility | v2.5_500k | PASS | 0.985 | 0.856 | 0.447 | 740 | 33 | 0.236 | n/a |
| mixed_stress | production | PASS | 0.951 | 0.996 | 0.911 | 6 | 33 | 0.292 | n/a |
| mixed_stress | v2.5_250k | PASS | 0.936 | 1.000 | 0.934 | 0 | 38 | 0.380 | n/a |
| mixed_stress | v2.5_500k | PASS | 0.888 | 1.000 | 0.943 | 2 | 33 | 0.368 | n/a |
| premium_sla_pressure | production | PASS | 0.980 | 0.769 | 0.427 | 178 | 28 | 0.160 | 0.504 |
| premium_sla_pressure | v2.5_250k | PASS | 0.988 | 0.999 | 0.365 | 2 | 20 | 0.466 | 0.128 |
| premium_sla_pressure | v2.5_500k | PASS | 1.000 | 1.000 | 0.357 | 0 | 36 | 0.260 | 0.013 |
| route_disruption_congestion | production | PASS | 0.929 | 0.822 | 0.575 | 171 | 40 | 0.264 | n/a |
| route_disruption_congestion | v2.5_250k | PASS | 0.946 | 1.000 | 0.482 | 0 | 29 | 0.259 | n/a |
| route_disruption_congestion | v2.5_500k | FAIL | 0.861 | 0.976 | 0.487 | 121 | 9 | 0.258 | n/a |
| vehicle_scarcity_capacity_shock | production | PASS | 0.914 | 1.000 | 0.731 | 0 | 29 | 0.293 | n/a |
| vehicle_scarcity_capacity_shock | v2.5_250k | PASS | 0.942 | 1.000 | 0.764 | 0 | 24 | 0.267 | n/a |
| vehicle_scarcity_capacity_shock | v2.5_500k | PASS | 0.950 | 1.000 | 0.814 | 0 | 33 | 0.350 | n/a |

## Causal Findings

The exact-resume mechanics are fixed for new checkpoints. The 500k trainer restored DQN replay with `replay_size=250000` and restored RNG state before continuing. The final 500k checkpoint also contains replay/RNG state and is exact-resume capable.

The old V2.4 500k failure was partly explained by a continuation architecture gap: model/optimizer/global step were resumed while DQN replay and RNG started fresh. The V2.5 ladder confirms that gap was real because premium SLA, high-holding, and the old route dispatch-success collapse were materially improved after exact replay/RNG restoration.

However, exact replay/RNG restoration is not sufficient for stable 500k behavior. The remaining 500k failure is a policy-stability problem:

- Premium SLA recovered strongly: service `1.000`, no-work `0`, missed useful dispatch `0.013`.
- High holding stayed clean: service `0.983`, dispatch success `1.000`, no-work `0`.
- Route dispatch success improved versus production and old V2.4 500k, but route service fell to `0.861`, below the `0.880` threshold.
- Lead-time service remained high at `0.985`, but dispatch action quality regressed with no-current-work/no-unassigned total `740` and action 33 concentration.

Lead-time action-quality detail:

- 250k lead-time no-work total: `7`
- 500k lead-time no-work total: `740`
- 500k top failed no-op actions:
  - action 27: 209 failed no-ops
  - action 33: 114 failed no-ops
  - action 45: 30 failed no-ops
  - action 29: 13 failed no-ops
  - action 32: 1 failed no-op

Route detail:

- 250k route service: `0.946`
- 500k route service: `0.861`
- 500k route dispatch success: `0.976`
- 500k route no-work total: `121`, lower than production `171`
- 500k top action: action 9 hold, count share `0.258`
- 500k route failed no-op actions: action 46 = 45, action 32 = 12, action 27 = 3, action 45 = 1

Interpretation: V2.5 exact resume prevented the earlier premium/high-holding collapse and old route dispatch-success failure, but it did not preserve the full passing 250k service/action-quality profile through 500k. The next V-cycle should diagnose retention and action-quality drift under exact replay rather than change reward/config blindly.

## Tests And Verification

Pre-training exact-resume tests:

- `python -m unittest tests.learn.test_joint_reward_blending.DQNReplayBufferStorageTests.test_replay_state_dict_round_trips_ring_storage_and_cursor tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_checkpoint_payload_includes_replay_state_rng_state_and_resume_metadata tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_periodic_checkpoint_save_writes_replay_state_to_joint_latest_only tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_load_training_checkpoint_restores_replay_rng_and_global_step tests.learn.test_train_joint_curriculum.TrainJointCurriculumPlumbingTests.test_load_training_checkpoint_rejects_missing_replay_unless_explicitly_allowed`
- Result: PASS, 5 tests

Pre-training relevant learn tests:

- `python -m unittest tests.learn.test_joint_reward_blending tests.learn.test_train_joint_curriculum`
- Result: PASS, 45 tests
- `python -m unittest tests.learn.test_cold_start_contract tests.learn.test_model_registry tests.learn.test_pretrain_mdp_sanity_check tests.learn.test_promote_torch_joint_production tests.learn.test_register_torch_joint_candidate tests.learn.test_trace_sampling_contract`
- Result: PASS, 58 tests

Config TDD:

- 250k config test first failed because the config did not exist.
- After adding the 250k config, the 250k config test passed.
- 500k config test first failed because the config did not exist.
- After adding the 500k config, the 500k config test passed.
- After the ladder legitimately created the output dirs, the config tests were updated to assert unique unprotected checkpoint paths instead of permanent non-existence.
- Final config test command: `python -m unittest tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_stability_v2_5_exactresume_250k_config_loads_and_is_manual_run_safe tests.learn.test_curriculum_config.CurriculumConfigTests.test_prod_stability_v2_5_exactresume_500k_config_loads_and_requires_exact_resume -v`
- Result: PASS, 2 tests

Compile checks:

- `python -m py_compile tests\learn\test_curriculum_config.py`
- Result: PASS

## Protected-Path Status

No registry, production, baseline, DB, or source production checkpoint mutation was performed.

Protected hashes:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Source production checkpoint SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles:

- `models/production`: 4 files, 15040339 bytes, max_mtime_ns 1780972731823548200
- `models/baselines`: 2692 files, 7392576274 bytes, max_mtime_ns 1779666921467665500
- `db`: 7 files, 28330 bytes, max_mtime_ns 1779200011801591300

## Next Architecture Question

Do not run 1M from V2.5 500k. Do not promote/register either V2.5 rung. The next V-cycle should answer why exact-replay continuation still drifts on route service and lead-time no-useful-work action quality. Candidate investigation paths:

- compare 250k and 500k Q-values/action distributions on fixed lead-time and route state banks,
- add teacher-retention diagnostics before reward/config changes,
- evaluate whether replay retention alone needs a behavior-retention objective or checkpoint selection at shorter intervals,
- inspect action 27/33/45 no-useful-work pockets in lead-time and action 9 route-hold service timing in route disruption.
