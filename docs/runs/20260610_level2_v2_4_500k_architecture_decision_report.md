# Level 2 V2.4 500k Architecture Decision Report

Date: 2026-06-10

## Decision

Stop the Level 2 ladder before 1M.

Classification: `AUTOPILOT_BLOCKED_NEEDS_ARCHITECTURE_DECISION`

Reason: V2.4 250k passed after evaluator-semantics resolution, but V2.4 500k failed the long-run gate. The failure is not a simple config-length issue. The continuation run resumes model and optimizer state while starting with an empty replay buffer, then trains for another 250k steps without a retention mechanism for the passing 250k behavior. Periodic checkpoint diagnostics show degradation begins by 300k and becomes catastrophic by 400k.

## Protected-Path Status

No registry, production, baseline, DB, or source production checkpoint mutation was performed.

Protected hashes after the 500k run/evals:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Source production checkpoint SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles:

- `models/production`: 4 files, 15040339 bytes, max_mtime_ns 1780972731823548200
- `models/baselines`: 2692 files, 7392576274 bytes, max_mtime_ns 1779666921467665500
- `db`: 7 files, 28330 bytes, max_mtime_ns 1779200011801591300

## Artifacts

Passing 250k semantics gate:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`
- Checkpoint: `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610/joint_torch_latest.pt`
- Candidate eval: `models/eval/joint_torch_v5_prod_stability_v2_4_250k_20260610_offline_scenarios_semantics_v2_20260610`
- Production comparator eval: `models/eval/joint_torch_v5_balanced_retention_ft_200k_20260608_offline_scenarios_semantics_v2_20260610`
- Gate result: `docs/runs/20260610_level2_v2_4_semantics_gate_result.json`
- Gate decision: PASS, exit 0

Failed 500k continuation:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_4_500k_20260610.json`
- Parent resume checkpoint: `models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610/joint_torch_latest.pt`
- Output dir: `models/checkpoints/joint_torch_v5_prod_stability_v2_4_500k_20260610`
- Final checkpoint: `models/checkpoints/joint_torch_v5_prod_stability_v2_4_500k_20260610/joint_torch_latest.pt`
- Candidate eval: `models/eval/joint_torch_v5_prod_stability_v2_4_500k_20260610_offline_scenarios_20260610`
- Gate result: `docs/runs/20260610_level2_v2_4_500k_gate_result.json`
- Gate decision: FAIL, exit 2

Periodic diagnostic evals:

- 300k eval: `models/eval/joint_torch_v5_prod_stability_v2_4_300k_20260610_offline_scenarios_diag`
- 300k gate: `docs/runs/20260610_level2_v2_4_300k_diag_gate_result.json`
- 400k eval: `models/eval/joint_torch_v5_prod_stability_v2_4_400k_20260610_offline_scenarios_diag`
- 400k gate: `docs/runs/20260610_level2_v2_4_400k_diag_gate_result.json`

## Training Command

The 500k rung used resume semantics, not fresh fine-tune initialization:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:TORCH_NUM_THREADS='2'; $env:TORCH_NUM_INTEROP_THREADS='1'; python -m src.learn.train_joint_torch --config configs\training_joint_curriculum_v5_prod_stability_v2_4_500k_20260610.json --output-dir models\checkpoints\joint_torch_v5_prod_stability_v2_4_500k_20260610 --resume models\checkpoints\joint_torch_v5_prod_stability_v2_4_250k_20260610\joint_torch_latest.pt --seed 42 --device cpu --final-ppo-path models\checkpoints\joint_torch_v5_prod_stability_v2_4_500k_20260610\ppo_torch_joint_final_stability_v2_4_500k_20260610.pt --final-dqn-path models\checkpoints\joint_torch_v5_prod_stability_v2_4_500k_20260610\dqn_torch_joint_final_stability_v2_4_500k_20260610.pt --skip-registry --disable-trace-logging --torch-num-threads 2 --torch-num-interop-threads 1
```

Training exit code: 0.

The trainer reported:

```text
resume=warm_start replay_buffer=empty dqn_learning_starts=10000 metrics_state=reinitialized
```

Final checkpoint audit:

- `global_step`: 500000
- `artifact_kind`: `joint_final`
- `environment_id`: `joint_curriculum_v5_prod_stability_v2_4_500k_20260610`
- `total_timesteps`: 500000
- MDP contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: 73

## Gate Failures

V2.4 500k fatal failures:

- `premium_sla_pressure`: scenario threshold verdict is `FAIL`.
- `premium_sla_pressure`: `service_level 0.811 < 0.920`.
- `route_disruption_congestion`: dispatch success regressed from `0.862` to `0.622`.
- `high_holding_cost`: no-current-work/no-unassigned total exploded from `11` to `125`.
- `route_disruption_congestion`: no-current-work/no-unassigned total exploded from `868` to `2930`.
- `route_disruption_congestion`: top-action explosion on action 45, count `2350`, production count `92`, top percentage `0.236`.

## Scenario Comparison

| Scenario | Run | Verdict | Service | Dispatch success | Dispatch rate | No-work total | Top action | Top action % | Stuck | Active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_normal | production | PASS | 0.945 | 0.998 | 0.459 | 6 | 33 | 0.134 | 0 | 0 |
| baseline_normal | v2.4_250k | PASS | 0.961 | 1.000 | 0.446 | 1 | 16 | 0.181 | 0 | 1 |
| baseline_normal | v2.4_500k | PASS | 0.948 | 0.993 | 0.429 | 33 | 16 | 0.209 | 0 | 0 |
| premium_sla_pressure | production | PASS | 0.982 | 0.759 | 0.439 | 1260 | 28 | 0.159 | 0 | 0 |
| premium_sla_pressure | v2.4_250k | PASS | 0.988 | 0.999 | 0.365 | 2 | 20 | 0.466 | 0 | 0 |
| premium_sla_pressure | v2.4_500k | FAIL | 0.811 | 0.992 | 0.429 | 40 | 21 | 0.171 | 0 | 0 |
| high_holding_cost | production | PASS | 0.931 | 0.997 | 0.446 | 11 | 9 | 0.210 | 0 | 0 |
| high_holding_cost | v2.4_250k | PASS | 0.990 | 1.000 | 0.392 | 0 | 16 | 0.247 | 0 | 0 |
| high_holding_cost | v2.4_500k | PASS | 0.995 | 0.972 | 0.384 | 125 | 8 | 0.268 | 0 | 0 |
| lead_time_volatility | production | PASS | 0.914 | 0.998 | 0.460 | 8 | 4 | 0.231 | 0 | 0 |
| lead_time_volatility | v2.4_250k | PASS | 0.962 | 0.998 | 0.412 | 7 | 16 | 0.268 | 0 | 0 |
| lead_time_volatility | v2.4_500k | PASS | 0.935 | 1.000 | 0.437 | 1 | 8 | 0.229 | 0 | 0 |
| route_disruption_congestion | production | PASS | 0.904 | 0.862 | 0.551 | 868 | 40 | 0.266 | 0 | 0 |
| route_disruption_congestion | v2.4_250k | PASS | 0.946 | 1.000 | 0.482 | 0 | 29 | 0.259 | 0 | 0 |
| route_disruption_congestion | v2.4_500k | PASS | 0.964 | 0.622 | 0.673 | 2930 | 45 | 0.236 | 0 | 0 |
| vehicle_scarcity_capacity_shock | production | PASS | 0.908 | 0.998 | 0.748 | 9 | 29 | 0.276 | 0 | 0 |
| vehicle_scarcity_capacity_shock | v2.4_250k | PASS | 0.942 | 1.000 | 0.764 | 0 | 24 | 0.267 | 0 | 1 |
| vehicle_scarcity_capacity_shock | v2.4_500k | PASS | 0.970 | 0.995 | 0.712 | 38 | 36 | 0.149 | 0 | 1 |
| demand_spike_volatility | production | PASS | 0.840 | 1.000 | 0.756 | 0 | 29 | 0.209 | 0 | 0 |
| demand_spike_volatility | v2.4_250k | PASS | 0.948 | 1.000 | 0.956 | 0 | 29 | 0.265 | 0 | 1 |
| demand_spike_volatility | v2.4_500k | PASS | 0.955 | 1.000 | 0.987 | 2 | 32 | 0.310 | 0 | 0 |
| mixed_stress | production | PASS | 0.940 | 0.997 | 0.943 | 36 | 33 | 0.274 | 0 | 1 |
| mixed_stress | v2.4_250k | PASS | 0.936 | 1.000 | 0.934 | 0 | 38 | 0.380 | 0 | 0 |
| mixed_stress | v2.4_500k | PASS | 0.941 | 0.992 | 0.979 | 84 | 45 | 0.414 | 0 | 0 |

## Periodic Checkpoint Diagnostics

| Step | Premium svc | Premium missed useful dispatch | Premium top action | Route dispatch success | Route no-work total | Route top failed no-op | High-holding no-work | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production | 0.982 | 0.493 | 28 (0.159) | 0.862 | 868 | 41 (343) | 11 | n/a |
| 250k | 0.988 | 0.128 | 20 (0.466) | 1.000 | 0 | None (0) | 0 | PASS |
| 300k | 0.913 | 0.361 | 29 (0.137) | 0.983 | 92 | 37 (45) | 10 | FAIL |
| 400k | 0.356 | 0.985 | 3 (0.576) | 0.858 | 743 | 32 (320) | 0 | FAIL |
| 500k | 0.811 | 0.528 | 21 (0.171) | 0.622 | 2930 | 45 (1175) | 125 | FAIL |

300k gate failures:

- `baseline_normal`: service `0.920 < 0.930`.
- `premium_sla_pressure`: service `0.913 < 0.920`.
- `route_disruption_congestion`: service `0.860 < 0.880`.
- Baseline service regressed from production `0.945` to `0.920`.

400k gate failures:

- Baseline, high holding, lead time, mixed stress, premium SLA, and vehicle scarcity scenario threshold failures.
- Premium service collapsed to `0.356`; missed useful dispatch rate rose to `0.985`.
- Route no-work top-action explosion shifted to action 32.

500k gate failures:

- Premium service partially recovered to `0.811` but remained below threshold.
- Route action-quality collapse shifted from action 32 to action 45.

## Action Semantics

Relevant decoded actions:

- Action 8: hold, low_congestion, secondary_fleet, none.
- Action 20: hold, high_resilience, primary_fleet, none.
- Action 21: hold, high_resilience, primary_fleet, conservative.
- Action 24: dispatch, shortest, secondary_fleet, none.
- Action 25: dispatch, shortest, secondary_fleet, conservative.
- Action 28: dispatch, shortest, primary_fleet, none.
- Action 29: dispatch, shortest, primary_fleet, conservative.
- Action 32: dispatch, low_congestion, secondary_fleet, none.
- Action 37: dispatch, low_congestion, primary_fleet, conservative.
- Action 41: dispatch, high_resilience, secondary_fleet, conservative.
- Action 45: dispatch, high_resilience, primary_fleet, conservative.

## Causal Findings

### 1. The 250k failure was evaluator semantics, and it is resolved

The V2.4 250k baseline stuck-order failure was caused by treating any active assigned/in-transit order at the finite horizon as fatal. The active order had due time in the future and was not late. The evaluator now reports raw active horizon work separately while counting only overdue, explicitly late, or stale active work as `stuck_assigned_in_transit_orders`.

After that patch and independent GPT-5.5 review, V2.4 250k passed all scenario thresholds and the long-run gate.

### 2. The 500k failure is continuation instability, not stuck-order semantics

V2.4 500k has zero fatal stuck orders. The failure is behavioral:

- Premium service collapses while dispatch success remains high.
- Route disruption service remains high, but dispatch success collapses and no-work/no-unassigned action quality explodes.
- High holding service improves, but no-work/no-unassigned quality regresses enough to fail the long-run gate.

### 3. Premium failure is missed useful work, not dispatch infeasibility

Premium SLA:

- 250k: service `0.988`, dispatch success `0.999`, missed useful dispatch rate `0.128`.
- 500k: service `0.811`, dispatch success `0.992`, missed useful dispatch rate `0.528`.
- Premium pressure exists in all 5760 eval steps.
- Premium late/at-risk backlog rises from `2975` at 250k to `5297` at 500k.
- Dispatch-primary-under-premium-pressure drops from `1948` at 250k to `1277` at 500k.

Interpretation: the policy still succeeds when it dispatches, but it often chooses holds/reorders under useful dispatch pressure. This is a premium responsiveness regression, not a feasibility regression.

### 4. Route failure shifted from the old action-32 pocket to action 45

Route disruption:

- 250k: dispatch success `1.000`, no-work total `0`.
- 400k: top no-work failure was action 32.
- 500k: dispatch success `0.622`, no-work total `2930`, top failed no-op action 45 with `1175` failed no-ops.

At 500k:

- Action 45 = dispatch, high_resilience, primary_fleet, conservative.
- Action 45 attempts: `1360`.
- Action 45 successes: `185`.
- Action 45 failed no-ops: `1175`.
- Action 45 success ratio: `0.136`.
- High-resilience route success ratio: `0.472`.
- Low-congestion route success ratio: `0.992`.
- Shortest route success ratio: `0.997`.

Interpretation: the route-quality issue no longer primarily concerns low-congestion action 32. Longer continuation discovered a different high-resilience primary/conservative failure pocket.

### 5. The likely architecture fault is resume-with-empty-replay continuation

The 500k run resumed model and optimizer state, but the trainer explicitly reported:

```text
replay_buffer=empty
```

That means the DQN continuation starts from a good policy but relearns from only the second-cycle stream of fresh transitions. There is no replay retention of the passing 250k distribution, no teacher/behavior regularizer, and no checkpoint-selection gate until after the full 500k run. The periodic diagnostics show this is not just a terminal artifact:

- 300k already fails baseline, premium, and route service gates.
- 400k collapses most stress scenarios.
- 500k recovers some service but leaves premium below threshold and route action quality broken.

This is an architecture-level continuation problem, not a safe place for blind config tuning.

## Required Architecture Decision

Before any new 500k or 1M attempt, choose one of these designs:

1. **Replay-state continuation**
   - Persist and restore enough DQN replay state to make resume training a true continuation.
   - Add tests proving resume does not silently start with an empty replay buffer when the run is declared a ladder continuation.
   - Risk: larger checkpoints and serialization complexity.

2. **Teacher-retention continuation**
   - Keep the passed V2.4 250k checkpoint as a teacher.
   - Add a fixed scenario-state bank and penalize DQN action-value/logit drift on clean premium, route, high-holding, and baseline retention states.
   - Continue with fresh replay allowed, but preserve behavior on known-good policy regions.
   - Risk: new training objective needs careful review and TDD.

3. **Short gated continuation only**
   - Do not attempt a direct 500k continuation.
   - Train in 50k increments from the passing 250k checkpoint, run scenario/gate diagnostics at each rung, and stop immediately at the first regression.
   - This improves safety but does not solve the underlying retention problem by itself.

Recommended next V-cycle:

- Use option 2 plus option 3.
- Do not use the failed 300k, 400k, or 500k checkpoints as parents.
- Treat V2.4 250k as the last clean approved checkpoint.
- Add teacher-retention diagnostics for:
  - premium useful-dispatch opportunity under pressure,
  - route action 45 no-useful-work high-resilience dispatch,
  - route action 32 low-congestion regression recurrence,
  - high-holding no-current/no-unassigned dispatch attempts.
- Add an early-stop rung gate at 300k before any 400k/500k continuation.
- Require independent review before training any new V-cycle config.

No 1M config was created.
No further training was started.
