# Stability V2.2 Recovery Plan

> Agentic execution note: this plan continues the bounded stability autopilot after the failed V2 and V2.1 250k gates. Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` for checklist execution. Use TDD for code changes. Do not create 500k or 1M artifacts until a fresh 250k candidate passes all gates.

## Current State

The bounded stability autopilot has already produced two fresh 250k design cycles:

- `configs/training_joint_curriculum_v5_prod_stability_v2_250k_20260609.json`
- `models/checkpoints/joint_torch_v5_prod_stability_v2_250k_20260609`
- `models/eval/joint_torch_v5_prod_stability_v2_250k_20260609_offline_scenarios`
- `configs/training_joint_curriculum_v5_prod_stability_v2_1_250k_20260609.json`
- `models/checkpoints/joint_torch_v5_prod_stability_v2_1_250k_20260609`
- `models/eval/joint_torch_v5_prod_stability_v2_1_250k_20260609_offline_scenarios`

Both cycles used the approved production parent:

`models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`

Both cycles preserved the contract:

- MDP contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`
- Discrete action count: `48`

No 500k or 1M continuation is authorized from either failed 250k cycle.

## Gate Results

| Candidate | Scenario pass count | Hard blockers | Primary failure |
| --- | ---: | --- | --- |
| Production 200k parent | 8/8 | Zero, all PASS | None |
| Stability V2 250k | 4/8 | Zero, all PASS | Timing/service under-response after no-work guard |
| Stability V2.1 250k | 4/8 | Zero, all PASS | Hold/high-resilience/secondary concentration causing service collapse |

### Stability V2 Failed Scenarios

| Scenario | Service | True lateness | Dispatch success | Failure |
| --- | ---: | ---: | ---: | --- |
| `high_holding_cost` | 0.960 | 0.118 | 1.000 | Lateness above 0.080 |
| `lead_time_volatility` | 0.945 | 0.126 | 1.000 | Lateness above 0.100 |
| `premium_sla_pressure` | 0.912 | 0.000 | 0.997 | Service below 0.920 |
| `route_disruption_congestion` | 0.944 | 0.234 | 0.881 | Lateness above 0.120 |

### Stability V2.1 Failed Scenarios

| Scenario | Service | True lateness | Dispatch success | Dispatch rate | Hold rate | Primary failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_normal` | 0.676 | 0.193 | 1.000 | 0.387 | 0.613 | Baseline service and lateness collapse |
| `high_holding_cost` | 0.741 | 0.000 | 0.923 | 0.476 | 0.524 | Service collapse with no-current action 29 exposure |
| `lead_time_volatility` | 0.051 | 0.511 | 1.000 | 0.265 | 0.735 | Severe hold/secondary/high-resilience collapse |
| `premium_sla_pressure` | 0.053 | 0.501 | 1.000 | 0.153 | 0.847 | Severe hold/secondary/high-resilience collapse |

Important action fingerprints:

- Action 16 = hold + high_resilience + secondary_fleet + none.
- Action 4 = hold + shortest + primary_fleet + none.
- Action 20 = hold + high_resilience + primary_fleet + none.
- Action 29 = dispatch + shortest + primary_fleet + conservative.
- Action 33 = dispatch + low_congestion + secondary_fleet + conservative.

V2.1 premium SLA concentrated on action 16 at 64.6% and action 4 at 16.3%. Lead-time volatility concentrated on action 16 at 39.7%, action 33 at 24.7%, and action 20 at 16.7%. This is not a return of the next-gen no-op dispatch failure; it is an over-hold and secondary-fleet timing failure.

## Root-Cause Assessment

### Confirmed

1. V2 fixed the major no-current/no-unassigned dispatch explosion but became too late in timing-sensitive scenarios.
2. V2.1 did not restore timing robustness. It over-shifted into hold/high-resilience/secondary-fleet behavior in premium SLA and lead-time volatility.
3. V2.1 kept hard blockers clean, so the failure is policy quality and reward/curriculum balance, not forbidden hard-blocker leakage.
4. Both V2 and V2.1 configs kept `baseline_anchor_probability` at `0`, despite the V2 design plan requiring nonzero baseline anchoring unless tests prove it harmful.
5. The latest `src/act/env_5pl.py` and `tests/act/test_reward_physics_contract.py` timestamps are newer than the V2.1 eval outputs, so current source cannot be treated as evaluated by the V2.1 run.

### Likely

1. The V2.1 reward/curriculum balance still let hold labels dominate under useful dispatch pressure.
2. Premium and lead-time failures indicate insufficient pressure-gated useful dispatch urgency and insufficient primary-fleet timing retention.
3. The zero baseline anchor allowed the short 250k run to drift away from production baseline behavior.
4. The current post-run reward code appears aimed at this failure via `hold_under_lateness_pressure_penalty`, but it must be verified before any new training run.

### Ruled Out

1. Failed next-gen parent usage: both V2 and V2.1 configs point to the production checkpoint.
2. Contract drift: eval summaries remain on `physical_reality_v5_route_candidate_visibility`, observation dim 73.
3. Need for 500k or 1M evidence: V2.1 already failed the 250k gate.
4. Registry/production promotion issue: no registry or production mutation is needed or allowed.

## Recovery Posture

V2.2 must be the third and final allowed 250k design cycle under the goal's three-failed-cycle limit. If V2.2 250k fails, stop with `AUTOPILOT_STOPPED_AFTER_FAILED_GATES` and a root-cause report. Do not create a fourth 250k design cycle without human approval.

The safe posture is:

1. Verify the current post-V2.1 reward patch first.
2. Keep the no-work dispatch guard.
3. Keep pressure-gated useful dispatch urgency.
4. Keep pressure-gated primary-fleet timing justification.
5. Add or preserve bounded hold-under-lateness pressure penalty.
6. Restore nonzero baseline anchoring.
7. Use a short alternating schedule with repeated normal/premium/lead-time retention.
8. Train only a fresh 250k candidate from production.

## Proposed V2.2 Curriculum Direction

Create a fresh config only after tests pass:

- Config: `configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json`
- Output dir: `models/checkpoints/joint_torch_v5_prod_stability_v2_2_250k_20260610`
- Eval dir: `models/eval/joint_torch_v5_prod_stability_v2_2_250k_20260610_offline_scenarios`
- Parent: `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`
- Baseline anchor probability: `0.10`
- Total steps: `250000`

Proposed 250k stage schedule:

| Stage | Steps | Purpose |
| --- | ---: | --- |
| `normal_v4` | 20000 | Preserve production baseline behavior |
| `premium_sla` | 20000 | Retain premium urgency without no-work credit |
| `normal_v4` | 15000 | Prevent premium overfit |
| `lead_time_delay` | 20000 | Restore lead-time timing response |
| `route_disruption` | 20000 | Retain route resilience without no-op drift |
| `normal_v4` | 15000 | Baseline retention |
| `high_holding_cost` | 20000 | Restore inventory/service response |
| `premium_sla` | 20000 | Recheck premium timing |
| `demand_spike` | 20000 | Preserve demand spike pass |
| `lead_time_delay` | 20000 | Recheck lead-time timing |
| `mixed_stress` | 20000 | Preserve mixed stress pass |
| `vehicle_scarcity` | 15000 | Preserve scarcity pass |
| `normal_v4` | 25000 | Final production-retention anchor |

This schedule intentionally gives normal, premium, and lead-time repeated exposure while avoiding long uninterrupted stress blocks.

## Required Tasks

### Task A: Verify Current Reward Patch

- [ ] Run reward tests:

```powershell
python -m unittest tests.act.test_reward_physics_contract -v
```

- [ ] Run training metrics tests:

```powershell
python -m unittest tests.learn.test_train_joint_curriculum -v
```

- [ ] Compile changed files:

```powershell
python -m py_compile src\act\env_5pl.py src\learn\joint_metrics.py tests\act\test_reward_physics_contract.py tests\learn\test_train_joint_curriculum.py
```

Expected:

- Tests pass.
- `no_current_or_unassigned_dispatch_penalty` remains active for no-work/no-unassigned dispatch.
- `useful_dispatch_lateness_urgency_credit` remains zero for no-work dispatch.
- `primary_fleet_timing_justification_credit` remains useful-work gated.
- `hold_under_lateness_pressure_penalty` remains bounded and only active when useful dispatch work is actually available under pressure.

### Task B: Confirm Gate Baselines

- [ ] Run long-run gate production self-check:

```powershell
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label production_self_check
```

- [ ] Confirm failed branches still fail:

```powershell
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_250k_20260609_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label stability_v2_250k
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_1_250k_20260609_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label stability_v2_1_250k
```

Expected:

- Production self-check exits 0 and decision PASS.
- V2 and V2.1 exit 2 and decision FAIL.

### Task C: Create V2.2 Config With Tests

- [ ] Add a focused config test before creating the config.
- [ ] Test must prove:
  - Unique identity.
  - Production checkpoint parent only.
  - Contract `physical_reality_v5_route_candidate_visibility`.
  - Observation dim 73 and action count 48.
  - `baseline_anchor_probability` is nonzero.
  - Output paths are fresh and do not point to V2 or V2.1.
  - `manual_run_safety.requires_skip_registry` is true.
  - `manual_run_safety.requires_disable_trace_logging` is true.
  - `torch_num_threads` is 2 and `torch_num_interop_threads` is 1.
  - Stage schedule sums to 250000.
  - No failed next-gen, failed V2, failed V2.1, routepremium failed 200k, aborted after_rewardfix, invalid 700k, v3/v4, or probe checkpoint is referenced.
- [ ] Create the config only after the test fails for missing config.
- [ ] Run config tests and compile.

### Task D: Run Fresh V2.2 250k Only After A-C Pass

Training command shape:

```powershell
python -m src.learn.train_joint_torch `
  --config configs\training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json `
  --output-dir models\checkpoints\joint_torch_v5_prod_stability_v2_2_250k_20260610 `
  --init-from-joint-checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt `
  --final-ppo-path models\checkpoints\joint_torch_v5_prod_stability_v2_2_250k_20260610\ppo_torch_joint_final_stability_v2_2_250k_20260610.pt `
  --final-dqn-path models\checkpoints\joint_torch_v5_prod_stability_v2_2_250k_20260610\dqn_torch_joint_final_stability_v2_2_250k_20260610.pt `
  --device cpu `
  --torch-num-threads 2 `
  --torch-num-interop-threads 1 `
  --skip-registry `
  --disable-trace-logging
```

After training, run a read-only post-training audit before eval.

### Task E: Evaluate And Gate V2.2 250k

Evaluation command shape:

```powershell
python -m src.eval.evaluate_real_world_scenarios `
  --checkpoint models\checkpoints\joint_torch_v5_prod_stability_v2_2_250k_20260610\joint_torch_latest.pt `
  --scenario-dir configs\eval_scenarios `
  --output-dir models\eval\joint_torch_v5_prod_stability_v2_2_250k_20260610_offline_scenarios `
  --episodes 20 `
  --seed 42 `
  --device cpu `
  --deterministic
```

Gate command:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_2_250k_20260610_offline_scenarios\scenario_summary.json `
  --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json `
  --candidate-label stability_v2_2_250k
```

Expected to continue:

- 8/8 scenario PASS.
- All hard blockers zero.
- Long-run gate exits 0 and decision PASS.
- No baseline service regression.
- No premium or route dispatch-success regression.
- No no-current/no-unassigned top-action explosion.
- Production comparison warnings reviewed.

## Stop Rule

If V2.2 250k fails, stop the autopilot with:

`AUTOPILOT_STOPPED_AFTER_FAILED_GATES`

The stop report must include:

- V2, V2.1, and V2.2 output dirs.
- Scenario comparison table versus production.
- Gate decisions and exit codes.
- Why each failed cycle failed.
- Confirmation that registry, active models, models.jsonl, production, baselines, DB, and the source production checkpoint were not mutated.

## Verification Blocker Observed During This Resume

In the current execution environment, `python`, `py`, and `git` were not available on PATH. The normal PowerShell sandbox runner also failed before command launch, so repository commands were run through the Node REPL where possible. Because Python was unavailable, this resume could not run unit tests, py_compile, training, offline eval, or the long-run gate checker.

Do not treat this plan as runtime verification. It is a root-cause and recovery plan based on current artifacts and static source inspection.

Final classification:

`STABILITY_V2_2_RECOVERY_PLAN_READY`
