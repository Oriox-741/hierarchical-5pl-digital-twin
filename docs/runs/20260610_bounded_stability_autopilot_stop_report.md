# Bounded Stability Autopilot Stop Report - 20260610

## Final Classification

AUTOPILOT_STOPPED_AFTER_FAILED_GATES

## Decision

The bounded stability autopilot is stopped. Stability V2.2 was the third allowed
250k design cycle and failed the 250k offline scenario gate and long-run gate.
No 500k or 1M continuation is authorized from V2.2.

Production remains the only approved parent and operational model:

`models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`

## Commands And Gates Run In This Resume

Python was available:

`python --version` -> `Python 3.12.9`

Pre-training tests and checks:

| Command | Result |
| --- | --- |
| `python -m unittest tests.act.test_reward_physics_contract -v` | PASS, 123 tests |
| `python -m unittest tests.learn.test_train_joint_curriculum -v` | PASS, 30 tests |
| `python -m unittest tests.learn.test_curriculum_config -v` | PASS, 64 tests |
| `python -m unittest tests.eval.test_long_run_gate -v` | PASS, 13 tests |
| `python -m py_compile ...` | PASS |
| Production long-run self-check | PASS, exit 0 |
| Stability V2 250k long-run gate | FAIL, exit 2, expected |
| Stability V2.1 250k long-run gate | FAIL, exit 2, expected |

V2.2 training:

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

Training completed with final checkpoint global step `250000`. Stderr was empty.

Post-training audit highlights:

| Check | Result |
| --- | --- |
| Final `joint_torch_latest.pt` exists | PASS |
| Final PPO/DQN paths exist | PASS |
| Training metrics rows | 62 |
| Last metrics global step | 250000 |
| Required action-quality telemetry fields | Present |
| Required lateness/timing telemetry fields | Present |
| Max blocked action rate in metrics | 0.0 |
| Max projected action rate in metrics | 0.0013020833333333333 |
| Final DQN action entropy | 3.2878296282131334 |
| Max top action concentration in metrics | 0.11290147569444445 |

V2.2 offline eval:

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

V2.2 long-run gate:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_2_250k_20260610_offline_scenarios\scenario_summary.json `
  --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json `
  --candidate-label stability_v2_2_250k
```

Result: `FAIL`, exit code `2`.

Fatal failures:

- `high_holding_cost`: service `0.826 < 0.900`; true lateness `0.183 > 0.080`.
- `lead_time_volatility`: true lateness `0.252 > 0.100`.
- `vehicle_scarcity_capacity_shock`: service `0.826 < 0.850`.
- `route_disruption_congestion`: no-current-work/no-unassigned total exploded from production `171` to `915`.
- `route_disruption_congestion`: action `41` no-current/no-unassigned top-action explosion, count `381`, production count `135`, top percentage `0.162`.

## Artifact Paths

Configs:

- `configs/training_joint_curriculum_v5_prod_stability_v2_250k_20260609.json`
- `configs/training_joint_curriculum_v5_prod_stability_v2_1_250k_20260609.json`
- `configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json`

Checkpoint dirs:

- `models/checkpoints/joint_torch_v5_prod_stability_v2_250k_20260609`
- `models/checkpoints/joint_torch_v5_prod_stability_v2_1_250k_20260609`
- `models/checkpoints/joint_torch_v5_prod_stability_v2_2_250k_20260610`

Eval dirs:

- `models/eval/joint_torch_v5_prod_stability_v2_250k_20260609_offline_scenarios`
- `models/eval/joint_torch_v5_prod_stability_v2_1_250k_20260609_offline_scenarios`
- `models/eval/joint_torch_v5_prod_stability_v2_2_250k_20260610_offline_scenarios`

Training logs:

- `tmp/stability_v2_2_250k_training_stdout.log`
- `tmp/stability_v2_2_250k_training_stderr.log`

## Scenario Comparison Versus Production

| Scenario | Production svc/late/dispatch/no-work | V2 svc/late/dispatch/no-work | V2.1 svc/late/dispatch/no-work | V2.2 svc/late/dispatch/no-work |
| --- | --- | --- | --- | --- |
| `baseline_normal` | PASS 0.938 / 0.000 / 1.000 / 0 | PASS 0.945 / 0.000 / 1.000 / 0 | FAIL 0.676 / 0.193 / 1.000 / 0 | PASS 0.944 / 0.000 / 1.000 / 1 |
| `demand_spike_volatility` | PASS 0.843 / 0.177 / 1.000 / 0 | PASS 0.949 / 0.000 / 1.000 / 0 | PASS 0.865 / 0.078 / 1.000 / 0 | PASS 0.878 / 0.003 / 1.000 / 0 |
| `high_holding_cost` | PASS 0.943 / 0.000 / 1.000 / 0 | FAIL 0.960 / 0.118 / 1.000 / 0 | FAIL 0.741 / 0.000 / 0.923 / 62 | FAIL 0.826 / 0.183 / 0.999 / 4 |
| `lead_time_volatility` | PASS 0.936 / 0.000 / 1.000 / 0 | FAIL 0.945 / 0.126 / 1.000 / 0 | FAIL 0.051 / 0.511 / 1.000 / 0 | FAIL 0.908 / 0.252 / 1.000 / 0 |
| `mixed_stress` | PASS 0.951 / 0.000 / 0.996 / 6 | PASS 0.933 / 0.009 / 1.000 / 0 | PASS 0.942 / 0.028 / 1.000 / 0 | PASS 0.878 / 0.010 / 1.000 / 0 |
| `premium_sla_pressure` | PASS 0.980 / 0.000 / 0.769 / 178 | FAIL 0.912 / 0.000 / 0.997 / 2 | FAIL 0.053 / 0.501 / 1.000 / 0 | PASS 0.932 / 0.002 / 1.000 / 1 |
| `route_disruption_congestion` | PASS 0.929 / 0.000 / 0.822 / 171 | FAIL 0.944 / 0.234 / 0.881 / 76 | PASS 0.950 / 0.000 / 0.888 / 93 | PASS 0.991 / 0.000 / 0.829 / 915 |
| `vehicle_scarcity_capacity_shock` | PASS 0.914 / 0.000 / 1.000 / 0 | PASS 0.944 / 0.032 / 1.000 / 0 | PASS 0.884 / 0.004 / 1.000 / 0 | FAIL 0.826 / 0.011 / 1.000 / 1 |

Pass counts:

| Candidate | Scenario pass count | Hard blockers | Long-run gate |
| --- | ---: | --- | --- |
| Production 200k parent | 8/8 | Zero | PASS, exit 0 |
| Stability V2 250k | 4/8 | Zero | FAIL, exit 2 |
| Stability V2.1 250k | 4/8 | Zero | FAIL, exit 2 |
| Stability V2.2 250k | 5/8 | Zero | FAIL, exit 2 |

## Root-Cause Summary

V2.2 improved the V2.1 premium SLA collapse and restored baseline behavior, but it
did not produce a clean stability candidate:

- Timing-sensitive failures remain in `high_holding_cost` and `lead_time_volatility`.
- `vehicle_scarcity_capacity_shock` regressed below the service threshold.
- Route disruption service and dispatch success were acceptable, but action-quality
  exposure was not: no-current/no-unassigned rose to `915`, triggering the strict
  long-run gate.
- The repeated pattern across V2, V2.1, and V2.2 is reward/curriculum balance
  instability, not hard-blocker leakage. All three 250k candidates kept hard
  blockers zero but failed production-grade scenario behavior.

Why the cycles failed:

- V2: fixed the next-gen no-current/no-unassigned explosion but became too late
  in high-holding, lead-time, premium, and route-disruption scenarios.
- V2.1: overcorrected into hold/high-resilience/secondary timing behavior and
  collapsed baseline, lead-time, premium, and high-holding service.
- V2.2: restored baseline and premium enough to pass scenario thresholds but
  retained lateness/service failures and reintroduced a route-disruption
  action-quality explosion.

## No-Mutation Confirmation

The V2.2 run used `--skip-registry` and `--disable-trace-logging`.

Protected path checks after training and eval:

| Protected artifact | Evidence |
| --- | --- |
| `models/registry/active_models.json` | SHA256 unchanged: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD` |
| `models/registry/models.jsonl` | SHA256 unchanged: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE` |
| Source production checkpoint | SHA256 unchanged: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E` |
| `models/production` | File count, total bytes, and max last-write timestamp unchanged |
| `models/baselines` | File count, total bytes, and max last-write timestamp unchanged |
| `db` | File count, total bytes, and max last-write timestamp unchanged |

No training or offline evaluation process remained active at the final audit.

## Files Changed Or Created

Existing V2.2 readiness changes present in this repository state:

- `configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json`
- `tests/learn/test_curriculum_config.py`

Artifacts created by this run:

- `models/checkpoints/joint_torch_v5_prod_stability_v2_2_250k_20260610`
- `models/eval/joint_torch_v5_prod_stability_v2_2_250k_20260610_offline_scenarios`
- `tmp/stability_v2_2_250k_training_stdout.log`
- `tmp/stability_v2_2_250k_training_stderr.log`
- `docs/runs/20260610_bounded_stability_autopilot_stop_report.md`

No registry, production, baseline, DB, or source production checkpoint mutation is
authorized or observed.
