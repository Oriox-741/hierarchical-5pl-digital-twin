# Bounded Stability Autopilot Resume Status - 20260610

## Objective

Continue the bounded stability autopilot toward either a clean experimental 1M candidate or a root-cause stop report, while preserving all no-mutation constraints from `docs/goals/bounded_stability_autopilot_goal.txt`.

## Current Artifact Audit

Existing completed 250k cycles:

| Cycle | Config | Checkpoint dir | Eval dir | Status |
| --- | --- | --- | --- | --- |
| Stability V2 250k | `configs/training_joint_curriculum_v5_prod_stability_v2_250k_20260609.json` | `models/checkpoints/joint_torch_v5_prod_stability_v2_250k_20260609` | `models/eval/joint_torch_v5_prod_stability_v2_250k_20260609_offline_scenarios` | Failed 250k gate per V2.2 recovery plan |
| Stability V2.1 250k | `configs/training_joint_curriculum_v5_prod_stability_v2_1_250k_20260609.json` | `models/checkpoints/joint_torch_v5_prod_stability_v2_1_250k_20260609` | `models/eval/joint_torch_v5_prod_stability_v2_1_250k_20260609_offline_scenarios` | Failed 250k gate per V2.2 recovery plan |

Fresh V2.2 readiness artifact created in this resume:

- `configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json`

No V2.2 checkpoint or eval directory exists yet:

- `models/checkpoints/joint_torch_v5_prod_stability_v2_2_250k_20260610`
- `models/eval/joint_torch_v5_prod_stability_v2_2_250k_20260610_offline_scenarios`

## Changes Made In This Resume

- Added a focused V2.2 config-readiness test in `tests/learn/test_curriculum_config.py`.
- Removed the stale V2.1 freshness assertion that required the V2.1 output directory to be absent even though the V2.1 250k run now exists.
- Tightened the V2.2 config test so it checks the output path is uniquely V2.2 without becoming stale after the eventual V2.2 run.
- Added `configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json`.

## V2.2 Static Validation

Static validation through the Node REPL confirmed:

- JSON parses successfully.
- Parent checkpoint is exactly `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`.
- Contract is `physical_reality_v5_route_candidate_visibility`.
- Observation dim is `73`.
- Discrete action count is `48`.
- `baseline_anchor_probability` is `0.10`.
- Stage schedule sums to `250000`.
- Manual run safety requires `--skip-registry` and `--disable-trace-logging`.
- Manual run safety uses `torch_num_threads: 2` and `torch_num_interop_threads: 1`.
- Config text does not reference failed nextgen, failed V2, failed V2.1, routepremium failed 200k, invalid 700k, v3/v4, or probe checkpoint fragments.

## Runtime Blocker

This environment still does not have `python`, `py`, or `git` on `PATH`.

Observed command probes:

| Command | Result |
| --- | --- |
| `python --version` | `ENOENT` |
| `py --version` | `ENOENT` |
| `where.exe python` | no match |
| `where.exe py` | no match |
| `where.exe git` | no match |
| explicit `<USER_HOME>\AppData\Local\Programs\Python\Python312\python.exe --version` | `ENOENT` / PowerShell access denied and not executable |

Because Python is unavailable, this resume did not run:

- `python -m unittest tests.act.test_reward_physics_contract -v`
- `python -m unittest tests.learn.test_train_joint_curriculum -v`
- `python -m unittest tests.learn.test_curriculum_config -v`
- `python -m py_compile ...`
- `python -m src.eval.check_long_run_gate ...`
- V2.2 250k training
- V2.2 offline eval

## Safe Next Commands Once Python Is Available

Run these before any V2.2 training:

```powershell
python -m unittest tests.act.test_reward_physics_contract -v
python -m unittest tests.learn.test_train_joint_curriculum -v
python -m unittest tests.learn.test_curriculum_config -v
python -m py_compile src\act\env_5pl.py src\learn\joint_metrics.py tests\act\test_reward_physics_contract.py tests\learn\test_train_joint_curriculum.py tests\learn\test_curriculum_config.py
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label production_self_check
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_250k_20260609_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label stability_v2_250k
python -m src.eval.check_long_run_gate --candidate-summary models\eval\joint_torch_v5_prod_stability_v2_1_250k_20260609_offline_scenarios\scenario_summary.json --production-summary models\eval\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios\scenario_summary.json --candidate-label stability_v2_1_250k
```

Only after those pass with expected gate behavior should V2.2 250k training be launched from production with `--skip-registry`, `--disable-trace-logging`, `--torch-num-threads 2`, and `--torch-num-interop-threads 1`.

## No-Mutation Confirmation For This Resume

This resume did not intentionally mutate:

- `models/registry`
- `models/registry/active_models.json`
- `models/registry/models.jsonl`
- `models/production`
- `models/baselines`
- `db`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`
- Any existing checkpoint or eval output directory

The goal is still active. A clean 1M candidate has not been reached.
