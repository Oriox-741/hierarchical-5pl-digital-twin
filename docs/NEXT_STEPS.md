# Next Steps

This file describes the next safe actions for the Autonomous 5PL AI project after the fresh v5 clean 1M after-blockers run.

## Immediate Next Task

Run offline scenario evaluation for the latest valid v5 clean 1M after-blockers checkpoint family.

Final audit recommendation:

- `RUN_OFFLINE_SCENARIO_EVALUATION_NEXT`

Do not update the registry before this evaluation passes. Do not treat the model as production-ready. The correct status is pass with watches, ready for offline evaluation.

## Target Artifacts To Evaluate

Use the latest valid after-blockers artifacts only:

| Purpose | Path or value |
|---|---|
| Config | `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_blockers.json` |
| Environment id | `joint_curriculum_v5_clean_cold_start_1m_after_blockers` |
| Output dir | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers` |
| Preferred evaluation checkpoint | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/joint_torch_latest.pt` |
| Final PPO | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/ppo_torch_joint_final_v5_clean_cold_start_1m_after_blockers.pt` |
| Final DQN | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/dqn_torch_joint_final_v5_clean_cold_start_1m_after_blockers.pt` |
| Contract | `physical_reality_v5_route_candidate_visibility` |
| Observation dim | `73` |
| Discrete action count | `48` |

Use the joint checkpoint for the current evaluator unless the evaluator is intentionally extended to support split PPO/DQN checkpoint paths.

## Evaluation Entry Point

Repository evaluator:

- `src/eval/evaluate_real_world_scenarios.py`

Scenario config directory:

- `configs/eval_scenarios`

Known scenario config files:

| Scenario family | Config |
|---|---|
| normal_v4 | `configs/eval_scenarios/baseline_normal.json` |
| route_disruption | `configs/eval_scenarios/route_disruption_congestion.json` |
| premium_sla | `configs/eval_scenarios/premium_sla_pressure.json` |
| demand_spike | `configs/eval_scenarios/demand_spike_volatility.json` |
| lead_time_delay | `configs/eval_scenarios/lead_time_volatility.json` |
| vehicle_scarcity | `configs/eval_scenarios/vehicle_scarcity_capacity_shock.json` |
| high_holding_cost | `configs/eval_scenarios/high_holding_cost.json` |
| mixed_stress | `configs/eval_scenarios/mixed_stress.json` |

Suggested offline evaluation command, to be run only when the user explicitly asks for evaluation:

```powershell
python -m src.eval.evaluate_real_world_scenarios `
  --checkpoint models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/joint_torch_latest.pt `
  --scenario-dir configs/eval_scenarios `
  --output-dir models/eval/joint_torch_v5_clean_cold_start_1m_after_blockers_offline_scenarios
```

If a future task is audit-only, do not run this command. It performs offline inference evaluation and writes evaluation outputs.

## Required Evaluation Coverage

The offline evaluation must check all of these scenario families:

- `normal_v4`
- `route_disruption`
- `premium_sla`
- `demand_spike`
- `lead_time_delay`
- `vehicle_scarcity`
- `high_holding_cost`
- `mixed_stress`

For each family, record at minimum:

| Area | What to review |
|---|---|
| Service/lateness | Service level, lateness pressure, SLA failures, collapse warnings. |
| Route behavior | shortest, low_congestion, high_resilience route mix, candidate/guard telemetry where available. |
| Fleet behavior | primary vs secondary fleet use, no_vehicle exposure, already_assigned exposure. |
| Dispatch behavior | dispatch/hold ratio, current dispatch success, no-current-work rows. |
| Reorder behavior | none/conservative/aggressive/emergency distribution and emergency overuse. |
| Exploit gates | Hard blocker checks listed below. |
| Stability | NaN/Inf, blocked/projected actions, suspicious action-family concentration. |

## Watch Items

Carry these watches into offline evaluation:

| Watch | Why it matters |
|---|---|
| mixed_stress service/lateness | Final 1M audit saw mixed_stress service around 0.8759 and lateness around 0.0373. Evaluation must decide whether this is acceptable under combined stress. |
| route/mixed shortest low | Final route/mixed stress behavior kept shortest alive but low, especially late. Confirm it is contextually justified rather than reward suppression. |
| high_resilience/low_congestion dominance | high_resilience and low_congestion dominated under mixed stress. Confirm it is operationally plausible and not a new attractor. |
| premium secondary-fleet bias | Premium/SLA had mild secondary-fleet bias. Confirm SLA behavior remains healthy and primary fleet is used when justified. |

## Hard Blocker Checks

Offline evaluation or any future audit must fail or stop for review if any of these regress:

| Blocker | Expected value |
|---|---|
| fake dispatch credit | 0 |
| customer revisit exploit | 0 |
| no-current-work delivery credit | 0 |
| hold delivery leak | 0 |
| action8 credit leak | 0 |
| unsafe 24/25 credit | 0 |
| route-failure zero-success positive credit | 0 |
| NaN/Inf in metrics or reward components | 0 |

Also verify:

- Hold route labels remain neutral.
- Action 8 remains route-neutral and delivery-credit blocked.
- 24/25 shortest-secondary credit is only allowed in safe useful contexts.
- Valid current dispatch still receives delivery/progress/feasibility credit.
- Observation dim remains 73.
- Discrete action count remains 48.
- Contract remains `physical_reality_v5_route_candidate_visibility`.

## Unsafe Actions

Do not do any of the following unless the user explicitly requests it and the task scope allows it:

- Do not update registry.
- Do not register the after-blockers model before offline evaluation passes.
- Do not resume the old invalid `joint_curriculum_v5_clean_cold_start_1m` output.
- Do not use `models/checkpoints/joint_torch_v5_clean_cold_start_1m` except as historical evidence.
- Do not use old v4/v3 checkpoints for v5 evaluation or registry candidacy.
- Do not mutate DB rows during audits.
- Do not truncate, delete, purge, or clean old outputs.
- Do not modify models/baselines or production checkpoint folders during documentation or audit tasks.

## Decision Path

Use this decision flow after offline evaluation:

| Evaluation outcome | Next decision |
|---|---|
| All scenarios pass and hard blockers remain 0 | Prepare registry-candidate review; still verify checkpoint metadata and outputs first. |
| Scenarios pass but watches remain material | Run targeted read-only analysis and decide whether a small diagnostic or repair is needed. |
| Any hard blocker appears | Stop; classify as invalid exploit or failed repair; do not register. |
| Mixed stress fails service/lateness | Investigate stress calibration, route/fleet/reorder balance, and whether another targeted diagnostic is needed before rerun. |
| Route dominance is implausible | Investigate route candidate scoring, route reward pressure, and action-family concentration before any fresh 1M. |

## Starting Context For New Threads

When opening a new Codex/ChatGPT thread, provide or point to:

1. `docs/PROJECT_HANDOFF.md`
2. `docs/EXPERIMENT_HISTORY.md`
3. `docs/NEXT_STEPS.md`

Then ask the new thread to verify current filesystem state before running any evaluation or mutation. The new thread should not rely on memory alone.
