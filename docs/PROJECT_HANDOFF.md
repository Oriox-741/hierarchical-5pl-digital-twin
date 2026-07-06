# Project Handoff

This file is the portable memory snapshot for continuing the Autonomous 5PL AI project from a fresh ChatGPT/Codex account or thread. Treat it as the first document to read before running, evaluating, repairing, or registering any model.

## Project Purpose

Autonomous 5PL AI is a logistics reinforcement-learning ecosystem for coordinating supply-chain decisions across inventory, dispatch, routing, fleet selection, and reorder policy. The current learning system is a joint PPO + DQN curriculum setup:

| Component | Responsibility |
|---|---|
| PPO policy | Continuous strategic/logistics control action. |
| DQN policy | Discrete tactical dispatch/action-family decision. |
| Joint curriculum | Trains across normal and stressed operational scenarios. |
| Environment/reward contract | Enforces physical-reality safeguards against fake logistics progress. |
| Scenario evaluator | Runs offline, inference-only real-world scenario checks before registry consideration. |

## Current Contract

The current valid model contract is:

| Field | Value |
|---|---|
| MDP contract | `physical_reality_v5_route_candidate_visibility` |
| Observation dim | `73` |
| Discrete action count | `48` |
| Model system | PPO + DQN joint curriculum |
| Legacy observation compatibility | First 57 old observation features preserved |
| New v5 route features | 16 appended route-candidate visibility features at indices 57..72 |

### V5 Appended Observation Features

Indices 57..72 are route-candidate visibility features made available pre-action:

| Index | Feature |
|---:|---|
| 57 | `shortest_candidate_score` |
| 58 | `low_congestion_candidate_score` |
| 59 | `high_resilience_candidate_score` |
| 60 | `shortest_candidate_score_gap` |
| 61 | `low_congestion_candidate_score_gap` |
| 62 | `high_resilience_candidate_score_gap` |
| 63 | `route_pressure_reliability_share` |
| 64 | `route_pressure_congestion_share` |
| 65 | `route_pressure_balance` |
| 66 | `route_alt_pressure_imbalance` |
| 67 | `shortest_candidate_near_best` |
| 68 | `shortest_secondary_safe_context` |
| 69 | `shortest_secondary_brittle_risk` |
| 70 | `secondary_fleet_feasible_dispatch_ratio` |
| 71 | `secondary_fleet_vehicle_availability_pressure` |
| 72 | `useful_dispatch_opportunity` |

Forbidden post-action telemetry must not be exposed in the pre-action observation.

## Latest Valid Full Training Run

This is the current safest artifact family for evaluation. It is not production-ready and is not registry-ready yet; it passed final audit with watches and should be evaluated offline next.

| Item | Path or value |
|---|---|
| Config | `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_blockers.json` |
| Environment id | `joint_curriculum_v5_clean_cold_start_1m_after_blockers` |
| Output dir | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers` |
| Final PPO | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/ppo_torch_joint_final_v5_clean_cold_start_1m_after_blockers.pt` |
| Final DQN | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/dqn_torch_joint_final_v5_clean_cold_start_1m_after_blockers.pt` |
| Joint latest checkpoint | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/joint_torch_latest.pt` |
| Final global step | `1,000,000` |
| Final audit classification | `FRESH_V5_CLEAN_1M_AFTER_BLOCKERS_PASS_WITH_WATCHES_READY_FOR_EVALUATION` |
| Final audit recommendation | `RUN_OFFLINE_SCENARIO_EVALUATION_NEXT` |

The final audit found target trace isolation clean: 20,000 rows for the final 1M env with trace interval 100, split into 10,000 PPO rows and 10,000 DQN rows. Final checkpoints and metrics were consistent at global step 1,000,000.

## Do Not Use

These artifacts are historical only and must not be resumed or used as current model candidates.

| Artifact | Status | Reason |
|---|---|---|
| `joint_curriculum_v5_clean_cold_start_1m` | Invalid historical run | Curriculum stayed `normal_v4` only, delivery-credit attribution leak existed, checkpoint was 700000 while metrics continued to 712704. |
| `models/checkpoints/joint_torch_v5_clean_cold_start_1m` | Invalid historical output dir | Old 700k/712704 mismatch run; do not resume. |
| Old v4/v3 checkpoints | Historical only | Wrong contract/dimension for current v5 route-candidate visibility work. |
| Production checkpoint folders | Do not modify | Registry/prod promotion requires explicit future approval after offline evaluation. |

## Completed Blockers And Repairs

1. Hold-route neutralization
   - `dispatch == hold` makes route labels semantically neutral.
   - `selected_route_effective_for_reward = -1` for hold.
   - Hold receives no route candidate credit, route adaptation credit, or in-flight delivery credit.

2. Action 8 fix
   - Action 8 decodes to hold + low_congestion + secondary fleet + no reorder.
   - Action 8 receives no route candidate credit, no low_congestion adaptation credit, no route adaptation credit, and no `dqn_delivery_credit`.

3. Shortest-dispatch action-context repair
   - Actions 24/25 are shortest + secondary fleet + none/conservative reorder dispatch.
   - The guard was split into safe-vs-brittle shortest-secondary logic.
   - Safe useful 24/25 can receive bounded credit only under safe moderate conditions.
   - Severe, failure, revisit, no-useful-work, fake, blocked, no_vehicle, already_assigned, service-degraded, and concentration cases remain guarded.

4. Observation candidate visibility v5
   - Observation dim is now 73.
   - Route candidate scores and safe/brittle shortest-secondary context are visible pre-action.
   - Forbidden post-action reward telemetry is excluded.

5. Curriculum blocker fix
   - `CurriculumSampler` now uses `total_timesteps/global_step`.
   - `stage_schedule` is no longer silently ignored.
   - Full clean curriculum exposes normal, route, premium, demand, lead-time, vehicle, holding, and mixed stress stages.

6. Delivery-credit leak fix
   - `dqn_delivery_credit` now requires current dispatch work.
   - Dispatch actions with zero current dispatched orders and zero dispatch success cannot receive `dqn_delivery_credit` from in-flight `delivered_delta`.
   - Hold/action8 remains blocked.
   - Valid current dispatch still gets delivery/progress/feasibility credit.

7. Resume safety
   - Same-output resume is guarded when metrics are ahead of checkpoint step.

## Current Status

The latest final 1M audit found:

| Check | Result |
|---|---|
| Full curriculum | Ran correctly |
| Trace isolation | Clean for target env |
| Checkpoint/metrics consistency | Clean at 1,000,000 |
| Fake dispatch credit | 0 |
| Customer revisit | 0 |
| No-current-work delivery-credit leak | 0 |
| Hold delivery leak | 0 |
| Action8 credit leak | 0 |
| Unsafe 24/25 credit leak | 0 |
| Blocked positive credit | 0 |
| NaN/Inf | 0 |

Main watches remain:

- `mixed_stress` service around 0.8759 and lateness around 0.0373.
- Route/mixed stress has low shortest share.
- high_resilience and low_congestion dominate under mixed stress.
- premium/SLA has mild secondary-fleet bias.

The current model is a pass with watches, ready for offline scenario evaluation. It is not registry-ready.

## Next Action

Run offline scenario evaluation against the latest valid v5 checkpoint family, starting from:

- `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/joint_torch_latest.pt`
- Scenario configs under `configs/eval_scenarios`
- Evaluator entry point: `src/eval/evaluate_real_world_scenarios.py`

Do not update registry until offline scenario evaluation passes.

## Operating Rules For Future Codex/ChatGPT Work

- Before changing behavior, read this file, `docs/EXPERIMENT_HISTORY.md`, and `docs/NEXT_STEPS.md`.
- Do not resume `joint_curriculum_v5_clean_cold_start_1m` or its output dir.
- Do not use old v3/v4 checkpoints for v5 evaluation or registry consideration.
- Do not mutate database rows, checkpoints, registry, baselines, or production checkpoint folders unless the user explicitly asks.
- Keep training, smoke, and live evaluation separate from read-only audit work.
- Use only the target env id when making a primary verdict about a run.
- Treat old diagnostic runs as references, not primary evidence for current model validity.
- Verify contract, observation dim, action count, checkpoint step, metrics step, trace cadence, and exploit gates before claiming a model is ready.
- Say "pass with watches" unless offline evaluation and registry-readiness checks explicitly justify stronger wording.
