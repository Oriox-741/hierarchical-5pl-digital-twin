# Experiment History

This file records the important experiment sequence for the Autonomous 5PL AI project. It is intended for a new Codex/ChatGPT thread with no prior chat memory.

## Chronological Summary

| Order | Area | Main artifact/env | Outcome |
|---:|---|---|---|
| 1 | Earlier physical-reality and route reward repairs | v4 route-disruption diagnostic family | Repaired several route reward and dispatch feasibility issues, but shortest-dispatch behavior remained fragile. |
| 2 | Structural route reward normalization | `joint_curriculum_v4_fresh_route_disruption_post_architecture_fix_50k_diagnostic` | Added route candidate scores, guarded selected-route candidate credit, safe useful shortest candidate credit, and high_resilience passive-credit dampening. |
| 3 | Hold-route neutralization | `joint_curriculum_v4_fresh_route_disruption_post_hold_neutralization_50k_diagnostic` | Made route labels neutral when dispatch is hold; action 8 no longer receives route or delivery credit. |
| 4 | Shortest-dispatch context repair | `joint_curriculum_v4_fresh_route_disruption_post_shortest_dispatch_context_50k_diagnostic` | Confirmed actions 24/25 decode correctly, but learned comparison still favored 32/33 because safe 24/25 facts were not visible pre-action. |
| 5 | V5 observation candidate visibility | `physical_reality_v5_route_candidate_visibility` | Added 16 route-candidate visibility features at indices 57..72; observation dim became 73. |
| 6 | V5 route diagnostic 50k | `joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic` | Route-only diagnostic; useful for route behavior reference, not a final model candidate. |
| 7 | V5 route diagnostic 100k | `joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic` | Route-only diagnostic reference; dispatch-only shortest was about 17.16%, low_congestion about 63.78%, action 32/33 about 33.25%, action 8 about 3.8%. |
| 8 | Invalid v5 clean 1M attempt | `joint_curriculum_v5_clean_cold_start_1m` | Rejected. Curriculum stayed `normal_v4` only, delivery-credit attribution leak existed, and checkpoint/metrics were inconsistent. |
| 9 | Blocker repairs after invalid run | Source/config/test changes around curriculum, delivery credit, and resume safety | Fixed full-curriculum scheduling, current-dispatch-work delivery attribution, and metrics-ahead resume guard. |
| 10 | Fresh v5 clean 1M after blockers | `joint_curriculum_v5_clean_cold_start_1m_after_blockers` | Passed final audit with watches and is ready for offline scenario evaluation. |

## Route Reward Repair Attempts

The route work began from repeated route-disruption diagnostics showing that action decoding was not the primary problem. Actions 24/25 decoded correctly as shortest + secondary fleet + none/conservative dispatch, but learned reward comparison still favored low_congestion actions 32/33 under route-disruption conditions.

Key findings:

- low_congestion received normal route adaptation credit.
- shortest paid route risk penalty and only received relief/candidate credit through narrow useful-work and guard gates.
- safe 24/25 examples were sparse and did not survive exploration decay well.
- DQN used epsilon-greedy over 48 actions with uniform replay.
- Replay had no action-family balancing metadata.
- A major observation-contract issue was found: "safe 24/25 allowed" facts were post-action reward telemetry, not visible in the 57D pre-action observation.

This led to the v5 observation-contract change rather than another small reward-only patch.

## Structural Route-Side Normalization

Representative environment id:

- `joint_curriculum_v4_fresh_route_disruption_post_architecture_fix_50k_diagnostic`

Reported decision:

- `ROUTE_ARCHITECTURE_FIX_IMPLEMENTED`

What changed:

- Added route candidate scoring.
- Added guarded selected-route candidate credit.
- Allowed safe useful shortest to receive bounded candidate credit.
- Added high_resilience passive-credit dampening.
- Preserved observation dim, action count, production configs, checkpoints, DB schema, registry, baselines, and checkpoint folders at that stage.

Purpose:

- Normalize route reward geometry without changing the action space.
- Make shortest viable only when actually useful and safe.
- Keep severe/failure/revisit/fake/no-useful-work cases guarded.

## Hold-Route Neutralization

Representative environment id:

- `joint_curriculum_v4_fresh_route_disruption_post_hold_neutralization_50k_diagnostic`

Contract:

- For `dispatch == hold`, route labels are semantically inert.
- `selected_route_effective_for_reward = -1`.
- Route candidate credit is blocked.
- Route adaptation credit is blocked.
- Shortest relief is blocked.
- In-flight `delivered_delta` does not count as route-useful work.
- hold + shortest, hold + low_congestion, and hold + high_resilience are equivalent with respect to route reward.

Action 8:

- Action 8 = hold + low_congestion + secondary fleet + no reorder.
- It must receive no route candidate credit, no low_congestion adaptation advantage, no route adaptation credit, and no DQN delivery credit.

Result:

- Legacy sanity-check expectations were updated to match the new hold-route neutralization contract.
- Dispatch actions still keep route labels meaningful.

## Shortest-Dispatch Context Repair

Representative environment id:

- `joint_curriculum_v4_fresh_route_disruption_post_shortest_dispatch_context_50k_diagnostic`

Key diagnosis:

- Actions 24/25 decoded correctly.
- The previous shortest-dispatch repair failed because the safe/brittle context needed by the DQN was only visible as post-action reward telemetry.
- The 57D observation did not expose enough pre-action route-candidate viability information.

Decision:

- Final recommendation was to fix observation candidate features next.
- Implementation posture was observation-contract change required.

## V5 Observation Candidate Visibility

Current contract:

- `physical_reality_v5_route_candidate_visibility`

Observation change:

- Previous first 57 features preserved.
- 16 route-candidate visibility features appended at indices 57..72.
- Observation dim became 73.
- Discrete action count remained 48.

Goal:

- Make route candidate scores, candidate gaps, route pressure shares, shortest near-best state, shortest secondary safe/brittle context, secondary fleet feasibility, vehicle availability pressure, and useful dispatch opportunity visible before action selection.

Important guard:

- Do not expose forbidden post-action telemetry in the pre-action observation.

## V5 50k Route Diagnostic

Environment id:

- `joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic`

Context:

- Route-only diagnostic after v5 observation candidate visibility.
- Historical reference, not a registry candidate.

Observed route behavior from later comparison audit:

| Metric | Result |
|---|---:|
| DQN trace rows | 1,000 |
| Stage | 100% `route_disruption` |
| Dispatch rate | 52.10% |
| Dispatch-only low_congestion | 67.2% |
| Dispatch-only high_resilience | 25.3% |
| Dispatch-only shortest | 7.5% |
| Action 8 share | 8.50% |
| Action 24/25 share | 0.60% |
| Action 32/33 share | 33.00% |
| Action 40/41 share | 11.20% |

Decision:

- Useful route-disruption reference only.
- Not a final model candidate and not a substitute for full-curriculum evaluation.

## V5 100k Route Diagnostic

Environment id:

- `joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic`

Context:

- Longer route-only diagnostic after v5 observation candidate visibility.
- Used as a reference for route behavior during final 1M audit.

Observed route behavior:

| Metric | Result |
|---|---:|
| DQN trace rows | 2,000 |
| Stage | 100% `route_disruption` |
| Dispatch rate | 55.35% |
| Dispatch-only low_congestion | 63.8% |
| Dispatch-only high_resilience | 19.1% |
| Dispatch-only shortest | 17.2% |
| Action 8 share | 3.80% |
| Action 24/25 share | 1.00% |
| Action 32/33 share | 33.25% |
| Action 40/41 share | 8.55% |

Decision:

- Served as a route-disruption reference.
- Did not replace full clean curriculum training.

## Invalid V5 Clean 1M Attempt

Invalid config/env:

- `joint_curriculum_v5_clean_cold_start_1m`

Invalid output dir:

- `models/checkpoints/joint_torch_v5_clean_cold_start_1m`

Reason rejected:

| Blocker | Detail |
|---|---|
| Curriculum blocker | Curriculum stayed `normal_v4` only instead of running full scheduled curriculum. |
| Delivery-credit leak | `dqn_delivery_credit` could be attributed to dispatch actions with zero current dispatched orders or zero dispatch success via in-flight `delivered_delta`. |
| Resume/checkpoint inconsistency | Checkpoint was at 700000 while metrics continued to 712704. |

Decision:

- Historical only.
- Do not resume.
- Do not use for evaluation, registry, smoke, or production.

## Blockers Fixed After Invalid Run

1. Curriculum scheduling
   - `CurriculumSampler` now uses `total_timesteps/global_step`.
   - `stage_schedule` no longer silently stays at `normal_v4`.
   - Full clean curriculum exposes normal, route, premium, demand, lead-time, vehicle, holding, and mixed stages.

2. Delivery-credit attribution
   - `dqn_delivery_credit` requires current dispatch work.
   - Zero-current-work dispatch cannot receive delivery credit from in-flight completions.
   - Hold/action8 remain blocked.
   - Valid current dispatch still receives delivery/progress/feasibility credit.

3. Resume safety
   - Same-output resume is guarded when metrics are ahead of checkpoint step.

## Fresh V5 Clean 1M After Blockers

Current target:

| Item | Path or value |
|---|---|
| Config | `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_blockers.json` |
| Environment id | `joint_curriculum_v5_clean_cold_start_1m_after_blockers` |
| Output dir | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers` |
| Final PPO | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/ppo_torch_joint_final_v5_clean_cold_start_1m_after_blockers.pt` |
| Final DQN | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/dqn_torch_joint_final_v5_clean_cold_start_1m_after_blockers.pt` |
| Joint latest | `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/joint_torch_latest.pt` |
| Contract | `physical_reality_v5_route_candidate_visibility` |
| Observation dim | `73` |
| Discrete action count | `48` |
| Final global step | `1,000,000` |

Final audit classification:

- `FRESH_V5_CLEAN_1M_AFTER_BLOCKERS_PASS_WITH_WATCHES_READY_FOR_EVALUATION`

Final recommendation:

- `RUN_OFFLINE_SCENARIO_EVALUATION_NEXT`

Final audit highlights:

- Full curriculum ran correctly.
- Trace isolation clean: 20,000 target rows for the final 1M env with trace interval 100.
- Final checkpoints and metrics were consistent at 1,000,000.
- Exploit checks were clean:
  - fake dispatch credit = 0
  - customer revisit = 0
  - no-current-work delivery-credit leak = 0
  - hold delivery leak = 0
  - action8 credit leak = 0
  - unsafe 24/25 credit leak = 0
  - blocked positive credit = 0
  - NaN/Inf = 0

Remaining watches:

- `mixed_stress` service around 0.8759 and lateness around 0.0373.
- Route/mixed stress has low shortest share.
- high_resilience and low_congestion dominate under mixed stress.
- premium/SLA has mild secondary-fleet bias.

Decision:

- Ready for offline scenario evaluation.
- Not registry-ready.
- No fresh 1M rerun is indicated by the final audit alone.
