# Level 2 Architecture Decision: Teacher Retention vs Hierarchical Action Redesign

Date: 2026-06-11

## Decision

Recommended path: `RECOMMEND_HYBRID_TEACHER_THEN_HIERARCHICAL`

Final classification: `ARCHITECTURE_DECISION_HYBRID_PATH`

Use teacher-retention continuation as the next implementation step. Keep hierarchical/decomposed action redesign as the next architecture track if teacher retention cannot carry the exact-resume 250k behavior through 500k and 1M.

This is not a recommendation to run another blind reward/config V-cycle. The evidence now shows repeated action-pocket migration under a flat 48-action DQN: V2.2 high-resilience action 41/45, V2.3 low-congestion action 32, V2.4 high-resilience action 45, and V2.5 exact-resume lead-time action 33. Exact replay/RNG resume was necessary and fixed a real continuation bug, but V2.5 500k proves it is not sufficient.

## Non-Goals

- Do not train.
- Do not run offline eval.
- Do not create a new config.
- Do not update registry.
- Do not mutate `active_models.json` or `models.jsonl`.
- Do not mutate production, baselines, DB, or checkpoints.
- Do not start 250k, 500k, 1M, 3M, 5M, 10M, or 100M.
- Do not change the external contract: `physical_reality_v5_route_candidate_visibility`, observation dim 73, action count 48.
- Do not use failed V2/V2.1/V2.2/V2.3/V2.4/V2.5 checkpoints as parents.

## Evidence Table

| Run | Scenario PASS | Failed scenarios | Route svc/succ/no-work/top | Lead svc/succ/no-work/top | Premium svc/succ/no-work/missed/top | High svc/no-work | Mixed svc | Vehicle svc |
|---|---:|---|---|---|---|---|---|---|
| production | 8/8 | none | 0.929 / 0.822 / 171 / 40 dispatch-high_resilience-secondary-none 0.264 | 0.936 / 1.000 / 0 / 4 hold-shortest-primary-none 0.214 | 0.980 / 0.769 / 178 / 0.504 / 28 dispatch-shortest-primary-none 0.160 | 0.943 / 0 | 0.951 | 0.914 |
| V2 250k | 4/8 | high_holding, lead_time, premium, route | 0.944 / 0.881 / 76 / 16 hold-high_resilience-secondary-none 0.240 | 0.945 / 1.000 / 0 / 33 dispatch-low_congestion-secondary-conservative 0.241 | 0.912 / 0.997 / 2 / 0.479 / 4 hold-shortest-primary-none 0.166 | 0.960 / 0 | 0.933 | 0.944 |
| V2.1 250k | 4/8 | baseline, high_holding, lead_time, premium | 0.950 / 0.888 / 93 / 20 hold-high_resilience-primary-none 0.197 | 0.051 / 1.000 / 0 / 16 hold-high_resilience-secondary-none 0.397 | 0.053 / 1.000 / 0 / 0.847 / 16 hold-high_resilience-secondary-none 0.646 | 0.741 / 62 | 0.942 | 0.884 |
| V2.2 250k | 5/8 | high_holding, lead_time, vehicle | 0.991 / 0.829 / 915 / 9 hold-low_congestion-secondary-conservative 0.216 | 0.908 / 1.000 / 0 / 9 hold-low_congestion-secondary-conservative 0.372 | 0.932 / 1.000 / 1 / 0.513 / 9 hold-low_congestion-secondary-conservative 0.308 | 0.826 / 4 | 0.878 | 0.826 |
| V2.3 250k | 8/8 | none | 0.990 / 0.713 / 1697 / 32 dispatch-low_congestion-secondary-none 0.427 | 0.918 / 0.998 / 7 / 1 hold-shortest-secondary-conservative 0.294 | 0.982 / 1.000 / 0 / 0.208 / 20 hold-high_resilience-primary-none 0.432 | 0.973 / 2 | 0.820 | 0.949 |
| V2.4 250k semantics | 8/8 | none | 0.946 / 1.000 / 0 / 29 dispatch-shortest-primary-conservative 0.259 | 0.962 / 0.998 / 7 / 16 hold-high_resilience-secondary-none 0.268 | 0.988 / 0.999 / 2 / 0.128 / 20 hold-high_resilience-primary-none 0.466 | 0.990 / 0 | 0.936 | 0.942 |
| V2.4 300k diag | 5/8 | baseline, premium, route | 0.860 / 0.983 / 92 / 45 dispatch-high_resilience-primary-conservative 0.218 | 0.945 / 1.000 / 2 / 33 dispatch-low_congestion-secondary-conservative 0.144 | 0.913 / 1.000 / 1 / 0.361 / 29 dispatch-shortest-primary-conservative 0.137 | 0.957 / 10 | 0.912 | 0.882 |
| V2.4 400k diag | 2/8 | baseline, high_holding, lead_time, mixed, premium, vehicle | 0.981 / 0.858 / 743 / 13 hold-low_congestion-primary-conservative 0.145 | 0.576 / 1.000 / 0 / 21 hold-high_resilience-primary-conservative 0.504 | 0.356 / 1.000 / 0 / 0.985 / 3 hold-shortest-secondary-emergency 0.576 | 0.423 / 0 | 0.697 | 0.829 |
| V2.4 500k | 7/8 | premium | 0.964 / 0.622 / 2930 / 45 dispatch-high_resilience-primary-conservative 0.236 | 0.935 / 1.000 / 1 / 8 hold-low_congestion-secondary-none 0.229 | 0.811 / 0.992 / 40 / 0.528 / 21 hold-high_resilience-primary-conservative 0.171 | 0.995 / 125 | 0.941 | 0.970 |
| V2.5 250k | 8/8 | none | 0.946 / 1.000 / 0 / 29 dispatch-shortest-primary-conservative 0.259 | 0.962 / 0.998 / 7 / 16 hold-high_resilience-secondary-none 0.268 | 0.988 / 0.999 / 2 / 0.128 / 20 hold-high_resilience-primary-none 0.466 | 0.990 / 0 | 0.936 | 0.942 |
| V2.5 500k exact resume | 7/8 | route | 0.861 / 0.976 / 121 / 9 hold-low_congestion-secondary-conservative 0.258 | 0.985 / 0.856 / 740 / 33 dispatch-low_congestion-secondary-conservative 0.236 | 1.000 / 1.000 / 0 / 0.013 / 36 dispatch-low_congestion-primary-none 0.260 | 0.983 / 0 | 0.888 | 0.950 |

## Failure Pattern Table

| Cycle | What improved | What failed next | Dominant pocket |
|---|---|---|---|
| V2 | No-current/no-unassigned exposure greatly reduced from nextgen levels | Premium service, route lateness/action quality, lead-time/high-holding service | action 32 in route was already visible but not dominant |
| V2.1 | Tried timing/urgency repair | Baseline, premium, lead-time collapsed into hold/high-resilience/secondary behavior | action 16 hold-high_resilience-secondary-none |
| V2.2 | Baseline and premium recovered with anchor/interleaving | High-holding, lead-time, vehicle service; route action quality failed gate | action 41/45 high-resilience dispatch pockets; action 9 hold-low_congestion became broad attractor |
| V2.3 | 8/8 scenario thresholds passed | Long-run gate failed route action-quality | action 32 dispatch-low_congestion-secondary-none, 848 failed no-ops |
| V2.4 250k | Semantics-corrected gate passed | Legacy 500k continuation was not exact resume | old 250k lacked replay/RNG state |
| V2.4 500k | Some service remained strong | Premium service and route action quality failed | action 45 dispatch-high_resilience-primary-conservative, 1175 failed no-ops |
| V2.5 250k | Exact-resume-capable 250k passed | 500k exact resume still failed | no 250k failure |
| V2.5 500k | Premium, high-holding, route dispatch success recovered vs old V2.4 500k | Route service failed; lead-time no-current/no-unassigned exploded | lead-time action 33, plus action 27/45 no-work exposure; route action 9 hold service timing |

## Source Surface Audit

Relevant implementation facts:

- `src/act/discrete_action_mapper.py` defines the existing external 48-action product: 2 dispatch values x 3 route values x 2 fleet modes x 4 reorder modes. Action 33 decodes to dispatch + low_congestion + secondary_fleet + conservative; action 32 is dispatch + low_congestion + secondary_fleet + none; action 41 is dispatch + high_resilience + secondary_fleet + conservative; action 45 is dispatch + high_resilience + primary_fleet + conservative.
- `src/think/joint_policies.py` currently uses `DiscreteQNetwork(... nn.Linear(..., action_count))` and `argmax` over a flat 48-action Q vector. There is no internal factorized dispatch/route/fleet/reorder head.
- `src/learn/train_joint_torch.py` now supports exact resume: normal `--resume` restores model, PPO/DQN optimizer state, global step, DQN replay, and RNG; missing replay/RNG is rejected unless `--allow-empty-replay-resume` is explicitly used. `--init-from-joint-checkpoint` remains weights-only with fresh optimizer/replay/run state.
- `src/learn/joint_buffers.py` persists and validates DQN replay state with capacity, learning starts, observation dim, action count, cursor, size, total-added, observations, actions, rewards, next observations, and dones.
- `src/learn/curriculum.py` has baseline anchoring and deterministic stage sampling, but baseline anchoring is a sampling mechanism, not behavior retention. It can revisit normal states but cannot constrain Q drift on gate-critical state/action regions.
- `src/learn/joint_metrics.py` and `src/eval/scenario_metrics.py` already aggregate action quality by action id, action family, route, dispatch success, no-current/no-unassigned, reward component means, premium missed useful dispatch, and route candidate telemetry.
- `src/eval/long_run_gate.py` enforces production-comparison gates for dispatch success and no-current/no-unassigned top-action explosions. It correctly caught V2.3 action 32 and V2.5 action 33 despite scenario threshold PASS in those areas.
- `src/act/env_5pl.py` now has many useful-work gates and credit blockers, including no-current/no-unassigned penalties, low-congestion/high-resilience no-useful-work credit blockers, route candidate alignment gates, premium no-work blockers, and route mismatch penalties.

Interpretation: the environment and evaluator now have enough local reward and telemetry guards to identify bad pockets. The remaining issue is not missing a single obvious guard. The flat DQN can still move probability mass/Q preference into a new composite action pocket after each local repair.

## Root-Cause Matrix

| Hypothesis | Evidence | Status | Consequence |
|---|---|---|---|
| Legacy 500k failed because replay/RNG were not restored | V2.4 500k printed warm resume with empty replay; exact-resume patch added replay/RNG persistence and rejection of legacy resume | Confirmed but solved for new checkpoints | Necessary fix, not sufficient architecture |
| Exact resume alone solves 500k drift | V2.5 500k restored replay size 250k and RNG, then still failed route service and lead-time no-work action quality | Rejected | Need behavior retention or model-class change |
| Scenario evaluator semantics caused the current 500k fail | V2.4 250k semantics issue was fixed; V2.5 500k fail is route service and no-work action-quality, not stuck active horizon work | Rejected | Do not patch evaluator to pass |
| Reward patching can solve each pocket | Reward patches fixed specific pockets: premium no-work, high-resilience 41/45, low-congestion 32. But failures shifted to action 33 and route hold/service timing | Partially true, strategically weak | More reward tweaks risk action-pocket whack-a-mole |
| Flat 48-action DQN is prone to composite-action drift | Source uses one 48-logit Q head; evidence shows different composite actions become attractors across cycles | Likely | Hierarchical factorization is a robust long-term architecture |
| Teacher retention can stabilize current 250k -> 500k continuation | V2.5 250k is clean; V2.5 500k drift is a departure from both production and passing 250k behavior in gate-critical states | Likely | Least invasive next test; adds direct anti-forgetting pressure |
| Teacher retention will freeze production weaknesses | Production has known imperfections: premium missed useful dispatch 0.504, route no-work 171, route dispatch success 0.822 | Real risk | Use dual teachers and selective anchor masks, not global imitation |

## Approach A: Teacher Retention / Behavior Anchoring

Design sketch:

- Use two frozen teachers:
  - production teacher for baseline-normal, route/premium sanity, and operational fallback regions,
  - passing V2.5 250k teacher for improved premium, high-holding, route dispatch-success, and cleaned action-quality regions.
- Build fixed state banks from gate-critical scenario rollouts, preferably read-only deterministic banks:
  - baseline normal,
  - premium SLA useful-dispatch opportunities,
  - route disruption route-service/action-quality pockets,
  - lead-time volatility dispatch opportunity/no-work-risk states,
  - high-holding and vehicle scarcity retention states.
- Add DQN behavior retention loss on state-bank batches:
  - action match / cross entropy to selected teacher action,
  - optionally KL over teacher softmax(Q/tau),
  - optionally Q-margin retention: teacher-preferred action must remain above known bad pockets by margin.
- Add PPO behavior anchor only if PPO macro drift is implicated:
  - KL on continuous action mean/std against a teacher for state-bank observations,
  - keep it lower priority because current dominant failures are DQN action pockets.
- Use masks so retention does not imitate known production weaknesses:
  - do not force production's failed/no-op route dispatches,
  - prefer V2.5 250k teacher where it is strictly better than production,
  - exclude states where both teachers violate no-current/no-unassigned or scenario thresholds.
- Report retention telemetry:
  - teacher action agreement by scenario and action family,
  - KL/CE loss by scenario bank,
  - bad-pocket margin for action 32/33/41/45/46,
  - route service and lead-time no-work gates at periodic checkpoints.

Pros:

- Least invasive and fastest to test.
- Preserves external contract 73/48.
- Directly targets the observed problem: forgetting/drift away from a clean 250k policy.
- Can use existing checkpoints and existing telemetry.
- Does not require changing environment reward physics before proving retention need.

Cons and risks:

- Can freeze production weaknesses if applied globally.
- Requires careful teacher selection and masks.
- Adds another training objective that needs tests and independent review.
- May stabilize 500k but not scale to 10M/100M if flat-action entanglement remains.

Expected benefit:

- Most likely to solve the immediate V2.5 250k -> 500k drift because the failure is measured as departure from clean 250k behavior under exact resume.

## Approach B: Hierarchical / Decomposed DQN Action Redesign

Design sketch:

- Preserve external 0..47 action semantics and `DISCRETE_ACTION_COUNT=48`.
- Replace the flat 48-logit DQN internals with factorized heads:
  - dispatch/hold head: 2 choices,
  - route head: 3 choices,
  - fleet head: 2 choices,
  - reorder head: 4 choices.
- Output an external 0..47 action by composing selected head decisions with the existing mapper.
- Training options:
  - additive factorized Q: `Q(a)=Q_dispatch(d)+Q_route(r | d)+Q_fleet(m | d,r)+Q_reorder(o | d,r,m)`,
  - hierarchical policy: choose dispatch first; route/fleet/reorder heads matter only when dispatch is selected; hold ignores route/fleet/reorder labels for action selection,
  - masked auxiliary losses so no-work dispatch states train dispatch head away from dispatch, not just one composite id.
- Add action-family telemetry to report head-level entropy, agreement, and errors.

Pros:

- Best long-term fit for the logistics action structure.
- Directly addresses composite-action entanglement: a bad route/fleet/reorder label should not poison the dispatch decision, and hold should not learn meaningless route/fleet labels.
- Can preserve external action count and compatibility if implemented behind the same `select_dqn_action()` API.
- More robust path for 1M/3M/10M/100M if flat DQN keeps discovering new pockets.

Cons and risks:

- Much more invasive than teacher retention.
- Changes checkpoint architecture and likely requires new compatibility/versioning rules.
- Existing checkpoints cannot be exact-resumed into a new head architecture without a distillation or conversion stage.
- Requires substantial TDD across policy bundle, checkpoint load/save, runtime, evaluator, and training losses.
- Easy to create new bugs around action masking and Q composition.

Expected benefit:

- Highest long-horizon robustness, but not the fastest safe next step.

## Approach C: Continue Reward / Curriculum Patching

Assessment:

- Reward/curriculum patching remains useful for confirmed local reward leaks.
- It is no longer a sufficient primary strategy.
- V2 through V2.5 show a clear whack-a-mole pattern:
  - no-work guard improves one failure class,
  - baseline anchoring restores baseline/premium but creates/grows hold/action pockets,
  - high-resilience patch shifts route failure to low-congestion action 32,
  - low-congestion patch and exact resume shift later failure to lead-time action 33 and route service timing.
- Continuing this as the main strategy risks overfitting gate symptoms while the flat DQN keeps moving mass to a new composite action.

Use reward/curriculum changes only after teacher-retention or hierarchical diagnostics identify a specific, tested missing signal.

## Approach D: Hybrid Path

Recommended:

1. Implement teacher retention first, using production and V2.5 250k as selective teachers.
2. Use fixed state-bank diagnostics to prove whether retention preserves gate-critical behavior through 500k.
3. If teacher retention fails to pass 500k or creates teacher-freezing regressions, move to hierarchical action redesign.
4. If teacher retention passes 500k and 1M, keep hierarchical redesign as a planned robustness improvement before 3M/10M/100M.

This hybrid path fits the current evidence: the immediate problem is behavior drift from a known passing 250k policy, while the deeper architectural risk is flat composite-action entanglement.

## Required Questions

1. Which approach is most likely to solve the current 500k drift?
   - Teacher retention first. V2.5 500k is an exact-resume drift away from clean V2.5 250k behavior, so direct anti-forgetting is the most targeted fix.

2. Which approach is least invasive and fastest to test?
   - Teacher retention. It can preserve 73/48, existing env, existing evaluator, and mostly existing trainer structure.

3. Which approach is most robust for eventual 1M/3M/10M/100M?
   - Hierarchical/decomposed action redesign, or the hybrid path if teacher retention remains stable. Flat 48-way Q learning is the likely long-horizon bottleneck.

4. Does teacher retention risk freezing production weaknesses?
   - Yes. Production has premium missed useful dispatch and route no-work exposure. Mitigation: use selective dual teachers, prefer V2.5 250k where it improves production, exclude known bad teacher states/actions, and gate retained behavior against production and 250k.

5. Does hierarchical redesign require changing action count 48 or can it preserve external action semantics?
   - It can preserve external action semantics. The internal network can factor decisions into heads and still encode the selected tuple through `DiscreteActionMapper.encode()` to a 0..47 action.

6. Can hierarchical heads be trained while still outputting a 0..47 action?
   - Yes. The policy can compose head outputs into a 0..47 action for env/runtime compatibility. The harder question is loss design: either factorized Q composition or hierarchical conditional heads must be tested carefully.

7. What tests would be required before either approach?
   - Teacher retention tests:
     - teacher checkpoint loading is read-only and contract-checked,
     - fixed state-bank tensors validate observation dim 73 and finite values,
     - DQN retention loss is zero/low when student matches teacher,
     - retention loss rises when bad action 32/33/41/45/46 is forced,
     - teacher masks exclude known production-bad or no-useful-work states,
     - trainer metrics include teacher agreement/KL/CE by scenario bank,
     - `--init-from-joint-checkpoint` and `--resume` semantics remain unchanged.
   - Hierarchical tests:
     - action head encode/decode round-trips all 48 actions,
     - hold masks ignore route/fleet/reorder labels where intended,
     - composed action remains in [0,47],
     - checkpoint schema/versioning rejects incompatible flat/hierarchical loads unless explicitly converted,
     - DQN loss backpropagates to all required heads,
     - deterministic runtime still returns one legal 0..47 action,
     - evaluator/gate summaries remain unchanged.

8. What telemetry would prove success or failure?
   - Success:
     - 500k gate PASS exit 0,
     - no route service regression below threshold,
     - lead-time no-current/no-unassigned remains near 250k/production, not 740,
     - action 32/33/41/45/46 no-work pockets stay below long-run gate thresholds,
     - teacher agreement remains high on state banks without scenario service collapse,
     - PPO macro telemetry does not regress baseline/high-holding/vehicle behavior.
   - Failure:
     - new dominant no-work pocket appears,
     - teacher agreement is high but service is poor, indicating frozen bad teacher behavior,
     - teacher agreement collapses before 500k,
     - route service/premium/lead-time drift recurs despite low retention loss,
     - flat DQN Q margins favor a bad composite action under state-bank probes.

9. What is the next implementation step if teacher-retention is chosen?
   - Build a read-only fixed state-bank and teacher-retention diagnostics first. Then add a TDD trainer patch for selective DQN behavior retention using production and V2.5 250k teachers. Do not create a new training config until the retention patch and diagnostics are independently reviewed.

10. What is the next implementation step if hierarchical redesign is chosen?
    - Write an architecture plan and TDD policy prototype that preserves `DiscreteActionMapper` and external 48-action output, adds factorized heads behind `JointPolicyBundle`, and defines checkpoint migration/rejection rules. Do not reuse old flat checkpoints as exact-resume parents without an explicit distillation/conversion design.

## Recommended Next Implementation Prompt

```text
USESuperpowers:
- using-superpowers
- writing-plans
- test-driven-development
- systematic-debugging
- verification-before-completion
- requesting-code-review

Level 2 teacher-retention design implementation plan only. Do not train, do not run offline eval, do not create a new config, and do not mutate production/registry/baselines/DB/checkpoints.

Objective:
Implement read-only teacher-retention diagnostics and a TDD trainer patch proposal that can anchor V2.5 exact-resume continuation without changing external contract physical_reality_v5_route_candidate_visibility, obs 73, action 48.

Requirements:
- Use production teacher and V2.5 250k teacher selectively.
- Build fixed gate-critical state-bank format/tests.
- Add DQN teacher-retention loss tests before implementation.
- Add metrics for teacher action agreement, KL/CE, bad-pocket Q margin for actions 32/33/41/45/46, and scenario-bank summaries.
- Preserve --resume exact replay/RNG behavior and --init-from-joint-checkpoint weights-only behavior.
- Do not create V2.6 config until patch is independently approved.
```

## Independent Review

One read-only reviewer inspected the architecture decision report, dashboard update, V2.5 gate JSONs, V2.5 architecture evidence, scenario-summary extraction, and relevant source surfaces.

Reviewer verdict:

```text
REVIEW_APPROVED
```

Reviewer summary: the hybrid recommendation is supported by the evidence; the report satisfies the requested scope; source/eval cross-checks match the decision; read-only constraints were honored.

## Protected No-Mutation Proof

Checked before report work:

- `models/registry/active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- Source production checkpoint SHA256: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
- `models/production`: 4 files, 15040339 bytes
- `models/baselines`: 2692 files, 7392576274 bytes
- `db`: 7 files, 28330 bytes

Only documentation files were intended to be mutated by this architecture decision task.
