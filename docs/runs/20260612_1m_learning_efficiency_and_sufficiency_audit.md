# 1M Learning-Efficiency and Sufficiency Audit

Date: 2026-06-12

Final classification: `ONE_M_LEARNING_EFFICIENCY_AUDIT_READY`

Status: read-only analysis plus documentation. No training, offline eval, long-run gate, dataset download, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output edit, training config creation, or 1.5M/2M/3M/5M/10M/100M run was performed.

## Executive Answer

The hierarchical v1 1M model did not learn a full logistics company from scratch in 1M isolated steps. It learned a bounded 5PL simulator policy under a specific contract, after several layers of prior project work had already made the problem much easier and better supervised:

- a production flat PPO+DQN parent already existed and passed 8/8 scenarios,
- reward and hard-blocker semantics had been repaired across earlier cycles,
- the v5 observation/action contract was fixed at obs 73 and external action 48,
- the hierarchical DQN decomposed the 48 flat action pockets into dispatch/route/fleet/reorder heads,
- flat-teacher distillation gave the hierarchical DQN a behavior-preserving warm start,
- 500k and 1M were exact resumes with DQN replay/RNG restored,
- the curriculum repeatedly exposed the same finite scenario family,
- the gates were strong enough to catch many previous bad solutions.

So yes, 1M is genuinely sufficient evidence for this simulator/eval contract because it passed the defined project gates and equal-budget residual-watch assurance. It does not prove real-world autonomous logistics readiness without company data, real TMS/WMS integration, real carrier economics, and monitored production telemetry.

## Short Answer For The User

It learned "so much" in 1M because the final run was not blank-slate discovery. Think of it as the last successful architecture-and-stability pass over a problem that had already been shaped by months of debugging inside the repo: the simulator was bounded, the action space was small and structured, a working production teacher existed, and the gates forced useful behavior instead of only high reward. The 1M steps were enough to adapt and stabilize a warm-started hierarchical policy, not to discover all logistics from first principles.

The difference between simulator production-ready and real-world autonomous is scope. Simulator production-ready means: under the v5 digital-twin contract, fixed scenarios, obs 73, action 48, known hard blockers, and long-run gates, the model is the best approved artifact. Real-world autonomous means: the same decisions remain safe and economically valid under real order streams, carrier constraints, inventory costs, failure reasons, and integration edge cases. That part still needs company data and monitoring.

## Timeline Of Cumulative Learning Before The Final 1M

| Phase | What was learned or repaired before hierarchical 1M | Why it matters for 1M sufficiency |
|---|---|---|
| Clean rewardfix/perfclean parent | Earlier rewardfix and perfclean work created a clean v5 parent path. | The final model inherited a stabilized simulator/reward surface rather than learning through stale reward identities. |
| Balanced-retention production parent | `joint_torch_v5_balanced_retention_ft_200k_20260608` passed 8/8 and became the old production comparator. | The 250k hierarchical run started from an already useful production PPO/DQN policy, not random logistics behavior. |
| Failed nextgen/V2 cycles | Nextgen 1M, V2/V2.1/V2.2/V2.3/V2.4/V2.5 exposed route, premium, lead-time, no-work, and action-pocket drift failures. | The later gate suite and architecture decision were informed by concrete failure modes, not blind tuning. |
| Exact resume fix | Normal `--resume` now restores model state, PPO/DQN optimizer state, global step, DQN replay, and RNG; legacy missing replay/RNG resumes are rejected unless explicitly allowed. | 500k and 1M continuation could preserve the 250k/500k distribution instead of drifting through empty-replay continuation. |
| Teacher retention attempt | Teacher-retention 250k failed, showing simple behavior anchoring was insufficient. | This eliminated a tempting but weaker explanation and justified the hierarchical architecture track. |
| Hierarchical architecture | `hierarchical_v1` preserves external action 48 but internally decomposes DQN Q values into dispatch, route, fleet mode, and reorder heads. | This reduces flat action-pocket entanglement while keeping the runtime/evaluator contract unchanged. |
| Flat-teacher distillation | The first 250k hierarchical model used `flat_teacher_distillation_v1` from the protected flat production checkpoint, with 4096 sampled observations and 512 supervised distillation steps. | The hierarchical DQN began near a known working policy rather than relearning all action semantics through RL. |
| Exact-resume ladder | 250k used `--init-from-joint-checkpoint` from production as a warm start; 500k resumed exactly from 250k; 1M resumed exactly from 500k. | The final 1M is cumulative: 250k + exact 500k + exact 1M, not three unrelated runs. |
| Promotion and assurance | 1M passed 8/8 scenarios, hard blockers zero, long-run gate PASS, equal-budget residual-watch gate PASS, and ops bundle clean. | The acceptance evidence is stronger than a single aggregate reward curve. |

## What "1M Steps" Means Here

In this project, `1M` means the trainer global step reached `1,000,000` simulator decision steps under `shared_global_parameters.total_timesteps=1000000`. Each step is a 5-minute decision interval by config (`decision_interval_seconds=300`) and the trainer loop increments `state.global_step` once per environment step.

Important details:

- The 250k config has 15 curriculum stages summing to 250,000 steps.
- The 500k config has 30 curriculum stages summing to 500,000 steps.
- The 1M config has 60 curriculum stages summing to 1,000,000 steps.
- `max_steps=288`, so one episode horizon corresponds to one simulated day of 5-minute decisions.
- The DQN replay buffer capacity is 500,000 transitions.
- The 250k checkpoint recorded replay size/position/total-added `250000 / 250000 / 250000`.
- The 500k checkpoint recorded replay size/position/total-added `500000 / 0 / 500000`.
- The 1M checkpoint recorded replay size/position/total-added `500000 / 0 / 1000000`.
- The 500k trainer startup log restored replay size 250k and RNG.
- The 1M trainer startup log restored replay size 500k and RNG.

The final 1M therefore represents cumulative exposure through a repeated scenario curriculum with exact replay/RNG continuity after the initial hierarchical warm start.

## Factors That Made 1M Sufficient

| Factor | Evidence | Effect |
|---|---|---|
| Bounded observation space | Observation dimension is fixed at 73. | The model does not infer arbitrary logistics state; it learns over a curated digital-twin feature vector. |
| Bounded external action space | DQN action count is 48: 2 dispatch x 3 route x 2 fleet x 4 reorder. | The tactical search space is small enough for DQN-style learning and gate-level action diagnostics. |
| Hierarchical DQN factorization | `HierarchicalDQNQNetwork` composes dispatch + reorder + dispatch-mask(route + mode) into external Q values. | Reduces flat action-pocket drift while keeping action 0..47 compatibility. |
| Production warm start | 250k initialized from `joint_torch_v5_balanced_retention_ft_200k_20260608`. | Much of the broad service/dispatch behavior came from a passing parent. |
| Flat-teacher distillation | 4096 sampled observations, 512 distillation steps, bounded loss, teacher action matching telemetry. | Transfers flat production DQN behavior into hierarchical heads before RL begins. |
| PPO loaded from teacher | Hierarchical initialization loads compatible PPO state from the flat teacher. | Strategic continuous behavior is not reset to random. |
| Dense simulator feedback | `env_5pl` exposes service, lateness, inventory, route, dispatch, no-work, and feasibility/reward components each step. | The model gets dense feedback on the exact simulator dynamics being gated. |
| Curriculum schedule | 1M config repeats normal, premium SLA, lead-time delay, high holding, vehicle scarcity, route disruption, demand spike, and mixed stress. | The final model sees all gate-relevant regimes repeatedly rather than relying on rare random exposure. |
| Exact resume | 500k/1M require restored replay/RNG and disallow legacy warm resume. | Reduces continuation instability that broke older 250k -> 500k ladders. |
| Strong gates | Long-run gate enforces scenario thresholds, hard blockers, dispatch-success comparisons, baseline service, and no-work top-action explosions. | Bad candidates with superficially good service or zero hard blockers were rejected in earlier cycles. |

## Why Blind 10M Is Not Automatically Better

| Reason | Project evidence | Consequence |
|---|---|---|
| More steps can amplify action-pocket drift | Nextgen 1M, V2.4 500k, and V2.5 500k all showed longer continuation can move mass into bad pockets. | Scaling steps without a trigger can make behavior worse. |
| Exact resume was necessary but not sufficient | V2.5 exact-resume 500k still failed route/lead-time behavior. | Stability requires architecture and gates, not just longer runs. |
| Teacher retention failed at 250k | The retention branch did not preserve passing behavior even at the first rung. | Simple anti-forgetting did not solve the long-horizon issue. |
| Finite scenario family can be overfit | The gate suite is strong but still defined by eight scenario files and known thresholds. | Longer training may optimize the simulator/gate distribution more narrowly. |
| Residual watches remain | Action 32 concentration in route disruption and action 24 concentration in mixed stress are accepted watches, not proofs of universal optimality. | More training should be trigger-driven, not automatic. |
| Real-world gaps remain | Company fleet economics, reorder economics, and integration failures are unvalidated. | A longer simulator run cannot prove facts the simulator does not contain. |

## Evidence For vs. What It Does Not Prove

| Area | Evidence the model has | What it does not prove |
|---|---|---|
| Baseline normal | Equal-budget service `0.990`, dispatch success `1.000`, PASS/PASS. | Real daily demand mix or warehouse execution variance. |
| Demand spike | Equal-budget service `0.859`, dispatch success `1.000`, PASS/PASS. | True peak-season order correlation or real carrier cutoff behavior. |
| High holding cost | Equal-budget service `0.966`, dispatch success `0.999`, PASS/PASS. | Real holding cost accounting or SKU-level finance validity. |
| Lead-time volatility | Equal-budget service `1.000`, dispatch success `0.999`, PASS/PASS. | Actual supplier delay distributions or procurement constraints. |
| Mixed stress | Equal-budget service `0.942`, dispatch success `0.999`, PASS/PASS; action 24 concentration accepted. | Real combined shock correlations, secondary-fleet economics, or reorder-none economics. |
| Premium SLA | Equal-budget service `1.000`, dispatch success `0.999`, PASS/PASS. | Actual premium customer penalties or contractual SLA rules. |
| Route disruption | Equal-budget service `0.901`, dispatch success `0.999`, PASS/PASS; service delta vs old production `-0.004` inside gate tolerance; action 32 concentration accepted. | Real congestion feeds, road-network disruptions, or carrier route failure taxonomy. |
| Vehicle scarcity | Equal-budget service `0.949`, dispatch success `1.000`, PASS/PASS. | Real fleet availability, carrier rejection behavior, or vehicle assignment locks. |
| Action quality | Hard blockers zero; no no-current/no-unassigned/failed-noop explosion on watched action 24/32 in equal-budget audit. | That every future telemetry regime will avoid no-work or failed-noop pockets. |
| Route-side plausibility | Amazon Last Mile small-sample work supports partial route-side validation. | Dispatch mode, fleet split, cost, inventory, and reorder economics. |

## Equal-Budget Comparator Snapshot

| Scenario | Old prod service/success | Hier 1M service/success | Delta service | Verdict |
|---|---:|---:|---:|---|
| `baseline_normal` | 0.945 / 0.998 | 0.990 / 1.000 | +0.045 | PASS/PASS |
| `demand_spike_volatility` | 0.840 / 1.000 | 0.859 / 1.000 | +0.019 | PASS/PASS |
| `high_holding_cost` | 0.931 / 0.997 | 0.966 / 0.999 | +0.035 | PASS/PASS |
| `lead_time_volatility` | 0.914 / 0.998 | 1.000 / 0.999 | +0.086 | PASS/PASS |
| `mixed_stress` | 0.940 / 0.997 | 0.942 / 0.999 | +0.003 | PASS/PASS |
| `premium_sla_pressure` | 0.982 / 0.759 | 1.000 / 0.999 | +0.018 | PASS/PASS |
| `route_disruption_congestion` | 0.904 / 0.862 | 0.901 / 0.999 | -0.004 | PASS/PASS |
| `vehicle_scarcity_capacity_shock` | 0.908 / 0.998 | 0.949 / 1.000 | +0.041 | PASS/PASS |

## Overfit-Risk Assessment

Overfit risk is present but not currently disqualifying.

Reasons the result is not merely a trivial gate overfit:

- Many earlier branches failed the same project gate family, including candidates with zero hard blockers.
- The long-run gate catches production-comparison regressions, dispatch-success regressions, and no-work top-action explosions, not only scenario threshold PASS.
- Equal-budget old-production versus hierarchical production comparison used the same scenario dir, 20 episodes per scenario, same seed policy, CPU deterministic, 8/8 PASS for both, and a gate PASS for hierarchical.
- Route and mixed residual watches were statistically rechecked and classified acceptable.
- Public Amazon Last Mile small-sample analysis gives partial external plausibility for route-side preferences.

Reasons overfit risk remains:

- The evaluator still uses a finite scenario family.
- The simulator is a curated digital twin, not live operations.
- The policy may have learned good behavior around known route/fleet/reorder abstractions rather than general logistics reasoning.
- Accepted action concentration in route and mixed stress must remain monitored.

Current conclusion: not enough evidence to call it simple overfit; enough evidence to require monitoring and company-data validation before claiming real-world autonomy.

## Undertraining-Risk Assessment

Undertraining risk is low under the current simulator/eval contract, but not zero.

Evidence against undertraining:

- 250k, 500k, and 1M all passed their ladder gates.
- 1M passed 8/8 scenario verdicts, hard blockers zero, and long-run gate PASS.
- Equal-budget residual-watch assurance passed with 8/8 scenarios and 160 episode rows for both comparator and hierarchical production.
- Dispatch success was near-perfect across all equal-budget hierarchical scenarios.
- The model improved service versus old production in seven of eight equal-budget scenarios and had only a small route service delta inside tolerance.
- Exact-resume metadata shows replay/RNG were present and restored through the ladder.

Evidence that still warrants watchfulness:

- Route disruption action 32 concentration is real.
- Mixed stress action 24 concentration is real.
- Route lateness and mixed route-failure warnings were accepted as nonfatal, not erased.
- The DQN replay capacity is 500k, so the final 1M checkpoint stores a rolling buffer rather than all historical transitions.

Current conclusion: 1M is sufficient for the approved gates. It is not evidence that additional training would be useless forever; it means additional training needs a trigger.

## Real-World Limitation Assessment

The model is simulator production-ready, not automatically real-world autonomous.

It does not prove:

- real company secondary-fleet cost/reliability,
- reorder-none economics under actual SKU/site constraints,
- carrier acceptance/rejection patterns,
- true route failure reason taxonomy,
- real TMS/WMS event semantics,
- real integration race conditions or lock conflicts,
- universal logistics optimality across geographies, customer mixes, or network topologies,
- that action 24/32 concentration is economically optimal in company operations.

The real-world calibration plan correctly says public route data can partially validate route-choice proxies, while company data is needed for fleet, order, inventory, dispatch, cost, and reorder calibration.

## Trigger-Based Future Training Policy

Do not start blind 3M/5M/10M/100M training.

More training becomes scientifically justified only if at least one trigger appears:

| Trigger | Evidence required | Next action |
|---|---|---|
| Monitoring regression | Service drops below scenario/production thresholds, lateness rises, dispatch success drops, or hard blockers appear. | Root-cause telemetry first; if model-caused, design gated 1.5M/2M extension. |
| Action-quality regression | Action 24/32 no-current, no-unassigned, failed-noop, route failure, or no-vehicle counters grow materially. | Diagnose whether simulator, runtime, or policy drift is responsible before training. |
| Company-data mismatch | Real secondary-fleet, route, cost, or reorder economics contradict simulator assumptions. | Calibrate data contract or simulator before model training. |
| Public route contradiction | Public route proxy analysis contradicts shortest/low-congestion route preferences under stress. | Revisit route reward/features before extension. |
| Hidden bug | Runtime, observation mapping, action mapping, gate, or metric bug found. | Fix with TDD and re-audit before training. |
| New validated scenario gap | A fresh, approved scenario exposes a real operations gap not covered by current eight scenarios. | Add gated scenario/eval plan before extension. |

If any trigger justifies training, start with a bounded 1.5M or 2M exact-resume extension plan, not blind 3M. The extension should use fresh dirs, preflight protected hashes, explicit scenario/gate criteria, exact replay/RNG resume, action-quality watches, and independent review.

## Recommended Next Action

Keep the current hierarchical v1 1M production model under monitoring. Prioritize real-world validation and telemetry readiness over longer simulator training:

1. Monitor service, lateness, dispatch success, no-current/no-unassigned, failed-noop, route failure, no-vehicle, top-action concentration, and action 24/32 rates.
2. Continue public route-side validation only as partial evidence.
3. Prepare company-data intake and KPI validation before making claims about secondary-fleet and reorder economics.
4. Open a training extension only after a concrete trigger, with 1.5M/2M gates first.

Plainly: the next best scientific move is better evidence, not more steps.

## Non-Goals

- No training.
- No offline evaluation.
- No long-run gate.
- No dataset download.
- No registry update.
- No `active_models.json` or `models.jsonl` mutation.
- No production mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No existing eval-output edit.
- No new training config.
- No 1.5M, 2M, 3M, 5M, 10M, or 100M run.
- No claim of real-world autonomous readiness without company data.

## Protected No-Mutation Proof

Pre-write protected state:

- `models/registry/active_models.json`: SHA256 `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: SHA256 `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: SHA256 `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: SHA256 `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`
- `models/baselines`: 2692 files, 7392576274 bytes
- `db`: 7 files, 28330 bytes
- `models/checkpoints`: 787 files, 4920830666 bytes
- `models/production`: 8 files, 328265276 bytes
- `models/eval`: 128 files, 388267659 bytes

Process scan before writing found no matching train/eval/gate/AWS/private-data process patterns.

Only this requested documentation file was added:

`docs/runs/20260612_1m_learning_efficiency_and_sufficiency_audit.md`

## Independent Review

Reviewer verdict: `ONE_M_LEARNING_EFFICIENCY_AUDIT_APPROVED`.

Allowed verdicts:

- `ONE_M_LEARNING_EFFICIENCY_AUDIT_APPROVED`
- `ONE_M_LEARNING_EFFICIENCY_AUDIT_NEEDS_FIXES`
- `ONE_M_LEARNING_EFFICIENCY_AUDIT_BLOCKED`
