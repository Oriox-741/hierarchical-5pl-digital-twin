# Thesis Advisor Presentation Outline

Date: 2026-06-12

## Purpose

Structure a clear explanation of CODEX PROJE for a thesis advisor.

## Slide Outline

## 5-Minute Version

1. Opening sentence:
   - "This project is an autonomous 5PL digital-twin control tower that uses reinforcement learning to make logistics decisions in simulation and wraps the approved model in production-style governance."
2. Problem:
   - Logistics decisions are coupled: inventory, dispatch, routing, fleet choice, reorder mode, service level, lateness, disruption, and cost all interact.
3. System:
   - The project implements a digital twin, PPO/DQN decision policies, scenario evaluation, long-run gates, and monitoring/governance tools.
4. Current result:
   - The current hierarchical v1 1M model passed 250k/500k/1M gates, 8/8 scenarios, hard blockers zero, equal-budget residual-watch gate PASS, and ops bundle clean.
5. Honest boundary:
   - It is simulator-production-ready, not yet a live real-world deployment. Company data is required for full validation.
6. Next step:
   - Monitor current model and validate with company data before further training.

## 10-Minute Version

1. Research motivation:
   - 5PL logistics requires adaptive coordination across orders, inventory, fleet, routes, and disruptions.
2. Digital twin:
   - `env_5pl` simulates inventory, demand, fleet, dispatch, routes, congestion, premium SLA, lead-time volatility, holding cost, vehicle scarcity, and mixed stress.
3. Decision model:
   - PPO handles continuous strategic controls.
   - DQN handles 48 discrete tactical actions.
   - `hierarchical_v1` factorizes the DQN into dispatch/route/mode/reorder heads while preserving the external 48-action contract.
4. Learning path:
   - Production flat parent.
   - Flat-teacher distillation.
   - 250k warm-start run.
   - 500k exact resume.
   - 1M exact resume.
5. Evaluation:
   - Eight scenario suite.
   - Hard blockers.
   - Long-run gate.
   - Equal-budget residual-watch evaluation.
6. Production lifecycle:
   - Candidate registration.
   - Active registry activation.
   - Copy-only production promotion.
   - Production manifest and rollback plan.
7. Monitoring and governance:
   - Artifact health.
   - Monitoring report generator.
   - Company-data intake validator.
   - Ops bundle.
8. Limitations:
   - No live TMS/WMS/ERP.
   - No private company data.
   - Secondary fleet/reorder economics unvalidated.
   - No blind 3M/5M/10M training.
9. Thesis position:
   - Digital twin + hierarchical RL + safe MLOps governance for logistics autonomy.

## 20-Minute Technical Version

1. Motivation and research question
   - Can a reinforcement-learning control tower coordinate multi-layer logistics decisions safely in a 5PL digital twin?
2. System architecture
   - Sense: telemetry and state abstractions.
   - Think: PPO/DQN policies and reward/feasibility logic.
   - Act: SimPy/Gymnasium digital twin.
   - Learn: synchronized torch joint trainer.
   - Govern: registry, production manifest, monitoring, ops bundle.
3. MDP contract
   - `physical_reality_v5_route_candidate_visibility`.
   - Observation dimension 73.
   - Continuous action dimension 5.
   - External discrete action count 48.
   - 5-minute decision interval and 288-step daily horizon.
4. Action design
   - Dispatch/hold.
   - Route: shortest, low congestion, high resilience.
   - Fleet: primary, secondary.
   - Reorder: none, conservative, aggressive, emergency.
5. Hierarchical DQN
   - Why flat 48-action DQN created action-pocket drift.
   - How `hierarchical_v1` decomposes Q values.
   - Why it preserves runtime compatibility.
6. Initialization
   - Why flat production cannot be exact-resumed into hierarchical heads.
   - `flat_teacher_distillation_v1` as a read-only teacher warm start.
7. Exact resume
   - Why replay/RNG persistence matters.
   - What previous V2/V2.4/V2.5 failures taught.
8. Curriculum and training ladder
   - 250k, 500k, 1M.
   - Exact resume from passed checkpoints.
   - No failed checkpoints as parents.
9. Evaluation/gating
   - Eight scenario suite.
   - Scenario thresholds.
   - Hard blockers.
   - Long-run gate.
   - Equal-budget residual-watch eval.
10. Production promotion
   - Candidate rows.
   - Active registry.
   - Copy-only production directory.
   - Manifest and hashes.
11. Monitoring and governance
   - Artifact health.
   - Monitoring report.
   - Company-data validator.
   - Ops bundle.
12. Real-world calibration
   - Amazon Last Mile route-side proxy.
   - Company data needed for fleet/reorder/cost.
13. Limitations and future work
   - Simulation-to-real gap.
   - Integration gap.
   - Data calibration.
   - Trigger-based future training only.

## Diagram Suggestions

1. Sense-Think-Act-Learn loop
   - Sense state -> Policy -> Action -> Digital twin -> Metrics/replay -> Governance.
2. Current model lifecycle
   - Production flat parent -> distillation -> 250k -> exact resume 500k -> exact resume 1M -> candidate -> active registry -> copy-only production.
3. DQN action decomposition
   - 48 external actions = 2 dispatch x 3 route x 2 fleet x 4 reorder.
   - Hierarchical heads combine back into 48 Q values.
4. Evaluation and gate stack
   - Scenario suite -> episode metrics -> scenario summary -> long-run gate -> residual-watch assurance.
5. Production governance stack
   - Registry -> production manifest -> artifact health -> monitoring report -> company-data validator -> ops bundle.
6. Real-world validation boundary
   - Simulator evidence vs public route data vs company data.

## Terminology Glossary

| Term | Advisor-friendly explanation |
|---|---|
| 5PL | Fifth-party logistics: a platform/control-tower layer coordinating logistics networks beyond one carrier or warehouse. |
| Digital twin | A simulated operational copy of a logistics system used for decision testing. |
| PPO | Reinforcement-learning algorithm used here for continuous strategic controls. |
| DQN | Reinforcement-learning algorithm used here for discrete tactical actions. |
| Torch joint | The local synchronized PPO+DQN PyTorch policy bundle. |
| Observation 73 | The fixed 73-feature state vector the policy sees. |
| Action 48 | The fixed 48-action tactical choice space. |
| Hierarchical DQN | A DQN whose internal heads represent action factors, while output remains 48 Q values. |
| Flat teacher distillation | Loading behavior from an older flat production model into a new hierarchical model using supervised Q/action matching. |
| Exact resume | Continuing training while restoring model, optimizer, global step, replay buffer, and RNG state. |
| Hard blocker | A condition that should fail a candidate regardless of average performance. |
| Long-run gate | A read-only checker that compares scenario summaries to production and prior gates. |
| Residual watch | A nonfatal behavior that passed gates but must be monitored. |
| Simulator production-ready | Accepted under this repository's simulation, evaluation, and governance contract. |
| Real-world autonomous | Safe live operation against real company systems and data; not yet proven here. |

## Advisor-Safe Framing

- Present the model as simulator-production-ready under the repository's evaluation and governance contract.
- Separate simulation autonomy from real-world deployment.
- Treat action 24 and action 32 as accepted residual watches, not as proof of real-world economic optimality.
- Use current dashboard, handoff, audits, production manifest, and generated reports as truth sources; treat older README/V2 notes as historical context.

## Suggested Demo Path

1. Start with the release handoff and production manifest.
2. Show the 48-action decomposition and explain action 24/32.
3. Show the hierarchical DQN architecture at a conceptual level.
4. Walk through the 250k -> 500k -> 1M ladder and exact-resume evidence.
5. Show 8/8 scenario PASS, equal-budget residual-watch PASS, artifact-health clean, synthetic monitoring clean, company-schema ready, and ops bundle clean.
6. Close with the limitation boundary: company data is needed before real-world deployment claims.

## Questions To Invite

- Which simulator assumptions should be validated first with company data?
- Which monitoring watches should become hard rollback triggers?
- What is the minimum real-world pilot scope that would be academically defensible?
- Should the thesis emphasize reinforcement learning architecture, digital-twin modeling, or governed MLOps?

## Likely Advisor Questions And Suggested Answers

### What exactly is the novelty?

The novelty is the integrated system: a 5PL digital-twin decision environment, synchronized PPO+DQN control, hierarchical action factorization, exact-resume continuation, scenario gates, and production-style governance in one traceable research artifact.

### Is this just a vehicle-routing problem?

No. Route choice is only one action factor. The model also handles dispatch/hold, fleet mode, reorder mode, inventory and capacity pressure, service/lateness, demand spikes, and disruptions.

### Is it fully autonomous?

In simulation, yes: it closes the loop from observation to action to environment transition. In the real world, not yet: it is not connected to TMS/WMS/ERP and does not execute real operations.

### Why did 1M steps work?

Because the 1M run was not from scratch. It inherited a passing flat production parent, used teacher distillation, used hierarchical action decomposition, continued with replay/RNG exact resume, and trained on a bounded curriculum.

### Why not train 10M?

The project history shows longer or blind continuation can worsen action-pocket behavior. More training should follow evidence, not hope. The current recommendation is monitor first, then train only if a trigger appears.

### What are action 24 and action 32?

Action 24 is dispatch + shortest route + secondary fleet + no reorder. Action 32 is dispatch + low-congestion route + secondary fleet + no reorder. They are accepted monitoring watches because they concentrate in some stress scenarios without current no-work/failed-noop explosions.

### What does the model prove?

It proves the project can build, train, evaluate, promote, and monitor a policy that passes the defined simulator/gate contract.

### What does it not prove?

It does not prove real-world optimality, real fleet economics, real reorder economics, or integration safety without company data.

### What data do we need next?

Orders, dispatch attempts, deliveries, route records, fleet/carrier records, inventory/reorder events, costs, timestamps, failure reasons, and join keys.

### What can be demonstrated safely?

Use existing reports, artifact-health JSON, synthetic monitoring report, synthetic company-data validator report, ops bundle report, and read-only architecture walkthroughs. Do not run training/eval/gates without separate approval.

## Closing Line

This is best presented as a governed autonomous logistics digital-twin prototype: technically deep enough for thesis research, operationally disciplined enough to resemble production MLOps, and honest enough to separate simulator success from real-world deployment proof.
