# Full Project Chronology And Future Roadmap

Date: 2026-06-11

Final classification: `FULL_PROJECT_CHRONOLOGY_AND_ROADMAP_READY`

## Scope

This is a read-only historical archaeology and roadmap synthesis for CODEX
PROJE. It reconstructs the project from docs, plans, run reports, release
handoffs, configs, tests, source structure, eval summaries, registry metadata,
and production manifests.

No training, offline eval, long-run gate, dataset download, registry update,
production mutation, baseline mutation, DB mutation, checkpoint mutation, model
config creation, or existing eval-output edit was performed for this report.

Allowed writes for this task:

- `docs/runs/20260611_full_project_chronology_and_future_roadmap.md`
- `docs/plans/20260611_codex_project_future_strategy_plan.md`
- `docs/00_PROJECT_DASHBOARD.md` link update

## Executive Summary

The project has moved through three major eras:

1. Contract repair and productionization of a flat PPO+DQN torch-joint system
   under v5 route-candidate visibility.
2. Repeated failed stability experiments around the old flat 48-action DQN,
   leading to exact-resume repair and the conclusion that resume mechanics were
   necessary but not sufficient.
3. A successful architectural shift to `hierarchical_v1`, initialized from the
   flat production teacher through `flat_teacher_distillation_v1`, followed by a
   clean 250k -> 500k -> 1M exact-resume ladder and copy-only production
   promotion.

Current production is:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- DQN architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation/action: `73 / 48`
- active registry: routed to hierarchical v1 PPO/DQN final artifacts
- production copy: present with manifest and matching hashes
- evaluation: 8/8 scenarios PASS, hard blockers zero, long-run gate PASS
- residual watches: accepted for monitoring after equal-budget assurance

The strategic direction should now shift from model search to production
hardening and real-world calibration. More training is not the next default
move. A 1.5M/2M extension is only justified if monitoring or company data
identifies a specific reproducible issue.

## Current Production State

Truth sources:

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `docs/runs/20260611_hierarchical_v1_production_assurance_audit.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`
- `models/registry/active_models.json`
- `models/registry/models.jsonl`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`

Current active registry mappings:

```json
{
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt",
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt"
}
```

Production artifacts:

| Artifact | SHA256 |
| --- | --- |
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` |
| `production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` |

Registry ids:

| Role | Status | Model id |
| --- | --- | --- |
| PPO `continuous_control` | active | `17ba1d28-0054-4f7c-ae9a-34cd305ebb89` |
| DQN `tactical_dispatch` | active | `f87e10d6-479f-44fc-99d1-6925bc9cb346` |
| PPO `continuous_control` | candidate | `3befa11c-e853-4d0f-a951-c2fbbb2a9898` |
| DQN `tactical_dispatch` | candidate | `71af6701-ac3a-40f2-bdd9-cc80868d5a1c` |

## Chronological Timeline

| Date | Phase | Key files / artifacts | Result |
| --- | --- | --- | --- |
| 2026-05-19 to 2026-05-25 | Foundational repair plans | `docs/superpowers/plans/*.md`, `docs/superpowers/project_issue_resolution_history.md` | Defined torch-joint direction, physical-reality repairs, dispatch feasibility, real-world arena ideas, and early 3M caution. |
| 2026-06-07/08 | Rewardfix / perfclean | `docs/plans/20260607_rewardfix_clean_training_and_eval_plan.md`, `docs/runs/20260608_clean_rewardfix_perfclean_1m_autopilot_status.md` | Stale rewardfix identities quarantined; clean perfclean path established. |
| 2026-06-08 | Clean 1M parent | `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json` | Clean v5 parent became source for later fine-tunes. |
| 2026-06-08/09 | Route/premium/mixed targeted fine-tune | `docs/plans/20260608_routepremium_mixed_targeted_fix_plan.md` | Narrow route and premium gains, but retention regressions. Not registry-ready. |
| 2026-06-09 | Balanced retention production | `docs/releases/20260609_torch_joint_v5_balanced_retention_production_handoff.md` | 8/8 scenarios PASS and hard blockers zero; became old production comparator. |
| 2026-06-09 | Nextgen 1M failure | `docs/plans/20260609_nextgen_longrun_redesign_plan.md` | 300k 2/8 PASS, 1M 6/8 PASS; branch stopped. |
| 2026-06-09/10 | V2/V2.1/V2.2 experiments | `docs/runs/20260610_bounded_stability_autopilot_stop_report.md` | V2 4/8, V2.1 4/8, V2.2 5/8; stopped after failed gates. |
| 2026-06-10 | V2.3/V2.4 diagnostics | `docs/runs/20260610_level2_postmortem_v2_cycle_design.md` | V2.3 thresholds passed but route action-quality gate failed. V2.4 250k passed after semantics fix; 500k failed. |
| 2026-06-10 | Exact-resume discovery/fix | `docs/runs/20260610_level2_continuation_exact_resume_patch_report.md` | Found missing DQN replay/RNG in resume; patched future exact resume. |
| 2026-06-10 | V2.5 exact resume | `docs/runs/20260610_level2_v2_5_exactresume_500k_architecture_report.md` | 250k PASS, 500k FAIL. Exact resume was necessary but insufficient. |
| 2026-06-11 | Teacher retention | `docs/runs/20260611_level2_teacher_retention_ladder_report.md` | Teacher-retention 250k failed; no 500k/1M. |
| 2026-06-11 | Hierarchical DQN architecture | `docs/runs/20260611_hierarchical_dqn_action_architecture_patch_report.md` | `hierarchical_v1` implemented; flat/hierarchical checkpoint mismatch rejected. |
| 2026-06-11 | Hierarchical initialization | `docs/runs/20260611_hierarchical_dqn_initialization_decision.md`, `docs/runs/20260611_hierarchical_dqn_initialization_patch_report.md` | Selected flat-teacher distillation warm-start. |
| 2026-06-11 | Hierarchical ladder | `docs/runs/20260611_hierarchical_dqn_ladder_report.md` | 250k PASS, 500k PASS, 1M PASS. |
| 2026-06-11 | Candidate/active registry | `docs/runs/20260611_hierarchical_v1_candidate_registration_audit.md`, `docs/runs/20260611_hierarchical_v1_active_registry_activation_audit.md` | Candidate rows registered; active registry switched to hierarchical v1. |
| 2026-06-11 | Production promotion | `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md` | Copy-only production promotion complete. |
| 2026-06-11 | Residual-watch assurance | `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md` | Equal-budget gate PASS; watches accepted for monitoring. |
| 2026-06-11 | Public route data calibration | `docs/plans/20260611_real_world_calibration_and_validation_plan.md`, `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md` | Amazon small sample supports partial route-side plausibility; company data still required. |
| 2026-06-11 | Production monitoring | `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md` | Operational monitoring and no-blind-3M policy documented. |

## Phase Narratives

### Early Rewardfix / Perfclean Phase

Goal:

- Make route, premium, dispatch, and reward telemetry trustworthy enough to
  evaluate candidates.
- Quarantine stale or aborted rewardfix identities.

Key files:

- `docs/plans/20260607_rewardfix_clean_training_and_eval_plan.md`
- `docs/EXPERIMENT_HISTORY.md`
- `docs/PROJECT_HANDOFF.md`

Learning:

- The project could not rely on legacy or partially aborted model identities.
- Valid artifacts needed explicit contract, observation dimension, action count,
  trace cadence, and exploit-gate verification.

Why it moved on:

- A clean v5/73/48 perfclean parent became the safer foundation for targeted
  fine-tuning.

### Clean 1M Parent Phase

Goal:

- Produce a full-curriculum v5 parent with hard-blocker repairs in place.

Key artifacts:

- `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json`
- `models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608_offline_scenarios`

Result:

- The parent was usable as a fine-tune source, but not the final production
  answer.
- The invalid earlier `joint_curriculum_v5_clean_cold_start_1m` stayed
  historical only because its curriculum stayed `normal_v4`, it had a
  delivery-credit leak, and checkpoint/metrics steps disagreed.

Why it moved on:

- Route and premium behavior still needed targeted repair.

### Route / Premium / Mixed Targeted Fine-Tune Phase

Goal:

- Fix route disruption, premium SLA, and mixed-stress behavior without losing
  the broad clean parent.

Key artifacts:

- `docs/plans/20260608_routepremium_mixed_targeted_fix_plan.md`
- `configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k.json`
- `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k_20260608_offline_scenarios`

Result:

- Route and premium improved, but retention failed.
- Eval summary showed 5/8 PASS: baseline, high-holding, and lead-time regressed.

Learning:

- Narrow schedules can repair local symptoms while destroying general behavior.

Why it moved on:

- Needed balanced retention, not a narrow route/premium-only fix.

### Balanced Retention Production Phase

Goal:

- Preserve broad behavior while carrying enough route/premium improvements.

Key artifacts:

- `configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json`
- `docs/releases/20260609_torch_joint_v5_balanced_retention_production_handoff.md`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/production_manifest.json`

Result:

- 8/8 scenarios PASS.
- Hard blockers zero.
- Candidate and active registry rows were created, then a copy-only production
  folder was made.
- This became the old production comparator and rollback target.

Residual watches:

- Route disruption high-resilience selection when not best.
- Demand spike lateness.
- Mixed-success route-failure warnings.

Why it moved on:

- The user wanted longer-horizon stability and stronger candidates.

### Nextgen 1M Failure Phase

Goal:

- Extend from production into a longer next-generation branch.

Key artifacts:

- `docs/plans/20260609_nextgen_longrun_redesign_plan.md`
- `models/eval/joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios`
- `models/eval/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios`

Results:

- 300k: 2/8 PASS.
- 1M: 6/8 PASS.
- Premium and route-disruption dispatch quality failed.

Learning:

- Longer training without gates can move from under-serving to over-dispatch.
- A candidate can keep hard blockers zero and still fail production-grade
  behavior.

Why it moved on:

- Future long runs required periodic eval gates, early stopping, and action
  quality instrumentation.

### V2 / V2.1 / V2.2 Stability Experiments

Goal:

- Use short 250k gated cycles to regain stability from the production parent.

Key artifacts:

- `docs/runs/20260610_bounded_stability_autopilot_stop_report.md`
- `configs/training_joint_curriculum_v5_prod_stability_v2_250k_20260609.json`
- `configs/training_joint_curriculum_v5_prod_stability_v2_1_250k_20260609.json`
- `configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json`

Results:

| Branch | Scenario PASS | Gate | Main failure |
| --- | ---: | --- | --- |
| V2 250k | 4/8 | FAIL | timing/service under-response |
| V2.1 250k | 4/8 | FAIL | baseline/premium hold collapse |
| V2.2 250k | 5/8 | FAIL | high-holding, lead-time, vehicle, route action quality |

Learning:

- The system was not suffering a single stable bug; policy failures migrated
  across action pockets.
- Baseline anchoring helped some scenarios but did not solve route/action
  quality.

Why it moved on:

- The bounded goal hit failed-cycle limits and required deeper postmortem rather
  than blind config tweaks.

### V2.3 / V2.4 Diagnostic And Semantics Phase

Goal:

- Add richer diagnostics and resolve whether failures were semantic/evaluator
  issues or training policy issues.

Key artifacts:

- `docs/runs/20260610_level2_postmortem_v2_cycle_design.md`
- `docs/runs/20260610_level2_phase2_review_request.md`
- `docs/runs/20260610_level2_phase3_review_request.md`
- `configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json`
- `configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json`

Results:

- V2.3: 8/8 scenario thresholds PASS, but long-run gate FAIL on route
  action-quality. Action 32 had 848 failed no-ops.
- V2.4 250k: passed after evaluator semantics were resolved.
- V2.4 500k: failed after continuation.

Learning:

- Scenario thresholds were not enough; action-quality gates caught failures
  that service-level thresholds missed.
- V2.4 500k degradation began by 300k and became severe by 400k/500k.

Why it moved on:

- The continuation architecture itself needed auditing.

### Exact-Resume Discovery And Fix

Goal:

- Determine whether 250k -> 500k continuation was a true continuation or a
  weights-only/warm-start continuation.

Key artifacts:

- `docs/runs/20260610_level2_continuation_exact_resume_patch_report.md`
- `src/learn/train_joint_torch.py`
- `src/learn/joint_buffers.py`
- `tests/learn/test_train_joint_curriculum.py`
- `tests/learn/test_joint_reward_blending.py`

Finding:

- Legacy `--resume` restored model/optimizer/global step, but DQN replay started
  empty and RNG was fresh.
- Future exact resume now persists replay state, replay cursor/size, RNG state,
  optimizer state, and resume metadata.
- Normal `--resume` rejects legacy checkpoints missing replay/RNG unless
  `--allow-empty-replay-resume` is explicitly used.

V2.5 result:

- 250k exact-resume-capable candidate passed.
- 500k exact resume failed route service and lead-time no-work action quality.

Learning:

- Exact resume was necessary but not sufficient. Architecture remained the
  limiting factor.

### Teacher-Retention Attempt

Goal:

- Preserve production behavior with read-only flat teachers and state banks.

Key artifacts:

- `docs/runs/20260610_level2_teacher_retention_vs_hierarchical_architecture_decision.md`
- `docs/runs/20260611_level2_teacher_retention_patch_report.md`
- `docs/runs/20260611_level2_teacher_retention_ladder_report.md`
- `src/learn/teacher_retention.py`

Result:

- Teacher-retention 250k failed the long-run gate.
- Baseline service/lateness regressed, lead-time collapsed, high-holding lateness
  exceeded threshold, and route service fell.

Learning:

- Flat teacher anchoring could not solve the composite-action pocket problem.

Why it moved on:

- The architecture decision had already identified hierarchical/decomposed DQN
  as the next path if teacher retention failed.

### Hierarchical DQN Architecture Phase

Goal:

- Preserve the external 48-action API while replacing flat internal DQN logits
  with dispatch, route, mode, and reorder heads.

Key artifacts:

- `docs/plans/20260611_hierarchical_dqn_action_architecture_plan.md`
- `docs/runs/20260611_hierarchical_dqn_action_architecture_patch_report.md`
- `src/think/joint_policies.py`
- `tests/think/test_hierarchical_dqn_policy.py`

Result:

- `hierarchical_v1` implemented.
- HOLD actions mask route/mode components, preventing meaningless hold-route
  labels from forming action-16-style pockets.
- Flat and hierarchical checkpoint mismatch is rejected.

Learning:

- The project needed a model-class change, not just reward/curriculum tuning.

### Hierarchical Initialization Phase

Goal:

- Decide how to initialize hierarchical heads from a flat production parent.

Options considered:

- random heads with transferred trunk,
- flat-to-hierarchical distillation,
- analytical flat-to-factorized conversion,
- PPO-only from production with random DQN heads.

Decision:

- Use `flat_teacher_distillation_v1` as an explicit read-only teacher
  warm-start. Do not treat the flat production checkpoint as a hierarchical
  exact-resume parent.

Key artifacts:

- `docs/runs/20260611_hierarchical_dqn_initialization_decision.md`
- `docs/runs/20260611_hierarchical_dqn_initialization_patch_report.md`
- `src/learn/hierarchical_dqn_initialization.py`
- `tests/learn/test_hierarchical_dqn_initialization.py`

### Hierarchical 250k / 500k / 1M Ladder

Goal:

- Build a clean 1M candidate through exact-resume ladder steps.

Key artifacts:

- `docs/runs/20260611_hierarchical_dqn_ladder_report.md`
- `configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json`
- `configs/training_joint_curriculum_v5_prod_hierarchical_v1_500k_20260611.json`
- `configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json`
- `models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`

Result:

| Rung | Result |
| --- | --- |
| 250k | PASS, gate exit `0` |
| 500k | PASS, gate exit `0` |
| 1M | PASS, gate exit `0` |

1M scenario summary:

- 8/8 PASS.
- 160 episode rows.
- Hard blockers zero.
- Long-run gate PASS.

Residual watches:

- route-disruption service below old production in original comparison,
- mixed-stress service slightly below old production in original comparison,
- action 32 concentration in route disruption,
- action 24 concentration in mixed stress,
- top-action concentration warnings,
- mixed-success route-failure warnings.

### Registry, Active, And Production Promotion

Goal:

- Move the hierarchical v1 1M candidate through registry and production without
  mutating unrelated artifacts.

Key artifacts:

- `docs/runs/20260611_hierarchical_v1_candidate_registration_audit.md`
- `docs/runs/20260611_hierarchical_v1_active_registry_activation_audit.md`
- `docs/runs/20260611_hierarchical_v1_production_promotion_audit.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`

Result:

- Candidate registration appended exactly two rows.
- Active activation appended exactly two active rows and updated only the active
  PPO/DQN mappings.
- Copy-only production promotion created the hierarchical production folder,
  copied exactly the approved three `.pt` artifacts, and wrote
  `production_manifest.json`.

Learning:

- Candidate registration, active activation, and production copy are separate
  approval domains and should remain separate.

### Residual-Watch Assurance

Goal:

- Determine whether residual watches were statistically meaningful or acceptable.

Key artifacts:

- `docs/runs/20260611_hierarchical_v1_residual_watch_statistical_assurance.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `models/eval/equal_budget_oldprod_20ep_seed42_20260611`
- `models/eval/equal_budget_hierarchical_v1_20ep_seed42_20260611`

Result:

- Equal-budget old production and hierarchical production both ran 20 episodes
  per scenario with seed 42.
- Both had 8/8 PASS and 160 rows.
- Equal-budget long-run gate PASS.
- Mixed stress no longer regressed.
- Route-disruption service delta was `-0.0037`, inside gate tolerance.

Learning:

- Equal-budget comparison is required before converting residual watches into a
  training trigger.

### Real-World Calibration / Public Route Data Phase

Goal:

- Validate the route-choice side of action 24 and action 32 without company data.

Key artifacts:

- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_public_route_proxy_validation_readiness.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`
- `scripts/real_world_calibration/*`

Result:

- Amazon Last Mile is the best first public dataset.
- Seven approved small files were downloaded in a separate task, totaling
  `5.007 MiB`.
- Small sample had 13 routes and 3,129 packages.
- Actual/greedy travel-time ratio mean was `0.967`; 9/13 actual routes were no
  worse than greedy.
- Directed travel-time asymmetry was nontrivial.

Learning:

- Public data supports partial route-side plausibility for action 24/32.
- It cannot validate secondary-fleet usage, no-reorder behavior, inventory, or
  company-specific costs.

## Branch Failure / Success Matrix

| Branch / artifact | Outcome | Why |
| --- | --- | --- |
| `joint_curriculum_v5_clean_cold_start_1m` | invalid historical | Curriculum stuck on `normal_v4`, delivery-credit leak, checkpoint/metrics mismatch. |
| stale `after_rewardfix` artifacts | abandoned | Partial aborted history; not safe as current evidence. |
| clean after-blockers / perfclean parent | useful parent | Established v5/73/48 clean lineage and hard-blocker repairs. |
| routepremium/mixed targeted FT | failed retention | Local route/premium gains caused baseline, lead-time, high-holding regressions. |
| balanced retention 200k | old production success | 8/8 PASS, hard blockers zero, registry/prod promoted. |
| nextgen 300k/1M | stopped | 300k 2/8 PASS, 1M 6/8 PASS; longer continuation regressed quality. |
| V2 | failed | Timing/service under-response and route weakness. |
| V2.1 | failed | Hold/high-resilience/secondary concentration and baseline/premium collapse. |
| V2.2 | failed | Baseline/premium recovered but high-holding, lead-time, vehicle, and route action quality failed. |
| V2.3 | failed gate | 8/8 thresholds, but route action 32 no-current/no-unassigned pocket. |
| V2.4 250k | passed after semantics fix | Evaluator/horizon semantics issue resolved. |
| V2.4 500k | failed | Warm resume with empty replay/fresh RNG; degradation by 300k/400k. |
| exact-resume patch | success | Future checkpoints carry replay/RNG; legacy unsafe resume rejected. |
| V2.5 250k | passed | Exact-resume-capable 250k from production parent. |
| V2.5 500k | failed | Exact resume restored mechanics but route service and lead-time action quality still failed. |
| teacher-retention 250k | failed | Flat teacher anchoring shifted pockets; did not solve composite action structure. |
| hierarchical_v1 ladder | success | 250k/500k/1M all PASS; hard blockers zero. |
| hierarchical active/prod | success | Registry active and copy-only production promotion completed safely. |
| equal-budget residual assurance | success | Watches acceptable for monitoring, not training trigger. |
| Amazon small-sample route analysis | partial success | Route-side plausibility only; company economics still unknown. |

## Key Technical Decisions

- Keep the external contract at v5/73/48.
- Use raw PyTorch synchronized PPO+DQN as the active training/runtime family.
- Treat SB3 artifacts as archived/legacy unless explicitly in a legacy task.
- Use full offline scenario eval and read-only long-run gate before extending
  horizons.
- Keep registry candidate, active activation, and production copy as separate
  approval events.
- Require exact resume for gated continuation: replay state, replay cursor,
  global step, optimizer state, curriculum position, and RNG.
- Reject flat/hierarchical checkpoint mismatches.
- Initialize hierarchical v1 from flat production only through explicit
  read-only distillation, not exact resume.
- Use equal-budget comparisons before escalating residual watches.
- Treat public route data as route-side evidence only.

## Technical Audit

### Current Architecture

The project follows a Sense -> Think -> Act -> Learn architecture:

- Sense: DB pool, telemetry ingestion, trace writer, synthetic stream.
- Act: SimPy/Gymnasium-style 5PL digital twin, observation builder, routing,
  inventory, action projection, safety projection.
- Think: PPO continuous actor/critic plus DQN tactical policy.
- Learn: synchronized joint transition collection, PPO rollout buffer, DQN
  replay, curriculum, checkpointing, exact resume, registry lifecycle.
- Eval: real-world scenario arena, scenario metrics, long-run gate.
- Orchestration: active registry resolution and torch-joint serving.

Current production runtime path:

`active_models.json` -> `PolicyService` -> active PPO/DQN pair validation ->
`load_torch_joint_policy()` -> `JointPolicyBundle(dqn_architecture="hierarchical_v1")`
-> continuous action length 5 and discrete action 0..47.

### Important Source Modules

| Area | Important files |
| --- | --- |
| Simulation/reward | `src/act/env_5pl.py`, `src/act/routing_physics.py`, `src/act/inventory_dynamics.py`, `src/act/discrete_action_mapper.py`, `src/act/observation_builder.py` |
| Policies | `src/think/joint_policies.py` |
| Training/resume | `src/learn/train_joint_torch.py`, `src/learn/joint_buffers.py`, `src/learn/curriculum.py`, `src/learn/joint_metrics.py` |
| Hierarchical init | `src/learn/hierarchical_dqn_initialization.py` |
| Registry/promotion | `src/learn/model_registry.py`, `src/learn/register_torch_joint_candidate.py`, `src/learn/promote_torch_joint_production.py` |
| Eval/gates | `src/eval/real_world_scenario_arena.py`, `src/eval/scenario_metrics.py`, `src/eval/long_run_gate.py`, `src/eval/check_long_run_gate.py` |
| Runtime | `src/orchestration/policy_service.py`, `src/orchestration/torch_joint_runtime.py` |
| Calibration | `scripts/real_world_calibration/*` |

### Current Known Risks

1. Action concentration remains real: action 32 in route disruption and action
   24 in mixed stress.
2. Public data validates route-side plausibility only; secondary fleet and
   reorder economics are still unvalidated.
3. `env_5pl.py`, `train_joint_torch.py`, and `scenario_metrics.py` are large
   high-complexity research modules.
4. Many training configs encode safety assumptions manually.
5. No blind scale-up should occur while the current production model is already
   gate-clean and monitoring-only.

### Accepted Residual Watches

- `route_disruption_congestion`: action 32 concentration and small
  route-service/lateness watch.
- `mixed_stress`: action 24 concentration.
- top-action concentration warnings.
- mixed-success route-failure warnings.

These watches are accepted because equal-budget eval passed, hard blockers were
zero, and no no-current/no-unassigned/failed-noop explosion was found.

### Do Not Touch Without Explicit Approval

- `models/registry/models.jsonl`
- `models/registry/active_models.json`
- `models/production/**`
- `models/baselines/**`
- `models/checkpoints/**`
- `models/eval/**` existing outputs
- `reports/eval/**` existing outputs, if present
- `db/**` and any DB rows
- production manifests
- binary checkpoint bodies
- dataset downloads beyond approved bounded samples
- 1.5M/2M/3M/5M/10M/100M runs
- new training configs

### Previous Approaches Not To Repeat

- Do not use the aborted SB3 staggered co-training path for production.
- Do not resume invalid `joint_curriculum_v5_clean_cold_start_1m`.
- Do not use old v3/v4/probe checkpoints as v5 parents.
- Do not use failed nextgen, V2, V2.1, V2.2, V2.3, V2.4, V2.5, or
  teacher-retention checkpoints as parents.
- Do not do weights-only continuation for gated DQN runs.
- Do not try another blind reward/config tweak when a gate fails.
- Do not treat route-side public data as validation of fleet/reorder economics.

### Tests And Gates

Protective test areas:

- `tests/act`: reward physics, action mapping, observation contract, routing,
  speed, replenishment ownership.
- `tests/think`: hierarchical DQN composition and metadata.
- `tests/learn`: curriculum, exact resume, replay/RNG state, registry,
  promotion, hierarchical init, teacher retention.
- `tests/eval`: scenario configs, evaluator, long-run gate.
- `tests/orchestration`: Torch-joint runtime and PolicyService.
- `tests/real_world_calibration`: public route-proxy helpers.

Protective gates:

- offline real-world scenarios over `configs/eval_scenarios`,
- long-run gate against production comparator,
- hard-blocker counters,
- registry dry-runs,
- active activation dry-run with explicit `--replace-active`,
- copy-only production promotion plan and manifest audit,
- runtime smoke with SB3 loader count zero,
- final read-only production audits.

### Mature Code Areas

- Torch-joint runtime loading and checkpoint validation.
- Registry candidate/active lifecycle and dry-run behavior.
- Production copy and manifest discipline.
- Exact resume replay/RNG mechanics.
- Hierarchical DQN metadata and compatibility checks.
- Long-run gate and scenario-threshold enforcement.

### Research-Grade Code Areas

- Reward and dispatch internals in `env_5pl.py`.
- Scenario summary aggregation in `scenario_metrics.py`.
- Trainer orchestration in `train_joint_torch.py`.
- Calibration scripts and route-proxy analysis.
- Any future architecture work beyond `hierarchical_v1`.

## Recovered Old Future Plans

### Still Valid

- Monitor hierarchical v1 production behavior and residual watches.
- Build a real company data contract for order, route, fleet, inventory, cost,
  and dispatch failure fields.
- Use public route data to validate route-choice proxies only.
- Add an artifact health / read-only audit CLI.
- Add action-watch visualizations and structured monitoring reports.
- Keep baseline updates separate and approval-gated.
- Use bounded 1.5M/2M research only after a concrete trigger.

### Obsolete Or Superseded

- `docs/NEXT_STEPS.md` says to evaluate the after-blockers checkpoint next.
  That was superseded by balanced-retention production and then hierarchical v1
  production.
- `docs/PROJECT_HANDOFF.md` says the after-blockers checkpoint is the safest
  artifact and not registry-ready. It remains useful history, but current truth
  is the hierarchical v1 production handoff.
- Teacher-retention as the next architecture step is obsolete; it was tried and
  failed at 250k.
- Candidate-readiness notes saying registry tooling must be patched are
  obsolete; tooling was patched and used for candidate registration and active
  activation.
- The AWS-unavailable note is partially obsolete; a full-path AWS CLI later
  downloaded only approved small files.

## Current Risk Register

| Risk | Severity | Evidence | Current control |
| --- | --- | --- | --- |
| Action 24/32 concentration | Medium | Equal-budget eval and monitoring runbook | Monitor rates and pair with service/lateness/failure counters. |
| Fleet/reorder economics unknown | High | Public data cannot validate secondary fleet or reorder none | Company data contract required before calibration or training. |
| Blind scale-up overfits watches | High | nextgen and V-cycle history | No blind 3M; use trigger-based 1.5M/2M only. |
| Large research modules are hard to reason about | Medium | `env_5pl.py`, `train_joint_torch.py`, `scenario_metrics.py` size | Add artifact tooling and split only after stable behavior. |
| Baseline/registry/prod mutation mistakes | High | Production lifecycle sensitivity | Explicit approval gates, hashes, dry-runs, audit reports. |
| Eval budget mismatch causes wrong conclusions | Medium | Residual-watch statistical audit | Use equal-budget eval before training decisions. |
| Public route sample too small | Medium | Amazon small sample has 13 routes | Treat as method validation; request bounded larger sample only with approval. |

## Open Questions

1. What are the real company costs and constraints for secondary fleet usage?
2. Does reorder `none` remain economically valid under company stockout,
   backlog, holding-cost, and replenishment data?
3. Are action 24/32 concentrations stable in live production windows?
4. Should action concentration be regularized in a future `hierarchical_v2`, or
   is it acceptable once company economics are calibrated?
5. Should the production baseline be updated to hierarchical v1, and what
   separate approval/audit process should govern that?
6. What monitoring artifact format should become the operator-facing dashboard?
7. Which parts of `env_5pl.py` should eventually be split for maintainability
   without changing behavior?

## Recommended Roadmap

The detailed strategy plan is:

`docs/plans/20260611_codex_project_future_strategy_plan.md`

Roadmap tracks:

- Track A: production hardening.
- Track B: real-world calibration.
- Track C: trigger-based model research extension.
- Track D: architecture evolution.
- Track E: data and tooling.
- Track F: productization.
- Track G: safety and governance.

Priority order:

1. Production hardening and read-only audits.
2. Company data contract and KPI mapping.
3. Monitoring report generation for accepted watches.
4. Public route-proxy expansion only if bounded and approved.
5. Tooling and productization around registry/prod/eval artifact health.
6. Research extension only after monitored regression or company-data mismatch.

## What Not To Do

- Do not train now.
- Do not run offline eval or long-run gate from this roadmap task.
- Do not start 1.5M/2M/3M/5M/10M/100M without a new explicit gated plan.
- Do not mutate registry, production, baselines, DB, checkpoints, or existing
  eval outputs.
- Do not use failed checkpoint branches as parents.
- Do not use `--allow-empty-replay-resume` in a gated ladder.
- Do not call public route evidence validation of secondary_fleet or reorder
  economics.
- Do not update baselines just because hierarchical v1 is production-promoted.

## Next 10 Best Codex Tasks

1. Read-only production refresh audit: dashboard, active registry, production
   manifest, hashes, process scan.
2. Convert the monitoring runbook into a structured telemetry/KPI data
   dictionary.
3. Draft a company-data request package for order, route, fleet, inventory,
   dispatch failure, and cost fields.
4. Build a read-only monitoring report spec for action 24/32, service/lateness,
   dispatch success, no-current/no-unassigned, failed-noop, route failure, and
   no-vehicle counters.
5. Add a read-only artifact health CLI that summarizes registry, production
   manifests, eval verdicts, hashes, and residual watches.
6. Prepare a bounded Amazon route-proxy expansion plan with explicit size gate;
   do not download until approved.
7. Analyze Amazon route stress buckets against shortest and low-congestion
   proxies after a bounded dataset is approved.
8. Draft a baseline-update decision memo template, but do not update baselines.
9. Add rollback/audit dry-run tests around current hierarchical production
   paths and prior active pair references.
10. If monitoring or company data exposes a concrete issue, draft a gated
    1.5M/2M extension plan with exact-resume, residual gates, and rollback
    rules. Do not train as part of the draft.

## Final Classification

`FULL_PROJECT_CHRONOLOGY_AND_ROADMAP_READY`
