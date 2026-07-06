# CODEX Model Lifecycle Governance Runbook

Date: 2026-06-11

Final classification: `STEP1_MODEL_LIFECYCLE_GOVERNANCE_READY`

## Scope

This runbook defines lifecycle approvals and mutation boundaries for CODEX
model work after hierarchical v1 1M production promotion.

It is a governance document only. It does not authorize training, offline eval,
long-run gates, dataset download, registry mutation, production mutation,
baseline mutation, DB mutation, checkpoint mutation, eval-output edits, or
training config creation.

## Current Production Reference

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- active PPO id: `17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- active DQN id: `f87e10d6-479f-44fc-99d1-6925bc9cb346`
- architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation/action: `73 / 48`
- status: active registry plus copy-only production promotion complete

## Approval Gate Matrix

| Operation | Requires explicit user approval | Required evidence before approval | Allowed mutation if approved | Independent review |
| --- | --- | --- | --- | --- |
| Read-only audit | no, unless user restricts | scope and protected paths | none | optional |
| Documentation/specification | usually no if requested | allowed write list | listed docs only | recommended for broad docs |
| Public metadata listing | yes if network/data risk unclear | estimated command and no-download proof | none | optional |
| Public dataset download | yes | size estimate, exact keys, target dir | approved fresh data dir only | recommended |
| Offline eval | yes | checkpoint, scenario dir, output dir, seed, episodes | approved fresh eval dir only | recommended |
| Long-run gate | yes | candidate/production summaries | gate result file only if approved | recommended |
| Training config creation | yes | design report, parent, constraints | approved config only | required for major configs |
| Training run | yes | config, parent, output dir, stop gates | approved fresh run dir/checkpoints | required after code/config patch |
| Candidate registration | yes | dry-run OK and metadata | append candidate rows only | required |
| Active registry activation | yes | dry-run OK and rollback plan | active rows plus active_models update only | required |
| Production copy/promotion | yes | active registry, eval/gate, hashes | target production dir and manifest only | required |
| Baseline update | yes | stable production and separate decision memo | approved baseline files only | required |
| DB mutation | yes | migration/rollback plan | approved DB changes only | required |
| Checkpoint mutation | yes | exceptional architecture decision | approved checkpoint operation only | required |

## Allowed Mutation Templates

Each mutation request should state:

- exact objective,
- exact files or directories allowed to change,
- exact files or directories forbidden,
- source artifact hashes,
- target artifact paths,
- rollback plan,
- verification commands,
- final classification options,
- whether independent review is required.

If the allowed mutation scope cannot be expressed precisely, the task should
not proceed.

## Forbidden Operations Without Separate Approval

- Updating `models/registry/active_models.json`.
- Appending to `models/registry/models.jsonl`.
- Creating or modifying `models/production/**`.
- Updating `models/baselines/**`.
- Mutating `db/**`.
- Mutating `models/checkpoints/**`.
- Editing existing `models/eval/**`.
- Downloading datasets.
- Running training.
- Running offline eval.
- Running long-run gates.
- Creating training configs.
- Starting 1.5M/2M/3M/5M/10M/100M.

## Registry Candidate Registration Rules

Candidate registration requires:

- dry-run command and `DRY_RUN_OK`,
- candidate checkpoint metadata verified,
- eval verdict `PASS`,
- hard blockers zero,
- long-run gate `PASS`,
- `production_ready=false`,
- `baseline_update=false`,
- no active registry write,
- explicit user approval for the real registration.

Allowed mutation after approval:

- append exactly the approved candidate rows to `models/registry/models.jsonl`.

Forbidden during candidate registration:

- no `active_models.json` write,
- no production copy,
- no baseline/DB/checkpoint mutation,
- no training/eval.

## Active Registry Activation Rules

Active registry activation requires:

- candidate rows exist and match expected ids,
- activation dry-run returns `DRY_RUN_OK`,
- old active pair is recorded for rollback,
- hierarchical metadata is preserved,
- explicit user approval for real activation.

Allowed mutation after approval:

- append exactly the approved active rows to `models/registry/models.jsonl`,
- update only the approved active roles in `models/registry/active_models.json`.

Forbidden during activation:

- no production copy,
- no baseline/DB/checkpoint mutation,
- no training/eval.

## Production Copy / Promotion Rules

Production copy requires:

- active registry points to the candidate artifacts,
- production-promotion plan exists,
- eval/gate/hard-blocker evidence is current,
- residual watches are accepted or blocked by the human approver,
- source file hashes recorded,
- target production dir is absent or explicitly approved for handling,
- explicit user approval for copy-only promotion.

Allowed mutation after approval:

- create the exact target production directory,
- copy exactly approved model artifacts,
- write `production_manifest.json`.

Forbidden during production copy:

- no registry mutation,
- no baseline mutation,
- no DB mutation,
- no source checkpoint mutation,
- no training/eval.

## Baseline Update Rules

Baseline update is not part of Step 1 and remains a separate future decision.

Baseline update requires:

- stable production monitoring evidence,
- business approval,
- exact baseline target list,
- rollback plan,
- no concurrent registry/production mutation unless separately approved.

## DB Mutation Rules

DB mutation requires:

- explicit migration plan,
- backup/rollback plan,
- data owner approval,
- privacy/access review where applicable,
- no model training side effects.

Documentation-only data request packages must not create tables or import data.

## Checkpoint Mutation Rules

Checkpoint mutation is forbidden by default.

Permitted checkpoint operations should normally be limited to:

- creating new checkpoints from an explicitly approved training run,
- copying approved checkpoints into a production directory under a copy-only
  promotion approval.

Direct mutation of existing checkpoint bodies requires an architecture decision
and explicit user approval.

## Eval Output Rules

Existing eval outputs are immutable audit evidence.

Future evals, when explicitly approved, must use fresh output directories and
must not overwrite old `models/eval/**` contents.

## Rollback Path

Rollback should be a separate approved operation unless an emergency policy
already grants authority.

Standard rollback:

1. Restore active registry to the prior production active pair.
2. Preserve registry rows for audit.
3. Quarantine production directory only with explicit approval.
4. Do not mutate baselines unless separately approved.
5. Do not mutate DB rows unless separately approved.
6. Run a read-only production/registry/runtime audit after rollback.

Prior production active artifact paths:

- PPO:
  `models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\ppo_torch_joint_final_balanced_retention_ft_200k.pt`
- DQN:
  `models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\dqn_torch_joint_final_balanced_retention_ft_200k.pt`

## No-Blind-Training Policy

No blind 3M/5M/10M/100M training should be started.

Reason:

- nextgen showed longer training can worsen behavior,
- V2/V2.5 showed continuation can migrate action-quality pockets,
- teacher retention failed at 250k,
- current hierarchical v1 passed 250k/500k/1M and equal-budget residual
  assurance,
- remaining gaps are primarily monitoring and company-data calibration.

## Trigger-Based Training Policy

Future 1.5M/2M research is only justified if a concrete trigger exists:

- repeated monitored service/lateness regression,
- action `24` or `32` concentration paired with operational degradation,
- no-current/no-unassigned/failed-noop reappears,
- company data contradicts route/fleet/reorder economics,
- route-choice evidence contradicts the current policy under comparable stress,
- a runtime/checkpoint bug is found and fixed with tests.

Any future training plan must include:

- exact-resume parent,
- no `--allow-empty-replay-resume`,
- fresh output dirs,
- intermediate gates,
- residual-watch gates,
- rollback plan,
- protected no-mutation proof,
- independent review.

## Exact-Resume Requirements

Exact-resume continuation requires:

- model state,
- optimizer state,
- DQN replay buffer state,
- replay cursor and size,
- RNG states,
- global step,
- curriculum position,
- metadata `exact_resume_capable=true`.

Weights-only initialization is not exact resume.

## Independent Review Requirements

Independent read-only review is required for:

- major code patches,
- training config readiness,
- candidate registration readiness,
- active activation readiness,
- production promotion readiness,
- broad lifecycle/governance docs,
- any future training or eval ladder plan.

Reviewer scopes must explicitly forbid training, eval, registry mutation,
production mutation, baseline mutation, DB mutation, checkpoint mutation, and
eval-output mutation unless the review task is separately approved to inspect
specific fresh outputs.

## Final Classifications

Use precise classifications:

- `LIFECYCLE_READ_ONLY_AUDIT_CLEAN`
- `LIFECYCLE_DOCS_READY`
- `LIFECYCLE_DRY_RUN_READY`
- `LIFECYCLE_WRITE_READY_FOR_MANUAL_APPROVAL`
- `LIFECYCLE_WRITE_COMPLETE_SAFE`
- `LIFECYCLE_NEEDS_FIXES`
- `LIFECYCLE_BLOCKED`

No classification automatically authorizes a protected mutation.

## Source Documents

- `docs/runs/20260611_full_project_chronology_and_future_roadmap.md`
- `docs/plans/20260611_codex_project_future_strategy_plan.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook.md`

## Final Classification

`STEP1_MODEL_LIFECYCLE_GOVERNANCE_READY`
