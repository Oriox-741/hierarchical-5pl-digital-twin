# Hierarchical DQN Initialization Decision

Date: 2026-06-11

## Decision

Recommended initialization path: `flat_teacher_distillation_v1`.

Final decision before patch review: use the protected flat production checkpoint only as a read-only teacher and compatible PPO/trunk source. Do not treat it as an exact resume parent for `hierarchical_v1`.

## Option Comparison

### A. Random hierarchical heads with transferred shared/trunk weights

Status: fallback only.

This preserves some representation from the flat production DQN trunk, but the route/dispatch/mode/reorder heads start with arbitrary tactical preferences. Because the current failure mode is action-pocket drift, random heads are too likely to create an immediate new pocket before RL corrects them.

### B. Flat production DQN to hierarchical distillation warm-start

Status: selected.

This is the safest path because the approved `hierarchical_v1` network still returns external `(batch, 48)` Q values. A frozen flat production teacher can supervise the hierarchical student on a finite sampled observation bank without touching env training, replay, registry, production, baselines, DB, or checkpoints.

The warm-start remains explicit and non-resume:

- fresh optimizer,
- empty replay,
- fresh RNG at training start,
- metadata records `hierarchical_init_method=flat_teacher_distillation_v1`,
- flat/hierarchical exact resume remains rejected.

### C. Analytical flat-to-factorized conversion

Status: rejected.

The additive decomposition is not uniquely identifiable from a flat 48-logit head. Many dispatch/route/mode/reorder component assignments can produce the same external Q vector, so this would create hidden arbitrary assumptions.

### D. Hierarchical from scratch from production PPO only

Status: rejected.

This keeps strategic PPO behavior but discards the operational DQN policy that is still the best available parent. It is higher regression risk than distillation and less informative than a behavior-preserving warm-start.

## Readiness Criteria

Before creating the first hierarchical 250k config:

- tests must prove flat checkpoints still reject hierarchical exact resume,
- flat teacher loading must be read-only and frozen,
- hierarchical student outputs must remain external `(batch, 48)`,
- distillation loss must be finite and bounded,
- a tiny supervised step must be able to match a flat teacher on a synthetic state bank,
- checkpoint metadata must record the initialization method,
- runtime must still return legal external action IDs,
- one read-only GPT-5.5 reviewer must approve the patch.

## Training Status

No training, offline eval, registry update, production mutation, baseline mutation, DB mutation, source production checkpoint mutation, existing checkpoint mutation, 250k, 500k, or 1M run was authorized by this decision.
