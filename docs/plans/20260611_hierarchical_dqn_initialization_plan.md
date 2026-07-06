# Hierarchical DQN Initialization Plan

Date: 2026-06-11

## Goal

Prepare `hierarchical_v1` for the first 250k candidate without pretending the flat production checkpoint is an exact-resume-compatible parent.

## Decision

Use explicit flat-teacher distillation warm-start:

- Load the production flat DQN checkpoint as a frozen, read-only teacher.
- Build a `hierarchical_v1` student.
- Copy compatible PPO weights and compatible DQN trunk weights from the production checkpoint.
- Train only the hierarchical DQN internals against frozen teacher external 48-action Q values on a finite sampled observation bank before RL starts.
- Start normal RL afterward with fresh optimizer, empty replay, fresh RNG, and metadata marking the checkpoint as a warm-started hierarchical candidate.

## Option Comparison

- Random hierarchical heads with transferred trunk: lower implementation risk but high behavioral discontinuity risk.
- Flat production DQN to hierarchical distillation: safest first implementation because it preserves the external 48-Q behavior surface while keeping flat/hierarchical checkpoint loading explicit.
- Analytical flat-to-factorized conversion: rejected because the additive decomposition is not uniquely identifiable from a flat 48-logit head.
- Hierarchical from scratch from production PPO only: rejected because it discards the current operational DQN tactical policy.

## Implementation Scope

- Add TDD coverage for read-only flat teacher loading, bounded distillation loss, tiny supervised action matching, init metadata, and flat/hierarchical exact-resume rejection.
- Add an explicit config section for hierarchical initialization.
- Create the first hierarchical 250k config only after code tests and independent read-only review pass.

## Non-Goals

- No training.
- No offline eval.
- No registry, production, baseline, DB, source checkpoint, or existing checkpoint mutation.
- No implicit flat-to-hierarchical load.
- No 500k or 1M artifact creation.
