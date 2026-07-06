# Governance

## Protected Artifact Policy

The original project treats these as protected: model registry, production models, checkpoints, baselines, databases, raw datasets, and existing evaluation outputs. This sanitized repository excludes those artifacts and must not be used to mutate them.

## Registry and Production Boundary

Model registration, activation, production copy, baseline update, and database mutation require separate explicit approval in the original controlled workspace. They are not part of this sanitized GitHub package.

## Monitoring and Rollback

Operational monitoring and rollback notes are documented in selected runbooks. The repository preserves the simulator/proxy evidence boundary: warnings are tracked, but live operational claims require company telemetry and human approval.

## Claim Boundaries

Do not claim live deployment, company-data validation, causal public replay superiority, or global SOTA. Public replay is descriptive proxy evidence.
