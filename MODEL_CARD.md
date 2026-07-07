# Model Card

## Model Identity

- Logical model: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Runtime: `torch_joint`
- DQN architecture: `hierarchical_v1`
- Hierarchical initialization: `flat_teacher_distillation_v1`
- Simulator contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: 73
- PPO controls: 5
- DQN actions: 48
- Training step recorded in evidence: 1,000,000

## Intended Use

Research, reproducibility review, simulator-based decision-policy analysis, thesis evidence inspection, and bounded public-proxy benchmarking.

## Out-of-Scope Use

This artifact is not a live logistics deployment, not a TMS/WMS/ERP integration, not a replacement for operational dispatch staff, not a validated company-data model, and not a global SOTA claim.

## Limitations

Evidence is simulator-grounded. Public replay data is descriptive proxy evidence and lacks the full state, action propensities, rewards, and operational contracts required for causal off-policy evaluation.

## Ablation Status

A formal ablation protocol exists, and evaluation-time ablations are scaffolded for neutral PPO controls, route-candidate masking, safety-projection diagnostics, and flat-vs-hierarchical comparison when predecessor artifacts are available. Training-time ablations such as no-teacher-distillation and reward-blend sensitivity remain planned future work. No completed ablation results are claimed unless generated reports are present.

## Governance Notes

Production artifacts, registry files, checkpoints, baselines, databases, and raw datasets are intentionally excluded. Promotion and rollback procedures are documented in the governance/runbook material, not automated by this repository.
