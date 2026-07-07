# Advisor Quick Read

## Project

Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration.

## One-sentence summary

This project studies hybrid continuous-discrete control for simulated 5PL logistics orchestration by combining PPO continuous controls with a hierarchical DQN tactical action surface for dispatch, route, fleet, and replenishment decisions.

## What to inspect first

1. `README.md` for scope and safe checks.
2. `MODEL_CARD.md` for model identity and claim boundaries.
3. `REPRODUCIBILITY.md` for safe verification and excluded artifacts.
4. `reports/thesis_handoff/` for method evidence truth tables.
5. `reports/benchmarks/` for selected benchmark summaries.
6. `docs/ablation_protocol.md` after it is added.

## What is intentionally not claimed

- No live TMS/WMS/ERP deployment.
- No company-specific validation.
- No causal off-policy evaluation from public replay.
- No global state-of-the-art claim.
- No claim that the public repository contains private datasets, checkpoint binaries, or production registry state.

## Current best research extension

The next extension is a formal ablation study plus company-data calibration pathway: neutral PPO controls, flat-vs-hierarchical DQN, no teacher distillation, masked route-candidate visibility, safety-projection diagnostics, reward-blend sensitivity, and logged-decision/off-policy evaluation readiness.
