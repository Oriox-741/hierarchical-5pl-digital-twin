# Final Advisor Master Package - 2026-06-14

## Classification

`FINAL_ADVISOR_MASTER_PACKAGE_READY`

## Project in One Sentence

CODEX-5PL is a 5PL digital-twin decision-control project that combines continuous operational controls and a 48-action hierarchical tactical policy for dispatch, route family, fleet mode, and reorder decisions under the `physical_reality_v5_route_candidate_visibility` contract.

## What the Model Actually Does

The current production model `joint_torch_v5_prod_hierarchical_v1_1m_20260611` maps a 73-dimensional simulator observation to 5 continuous PPO controls and one DQN discrete action in `0..47`.

## Directly Tested Evidence

- Hierarchical v1 1M passed internal 250k, 500k, and 1M gates.
- Equal-budget residual-watch comparison against old production passed.
- Production artifacts are active/copy-promoted and protected by artifact-health/ops bundle checks.

## Public Proxy Evidence

- LaDe: rows `31415`, action24 `0.0`, action32 `0.0`, missingness `0.3051177454274308`
- NYC_HVFHS: rows `100000`, action24 `0.00051`, action32 `2e-05`, missingness `0.3013698630136986`
- Olist: rows `96476`, action24 `0.002062689166217505`, action32 `0.0`, missingness `0.5479511690607132`

This proves adapter feasibility, coverage/missingness reporting, and production policy action tendencies on public proxy states. It does not prove causal superiority.

## If Asked: Is This Real-World Ready?

Answer: it is simulator-production ready with protected artifacts and monitoring runbooks. It is not live TMS/WMS/ERP deployment ready until company-data replay, telemetry validation, and integration tests are approved and completed.
