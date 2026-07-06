# Final Demo Script and Q&A - 2026-06-14

## Classification

`FINAL_DEMO_SCRIPT_AND_QA_READY`

## Demo Flow

1. Show the production truth: hierarchical v1 1M, 73 input features, 5 continuous outputs, 48 discrete actions.
2. Explain action decomposition with action 24 and action 32 as watch examples.
3. Show internal gate/equal-budget evidence.
4. Show public benchmark families: route, inventory, fleet/dispatch, public data expansion.
5. Show historical replay v0 large samples and missingness/confidence.
6. Close with company-data bridge: what remains needed for causal OPE.

### Is this global SOTA?

No. It is a SOTA pathway for a custom 5PL joint-control benchmark, not a global SOTA claim across all routing/inventory/fleet literature.

### Does public replay prove the model would beat real dispatchers?

No. Public replay is descriptive proxy evidence. It lacks behavior propensities, alternatives, and company reward definitions.

### Is it real-world ready?

It is simulator-production ready and artifact-protected. Live readiness requires company replay, telemetry, integration, and safety validation.
