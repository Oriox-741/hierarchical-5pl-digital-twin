# Final Benchmark Synthesis - 2026-06-14

## Classification

`FINAL_CODEX_5PL_EVIDENCE_SYNTHESIS_READY_WITH_LIMITATIONS`

## Production Truth

`joint_torch_v5_prod_hierarchical_v1_1m_20260611` is the active/copy-promoted hierarchical v1 production model under the 73/5/48 CODEX-5PL contract.

## Large Public Replay

```json
{
  "LaDe": {
    "rows": 31415,
    "action_24_rate": 0.0,
    "action_32_rate": 0.0,
    "missingness": 0.3051177454274308,
    "confidence": {
      "high": 31415
    }
  },
  "NYC_HVFHS": {
    "rows": 100000,
    "action_24_rate": 0.00051,
    "action_32_rate": 2e-05,
    "missingness": 0.3013698630136986,
    "confidence": {
      "high": 100000
    }
  },
  "Olist": {
    "rows": 96476,
    "action_24_rate": 0.002062689166217505,
    "action_32_rate": 0.0,
    "missingness": 0.5479511690607132,
    "confidence": {
      "medium": 96476
    }
  }
}
```

## Scorecard

Final integrated scorecard classification: `FINAL_INTEGRATED_SCORECARD_READY_WITH_LIMITATIONS` with `13` dimensions.

## Blockers

Final blocker register contains `9` external blockers. These are documented limitations, not hidden failures.

## Safe Claim

CODEX-5PL has a protected simulator-production candidate and broad benchmark/public-proxy evidence package.

## Unsafe Claims

Do not claim global SOTA, live TMS/WMS/ERP deployment, causal public-data OPE, or company-data validation without company data.
