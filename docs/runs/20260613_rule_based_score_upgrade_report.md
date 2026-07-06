# Rule-Based Continuous Score Upgrade Report

## Decision

`RULE_BASED_SCORE_DIAGNOSTIC_ONLY_NO_UPGRADE`

## Evidence

Diagnostic artifact:

- `reports/benchmarks/scorecard_upgrade_20260613/rule_based_continuous_diagnostics/rule_continuous_diagnostics.json`

The full continuous-aware benchmark supplied `5120` episode rows. The heuristic continuous mode changed control fields such as `capacity_buffer_fraction` and `safety_stock_multiplier`, but the heuristic-minus-neutral summary deltas were all `0.0` for mean service, worst service, dispatch success, and action-24 rate.

## Interpretation

The current heuristic is bounded and non-leaky, but it does not improve benchmark discrimination in the existing scenario set. A heuristic v2 would be speculative outcome tuning unless a scenario/control-sensitivity bug is first demonstrated.

## Score Result

| Area | Prior | Updated | Reason |
| --- | ---: | ---: | --- |
| Rule-based baseline score | 4.0 / 5 | 4.0 / 5 | Diagnostics clarified the limiter, but no stronger behavioral evidence was produced. |

## Next Safe Step

If this score must improve, build a dedicated sensitivity test for continuous controls before changing formulas. Do not tune against the current aggregate benchmark alone.

