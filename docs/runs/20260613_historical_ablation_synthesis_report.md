# Historical Ablation Synthesis

Decision: `HISTORICAL_ABLATION_SYNTHESIS_READY`

This is historical ablation-style evidence from existing artifacts only; no fresh ablation training was run.

## Branch Table

| Branch | Method | Scenario pass | Gate | Failure mode | Conclusion |
| --- | --- | ---: | --- | --- | --- |
| `flat_balanced_retention_production` | flat PPO+DQN balanced-retention fine-tune | 8/8 | PASS | none at promotion; residual watches only | Old flat production was a valid parent/comparator but had route dispatch-success and premium no-work weaknesses. |
| `nextgen_300k_1m` | longer flat continuation from production | 300k 2/8, 1M 6/8 | FAIL | premium and route action-quality drift | More flat training without stronger architecture/gates was not sufficient. |
| `stability_v2_v2_1_v2_2` | three bounded flat reward/curriculum cycles | 4/8, 4/8, 5/8 | FAIL | high-holding, lead-time, vehicle, and route no-work/action-pocket failures | Reward/curriculum tuning showed whack-a-mole action-pocket migration. |
| `v2_3_250k` | flat DQN with route/action semantics repairs | 8/8 scenario thresholds | FAIL | route action 32 no-useful-work / failed-noop action-quality failure | Scenario threshold pass alone was not enough; long-run action-quality gate was necessary. |
| `v2_4_250k_500k_legacy_continuation` | semantics-fixed 250k then legacy continuation | 250k 8/8, 500k 7/8 | 500k FAIL | continuation resumed without DQN replay/RNG; premium and route quality drift | Missing replay/RNG restoration was a confirmed continuation architecture bug. |
| `exact_resume_patch` | persist/restore DQN replay and Python/NumPy/Torch RNG on --resume | code-level patch | review approved | legacy checkpoints rejected unless explicitly allowed | Exact resume became mandatory for valid continuation, but it was necessary not sufficient. |
| `v2_5_exact_resume` | new exact-resume capable 250k then exact 500k | 250k 8/8, 500k 7/8 | 500k FAIL | route service fell below threshold; lead-time action 33 no-current/no-unassigned explosion | Exact replay/RNG fixed one bug but flat composite action drift remained. |
| `teacher_retention_250k` | flat teacher-retention / behavior anchoring | 4/8 PASS by gate table | FAIL | baseline/high-holding/lead-time/route service and lateness regressions | Teacher retention was too weak/indirect and did not preserve passing behavior. |
| `hierarchical_v1_ladder` | hierarchical DQN heads + flat_teacher_distillation_v1 + exact resume ladder | 250k PASS, 500k PASS, 1M PASS | PASS at all rungs | nonfatal residual watches only | Factorized DQN architecture plus distillation/exact resume solved the observed flat action-pocket instability under current gates. |
| `equal_budget_residual_watch` | fresh equal-budget old production vs hierarchical production eval | 8/8 for both; hierarchical gate PASS | PASS | route action 32 and mixed action 24 concentration remain watches | Residual watches are statistically/operationally acceptable under equal budget. |

## Comparison Axes

- `flat_dqn_vs_hierarchical_v1`: Flat 48-action DQN repeatedly shifted into composite action pockets (32/33/41/45/46), while hierarchical_v1 passed 250k/500k/1M and equal-budget assurance.
- `exact_resume_off_vs_on`: V2.4 legacy continuation exposed the empty replay/RNG bug; V2.5 exact resume improved premium/high-holding/route dispatch-success but still failed 500k, proving exact resume is necessary but not sufficient.
- `teacher_retention_vs_hierarchical`: Teacher retention failed at 250k despite active retention telemetry; hierarchical_v1 then passed the full ladder, making hierarchical architecture the stronger causal fix.
- `reward_patch_vs_architecture`: Reward/config patches repaired local pockets but failures migrated; architecture-level action factorization addressed the repeated failure pattern more robustly.
- `250k_vs_500k_vs_1m`: Flat ladders often degraded after 250k; hierarchical exact-resume ladder preserved pass status at 250k, 500k, and 1M.
- `public_route_validation`: Amazon/OR-Tools route benchmarks support route-side plausibility only and cannot validate fleet or reorder economics.

## Overall Conclusion

The strongest causal story is cumulative: exact resume fixed a real continuation bug, but the decisive product-quality improvement came from hierarchical_v1 action factorization plus flat-teacher distillation and exact-resume laddering. Further blind training is not recommended without a monitored trigger or company-data mismatch.

## Limitations

- Not a randomized controlled ablation.
- Some branches differ in reward/config and chronology, so causal claims are evidence-weighted rather than strictly isolated.
- Company fleet/reorder economics remain unvalidated.
