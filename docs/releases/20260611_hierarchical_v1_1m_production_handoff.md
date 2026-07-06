# Hierarchical V1 1M Production Handoff

Date: 2026-06-11

Final classification: `HIERARCHICAL_V1_1M_PRODUCTION_PROMOTION_COMPLETE_SAFE`

## Production Model

- Logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Runtime family: `torch_joint`
- DQN architecture: `hierarchical_v1`
- Hierarchical initialization method: `flat_teacher_distillation_v1`
- Metadata key `dqn_architecture`: `hierarchical_v1`
- Metadata key `hierarchical_init_method`: `flat_teacher_distillation_v1`
- Exact resume capable: `true`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`
- Continuous action dimension: `5`
- External discrete action count: `48`
- Training step: `1000000`

Production directory:

`models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611`

## Production Files

| File | SHA256 | Source Match |
| --- | --- | --- |
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` | yes |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` | yes |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` | yes |
| `production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` | n/a |

The production promotion was copy-only. It created the production directory, copied exactly the three `.pt` artifacts above, and wrote `production_manifest.json`.

## Registry IDs

| Role | Algorithm | Status | Registry ID |
| --- | --- | --- | --- |
| `continuous_control` | PPO | active | `17ba1d28-0054-4f7c-ae9a-34cd305ebb89` |
| `tactical_dispatch` | DQN | active | `f87e10d6-479f-44fc-99d1-6925bc9cb346` |
| `continuous_control` | PPO | candidate | `3befa11c-e853-4d0f-a951-c2fbbb2a9898` |
| `tactical_dispatch` | DQN | candidate | `71af6701-ac3a-40f2-bdd9-cc80868d5a1c` |

Active registry mappings after activation:

```json
{
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt",
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt"
}
```

Production promotion did not mutate the registry. The active registry was already routed to this model before the production copy.

## Ladder Summary

The hierarchical ladder passed each gate:

- 250k: `PASS`, gate exit code `0`
- 500k: `PASS`, gate exit code `0`
- 1M: `PASS`, gate exit code `0`

Lineage:

- 250k used the flat production checkpoint only as a read-only teacher/init source through `--init-from-joint-checkpoint`.
- 500k used exact `--resume` from the 250k hierarchical checkpoint.
- 1M used exact `--resume` from the 500k hierarchical checkpoint.
- `--allow-empty-replay-resume` was never used.

## 1M Evaluation Summary

Evaluation output:

`models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`

Summary:

- Scenario verdicts: `8/8 PASS`
- Episode rows: `160`
- Hard blockers: `zero`
- Long-run gate: `PASS`
- Gate fatal failures: none

Scenario comparison values are `service / lateness / dispatch / success / no-current+no-unassigned`.

| Scenario | Production | Hierarchical 1M | Verdict |
| --- | ---: | ---: | --- |
| `baseline_normal` | 0.938 / 0.000 / 0.487 / 1.000 / 0 | 0.990 / 0.000 / 0.372 / 1.000 / 0 | PASS |
| `demand_spike_volatility` | 0.843 / 0.177 / 0.728 / 1.000 / 0 | 0.859 / 0.027 / 0.793 / 1.000 / 0 | PASS |
| `high_holding_cost` | 0.943 / 0.000 / 0.387 / 1.000 / 0 | 0.966 / 0.000 / 0.449 / 0.999 / 2 | PASS |
| `lead_time_volatility` | 0.936 / 0.000 / 0.455 / 1.000 / 0 | 1.000 / 0.000 / 0.360 / 0.999 / 1 | PASS |
| `mixed_stress` | 0.951 / 0.000 / 0.911 / 0.996 / 6 | 0.942 / 0.000 / 0.987 / 0.999 / 8 | PASS |
| `premium_sla_pressure` | 0.980 / 0.000 / 0.427 / 0.769 / 178 | 1.000 / 0.000 / 0.357 / 0.999 / 1 | PASS |
| `route_disruption_congestion` | 0.929 / 0.000 / 0.575 / 0.822 / 171 | 0.901 / 0.008 / 0.499 / 0.999 / 2 | PASS |
| `vehicle_scarcity_capacity_shock` | 0.914 / 0.000 / 0.731 / 1.000 / 0 | 0.949 / 0.001 / 0.806 / 1.000 / 0 | PASS |

## Accepted Residual Watches

Human approver: `egeme`

The following residual watches were explicitly accepted before production copy:

- `route_disruption_congestion` service below production.
- `mixed_stress` service below production.
- top-action concentration warnings.
- mixed-success route-failure warnings.

These were accepted as nonfatal because scenario thresholds and the long-run gate passed. The production manifest records `residual_watches_accepted=true`.

## Runtime Smoke

Read-only runtime smoke after production copy:

- `load_torch_joint_policy` loaded `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt` on CPU.
- Deterministic zero-observation prediction returned continuous length `5`.
- Continuous values were finite and bounded in `[-1, 1]`.
- Discrete action was `1`, valid in `0..47`.
- `PolicyService.predict_joint(..., fallback_to_heuristic=False)` returned `algorithm=torch_joint`.
- SB3 PPO loader calls: `0`.
- SB3 DQN loader calls: `0`.

## Protected No-Mutation Proof

Promotion-scope mutation:

- Created `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611`.
- Copied exactly three approved `.pt` files.
- Wrote exactly `production_manifest.json`.

Protected areas:

- Baselines unchanged: `models/baselines` remained `2692` files, `7392576274` bytes.
- DB unchanged: `db` remained `7` files, `28330` bytes.
- Source checkpoints unchanged:
  - prior production source checkpoint hash remained `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
  - hierarchical source joint hash remained `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
  - hierarchical source PPO hash remained `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996`
  - hierarchical source DQN hash remained `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D`
- Registry unchanged by production copy:
  - `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
  - `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- No train/eval/gate process was running after promotion.

No training or offline evaluation was run during production promotion.

## Rollback Summary

If rollback is required:

1. Restore `models/registry/active_models.json` to the prior active pair:
   - `ppo:continuous_control -> models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\ppo_torch_joint_final_balanced_retention_ft_200k.pt`
   - `dqn:tactical_dispatch -> models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\dqn_torch_joint_final_balanced_retention_ft_200k.pt`
2. Preserve candidate and active rows in `models/registry/models.jsonl` for audit unless a separate cleanup plan is approved.
3. Quarantine `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611` only with explicit approval.
4. Do not mutate baselines.
5. Do not mutate DB rows.
6. Rerun a read-only registry/runtime/production audit after rollback.

## Non-Goals

- No baseline update.
- No DB cleanup.
- No registry mutation during production copy.
- No source checkpoint mutation.
- No 3M, 5M, 10M, or 100M run.

## Recommended Next Actions

- Monitor production behavior with special attention to route disruption, mixed stress, top-action concentration, and mixed-success route-failure warnings.
- Consider a baseline update only under separate explicit approval.
- Consider 3M research only under a new gated plan with explicit approval.
- Keep future registry, baseline, DB, and checkpoint mutations separate from this production handoff.

## Source Documents

- `docs/runs/20260611_hierarchical_v1_production_promotion_audit.md`
- `docs/runs/20260611_hierarchical_v1_active_registry_activation_audit.md`
- `docs/runs/20260611_hierarchical_v1_candidate_registration_audit.md`
- `docs/runs/20260611_hierarchical_dqn_ladder_report.md`
- `docs/runs/20260611_hierarchical_v1_1m_candidate_review_readiness.md`
- `docs/runbooks/20260611_hierarchical_v1_production_promotion_plan.md`

## Final Classification

`HIERARCHICAL_V1_1M_PRODUCTION_PROMOTION_COMPLETE_SAFE`
