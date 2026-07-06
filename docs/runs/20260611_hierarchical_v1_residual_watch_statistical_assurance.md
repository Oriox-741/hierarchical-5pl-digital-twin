# Hierarchical V1 Residual-Watch Statistical Assurance

Date: 2026-06-11

Pre-review classification: `HIERARCHICAL_V1_RESIDUAL_WATCHES_NEED_EQUAL_BUDGET_EVAL`

## Scope

This was a read-only statistical assurance audit for the current hierarchical v1 1M production model:

`joint_torch_v5_prod_hierarchical_v1_1m_20260611`

No training, offline evaluation, registry update, production mutation, baseline mutation, DB mutation, checkpoint mutation, or existing eval-output edit was performed. The only write was this report.

## Source Artifacts

- `docs/00_PROJECT_DASHBOARD.md`
- `docs/runs/20260611_hierarchical_v1_production_assurance_audit.md`
- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `docs/runs/20260611_hierarchical_dqn_ladder_report.md`
- Production comparator eval: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios`
- Hierarchical 1M eval: `models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`

## Current Production State

The active registry still points to the hierarchical v1 1M final artifacts:

```json
{
  "ppo:continuous_control": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt",
  "dqn:tactical_dispatch": "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt"
}
```

Production directory contents remain exactly:

- `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `joint_torch_latest.pt`
- `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `production_manifest.json`

No training, offline evaluation, or gate process was running at audit time.

## Method

The statistical comparison used existing `episode_metrics.jsonl` only:

- production comparator: `24` total episode rows, `3` per scenario
- hierarchical 1M: `160` total episode rows, `20` per scenario
- all count-like fields were interpreted by per-episode mean and per-step normalized rate, not raw equal-sample totals
- service residuals were checked with exact label-permutation tests over the existing samples and deterministic percentile bootstrap intervals
- route, fleet, reorder, and action distributions were aggregated by scenario and normalized by scenario step counts

Important limitation: the historical production comparator has only `3` episodes per scenario. That makes a strict statistical assurance verdict underpowered even when operational evidence is clean.

## Scenario-Level Normalized Comparison

Values below are per-episode means unless marked `per-step`.

| Scenario | n P/H | service P->H | late P->H | dispatch P->H | success P->H | no-work per-step P->H | delivered per-step P->H | routefail per-step P->H | no-vehicle per-step P->H | already-assigned per-step P->H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_normal` | 3/20 | 0.938->0.990 | 0.000->0.000 | 0.487->0.372 | 1.000->1.000 | 0.000000->0.000000 | 0.539->0.505 | 0.001157->0.000521 | 0.001157->0.000174 | 0.256->0.193 |
| `demand_spike_volatility` | 3/20 | 0.843->0.859 | 0.177->0.027 | 0.728->0.793 | 1.000->1.000 | 0.000000->0.000000 | 1.289->1.547 | 0.000000->0.002257 | 0.084491->0.144618 | 0.510->0.541 |
| `high_holding_cost` | 3/20 | 0.943->0.966 | 0.000->0.000 | 0.387->0.449 | 1.000->0.999 | 0.000000->0.000347 | 0.462->0.507 | 0.000000->0.001389 | 0.002315->0.000000 | 0.161->0.240 |
| `lead_time_volatility` | 3/20 | 0.936->1.000 | 0.000->0.000 | 0.455->0.360 | 1.000->0.999 | 0.000000->0.000174 | 0.509->0.542 | 0.001157->0.000347 | 0.000000->0.000347 | 0.212->0.080 |
| `mixed_stress` | 3/20 | 0.951->0.942 | 0.000->0.000 | 0.911->0.987 | 0.996->0.999 | 0.006944->0.001389 | 1.442->1.457 | 0.001157->0.001736 | 0.052083->0.050868 | 0.558->0.659 |
| `premium_sla_pressure` | 3/20 | 0.980->1.000 | 0.000->0.000 | 0.427->0.357 | 0.769->0.999 | 0.206019->0.000174 | 0.522->0.538 | 0.000000->0.000347 | 0.000000->0.000347 | 0.124->0.078 |
| `route_disruption_congestion` | 3/20 | 0.929->0.901 | 0.000->0.008 | 0.575->0.499 | 0.822->0.999 | 0.197917->0.000347 | 0.554->0.508 | 0.001157->0.001389 | 0.000000->0.000000 | 0.235->0.260 |
| `vehicle_scarcity_capacity_shock` | 3/20 | 0.914->0.949 | 0.000->0.001 | 0.731->0.806 | 1.000->1.000 | 0.000000->0.000000 | 0.861->0.868 | 0.000000->0.001389 | 0.002315->0.000347 | 0.380->0.447 |

Interpretation:

- Hierarchical 1M improves service in six of eight scenarios.
- The major production failure modes that motivated the ladder remain fixed: premium no-work and route-disruption no-work are almost eliminated on a per-step basis.
- The two service regressions are confined to `route_disruption_congestion` and `mixed_stress`.
- Route-failure raw counts are higher for hierarchical 1M, but normalized per-step rates remain small.

## Residual Watch Statistics

### Route Disruption

`route_disruption_congestion` service:

- production n=3: mean `0.929`, sd `0.061`, min/median/max `0.873 / 0.921 / 0.994`
- hierarchical n=20: mean `0.901`, sd `0.016`, min/q25/median/q75/max `0.876 / 0.889 / 0.900 / 0.909 / 0.942`
- observed delta: `-0.028`
- exact permutation p over existing samples: `0.085`
- bootstrap 95% interval for H-P delta: `[-0.091, 0.026]`
- common-seed subset deltas for seeds 92-94: `+0.0346`, `-0.0850`, `-0.0093`; mean `-0.0199`

Operational counters:

- no-work per-step improves from `0.197917` to `0.000347`
- dispatch success improves from `0.822` to `0.999`
- routefail per-step is nearly flat: `0.001157` to `0.001389`
- no-vehicle per-step remains `0.000000`
- already-assigned per-step moves from `0.235` to `0.260`
- worst hierarchical service episodes: `0.876`, `0.878`, `0.883`
- worst hierarchical lateness reaches `0.080`

Action watch:

- production action `32` attempts in route disruption: `0`
- hierarchical action `32` attempts: mean `136.85` per episode, `0.475` per step
- action `32` is therefore a real policy-style shift, not an artifact of raw unequal sample totals
- action `32` does not carry no-current/no-unassigned/failed-noop evidence in this scenario

Interpretation:

The route service drop is operationally meaningful enough to keep as a monitor. It is not a hard production-risk finding because the scenario and long-run gates pass, no-work is dramatically better, dispatch success is materially better, and route failures are low after normalization. Statistically, the current comparator is underpowered: the bootstrap interval includes zero and the exact permutation p-value does not cross a conventional 0.05 threshold.

### Mixed Stress

`mixed_stress` service:

- production n=3: mean `0.951`, sd `0.0047`, min/median/max `0.947 / 0.951 / 0.956`
- hierarchical n=20: mean `0.942`, sd `0.0106`, min/q25/median/q75/max `0.925 / 0.933 / 0.944 / 0.949 / 0.966`
- observed delta: `-0.009`
- exact permutation p over existing samples: `0.177`
- bootstrap 95% interval for H-P delta: `[-0.015, -0.002]`
- common-seed subset deltas for seeds 112-114: `-0.0073`, `+0.0006`, `+0.0028`; mean `-0.0013`

Operational counters:

- no-work per-step improves from `0.006944` to `0.001389`
- dispatch success improves from `0.996` to `0.999`
- routefail per-step moves from `0.001157` to `0.001736`
- no-vehicle per-step is essentially flat: `0.052083` to `0.050868`
- already-assigned per-step increases from `0.558` to `0.659`
- worst hierarchical service episodes: `0.925`, `0.929`, `0.930`

Action watch:

- production action `24` attempts in mixed stress: `0`
- hierarchical action `24` attempts: mean `167.10` per episode, `0.580` per step
- action `24` is a real concentration shift
- the concentration does not coincide with no-current/no-unassigned/failed-noop explosion

Interpretation:

The mixed service delta is small. Existing evidence does not show operational risk: lateness is effectively zero, no-work improves, dispatch success improves, no-vehicle is flat, and all gates pass. The action `24` concentration is real, but current evidence points to a concentrated policy preference rather than a runtime or reward bug. Because the production comparator has only three episodes, equal-budget confirmation is still the clean next statistical step.

## Distribution Findings

H1M is more concentrated than prior production in route, fleet, reorder, and action choices:

- `route_disruption_congestion`: production route mix is `high_resilience 76%`, `shortest 22%`, `low_congestion 2%`; H1M is `shortest 51%`, `low_congestion 49%`, `high_resilience 0.3%`.
- `mixed_stress`: production route mix is `low_congestion 51%`, `high_resilience 42%`, `shortest 7%`; H1M is `shortest 71%`, `low_congestion 29%`, `high_resilience 0.1%`.
- `mixed_stress`: production reorder mix includes `conservative 52%`, `none 36%`, `emergency 12%`; H1M is effectively `none 99.9%`.
- `route_disruption_congestion`: production fleet mix is `secondary 53%`, `primary 47%`; H1M is `secondary 98.5%`, `primary 1.5%`.

These shifts explain the watch shape: hierarchical v1 is much more decisive, but less diversified in the two watched stress regimes. The evidence does not indicate loader mismatch, invalid action IDs, SB3 fallback, missing hierarchical heads, missing checkpoint metadata, or a hard-blocker credit leak.

## Bug Assessment

No hidden bug is currently suspected.

Evidence against a bug:

- final production state audit was clean
- production assurance audit found runtime path `algorithm=torch_joint`, SB3 loader calls `0`, and legal discrete actions
- checkpoint metadata and production manifest matched hierarchical v1 contract
- all hard blockers were zero
- route-disruption and premium no-work regressions were fixed rather than reproduced
- action concentrations are coherent with distribution shifts and not paired with no-current/no-unassigned/failed-noop explosions
- current protected hashes match the prior clean production state

The residual watches should be treated as policy-behavior uncertainty, not as known implementation defects.

## Statistical Assurance Decision

The residual watches are operationally acceptable under current gate evidence, but not statistically closed.

Decision:

`RESIDUAL_WATCHES_NEED_EQUAL_BUDGET_EVAL`

Rationale:

- `route_disruption_congestion` has a visible `-0.028` service delta and route action `32` concentration.
- `mixed_stress` has a small `-0.009` service delta and action `24` concentration.
- Both scenarios pass gates and show no hard blocker.
- Existing production comparator budget is only `3` episodes per scenario, while H1M has `20`.
- Raw counts are not equal-budget comparable.
- Normalized counters reduce the severity, but equal-budget evidence is needed before calling the watches fully statistically acceptable.

## 3M Recommendation

Do not start 3M now.

The current evidence does not justify blind scale-up. More training could entrench the concentrated route/action preferences before confirming whether the residuals are real under equal seeds and equal episode budgets.

Safe next research step, if approved separately:

1. Run a fresh equal-budget assurance eval for old production and hierarchical production.
2. Use the same scenarios, same episode count, same seeds, and fresh output directories.
3. Do not update registry, baselines, DB, checkpoints, or existing eval outputs.
4. If equal-budget eval confirms operationally meaningful route/mixed service regression, design a gated `1.5M/2M` extension plan with explicit residual gates.
5. If equal-budget eval clears the residuals, retain current production and keep monitoring only.

## Protected No-Mutation Proof

Protected hashes at audit time:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- production joint: `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`
- production PPO: `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996`
- production DQN: `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D`
- production manifest: `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

Protected path profiles:

| Path | File count | Total bytes | Status |
| --- | ---: | ---: | --- |
| `models/baselines` | `2692` | `7392576274` | unchanged |
| `db` | `7` | `28330` | unchanged |
| `models/checkpoints` | `787` | `4920830666` | unchanged |
| `models/eval` | `120` | `354070196` | unchanged |

Process scan:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

## Dashboard Status

No dashboard edit was needed. `docs/00_PROJECT_DASHBOARD.md` already points to the final release handoff and current hierarchical production checkpoint. This report adds a statistical assurance note but does not change the production state.

## Independent Review

Reviewer verdict:

```text
RESIDUAL_WATCHES_NEED_EQUAL_BUDGET_EVAL
```

Reviewer rationale summary:

- Production comparator has `24` rows total, `3` per scenario; hierarchical has `160` rows total, `20` per scenario.
- The report correctly normalizes action and counter comparisons by episodes and steps.
- Route disruption service delta `-0.028` is real but underpowered; reviewer reproduced `p=0.085` and an interval crossing zero.
- Mixed stress service delta `-0.009` is small; reviewer reproduced `p=0.177`.
- Action `24` in mixed stress and action `32` in route disruption are real normalized concentration shifts, not raw-count mistakes.
- No blind `3M` recommendation is present.
- Hard blockers are zero, protected hashes match, and no missed production-risk bug evidence was found.

## Final Classification

`HIERARCHICAL_V1_RESIDUAL_WATCHES_NEED_EQUAL_BUDGET_EVAL`
