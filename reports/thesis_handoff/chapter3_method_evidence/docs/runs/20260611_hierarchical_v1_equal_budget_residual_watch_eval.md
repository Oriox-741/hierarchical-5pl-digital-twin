# Hierarchical V1 Equal-Budget Residual-Watch Eval

Date: 2026-06-11

Pre-review classification: `HIERARCHICAL_V1_EQUAL_BUDGET_RESIDUAL_WATCHES_ACCEPTABLE`

## Scope

This run executed the explicitly approved equal-budget residual-watch assurance eval for:

- old production comparator: `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`
- current hierarchical production: `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`

No training, registry update, production mutation, baseline mutation, DB mutation, checkpoint mutation, old eval-output edit, or 3M/5M/10M/100M run was performed.

Approved eval-output mutation was limited to these fresh dirs:

- `models/eval/equal_budget_oldprod_20ep_seed42_20260611`
- `models/eval/equal_budget_hierarchical_v1_20ep_seed42_20260611`

## Commands Run

Old production equal-budget eval:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:TORCH_NUM_THREADS='2'; python -m src.eval.evaluate_real_world_scenarios --checkpoint models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt --scenario-dir configs\eval_scenarios --output-dir models\eval\equal_budget_oldprod_20ep_seed42_20260611 --episodes 20 --seed 42 --device cpu --deterministic
```

Hierarchical production equal-budget eval:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:TORCH_NUM_THREADS='2'; python -m src.eval.evaluate_real_world_scenarios --checkpoint models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt --scenario-dir configs\eval_scenarios --output-dir models\eval\equal_budget_hierarchical_v1_20ep_seed42_20260611 --episodes 20 --seed 42 --device cpu --deterministic
```

Equal-budget long-run gate:

```powershell
python -m src.eval.check_long_run_gate --candidate-summary models\eval\equal_budget_hierarchical_v1_20ep_seed42_20260611\scenario_summary.json --production-summary models\eval\equal_budget_oldprod_20ep_seed42_20260611\scenario_summary.json --candidate-label hierarchical_v1_equal_budget_20ep_seed42
```

Gate result was saved to:

`docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_gate_result.json`

Gate exit code: `0`

Gate decision: `PASS`

Fatal failures: `[]`

## Eval Validation

| Eval | Scenario dir | Episodes | Seed | Device | Deterministic | Scenarios | Episode rows | Scenario verdicts | Hard blockers |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| old production | `configs/eval_scenarios` | 20 | 42 | cpu | yes | 8 | 160 | 8/8 PASS | zero |
| hierarchical v1 | `configs/eval_scenarios` | 20 | 42 | cpu | yes | 8 | 160 | 8/8 PASS | zero |

Required files are present in both output dirs:

- `scenario_summary.json`
- `scenario_summary.csv`
- `episode_metrics.jsonl`
- `real_world_evaluation_report.md`

Artifact hashes:

| Artifact | SHA256 |
| --- | --- |
| old production `scenario_summary.json` | `0D832E0658B134EEB5868749F76267C52A617B21F5B5A3896369EBD1ECFCF3C3` |
| old production `episode_metrics.jsonl` | `B0F1FEBB004D91935E9E0039B86287B6D4C0922E34BB0E19A20F2F5BC8D95E1D` |
| hierarchical `scenario_summary.json` | `6B59B008B2CD6ED2BFC486DB04C523E899261040C99937C8E4B05E47C984F424` |
| hierarchical `episode_metrics.jsonl` | `069892DF484DCE93B86A7CF28F1B74A1CF016C8B3CD1175521300E6AD0FBE266` |
| equal-budget gate JSON | `C9273E12607B884197BA8E1CB7C68646388D97E11FCE87B0CA0DC764C7A0CBAB` |

## Scenario Comparison

All values are equal-budget: 20 episodes per scenario, same scenario dir, same seed policy, CPU deterministic.

| Scenario | service old->H | late old->H | dispatch old->H | success old->H | no-work/step old->H | delivered/step old->H | routefail/step old->H | no_vehicle/step old->H | already/step old->H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_normal` | 0.945->0.990 (+0.045) | 0.000->0.000 | 0.459->0.372 | 0.998->1.000 | 0.001042->0.000000 | 0.522->0.505 | 0.001215->0.000521 | 0.000521->0.000174 | 0.245->0.193 |
| `demand_spike_volatility` | 0.840->0.859 (+0.019) | 0.163->0.027 | 0.756->0.793 | 1.000->1.000 | 0.000000->0.000000 | 1.345->1.547 | 0.000000->0.002257 | 0.101736->0.144618 | 0.545->0.541 |
| `high_holding_cost` | 0.931->0.966 (+0.035) | 0.000->0.000 | 0.446->0.449 | 0.997->0.999 | 0.001910->0.000347 | 0.515->0.507 | 0.001389->0.001389 | 0.000868->0.000000 | 0.201->0.240 |
| `lead_time_volatility` | 0.914->1.000 (+0.086) | 0.012->0.000 | 0.460->0.360 | 0.998->0.999 | 0.001389->0.000174 | 0.521->0.542 | 0.000521->0.000347 | 0.000174->0.000347 | 0.208->0.080 |
| `mixed_stress` | 0.940->0.942 (+0.003) | 0.002->0.000 | 0.943->0.987 | 0.997->0.999 | 0.006250->0.001389 | 1.441->1.457 | 0.001042->0.001736 | 0.050868->0.050868 | 0.588->0.659 |
| `premium_sla_pressure` | 0.982->1.000 (+0.018) | 0.004->0.000 | 0.439->0.357 | 0.759->0.999 | 0.218750->0.000174 | 0.516->0.538 | 0.000521->0.000347 | 0.000694->0.000347 | 0.131->0.078 |
| `route_disruption_congestion` | 0.904->0.901 (-0.004) | 0.000->0.008 | 0.551->0.499 | 0.862->0.999 | 0.150694->0.000347 | 0.533->0.508 | 0.000521->0.001389 | 0.000000->0.000000 | 0.238->0.260 |
| `vehicle_scarcity_capacity_shock` | 0.908->0.949 (+0.041) | 0.004->0.001 | 0.748->0.806 | 0.998->1.000 | 0.001563->0.000000 | 0.891->0.868 | 0.001389->0.001389 | 0.000868->0.000347 | 0.389->0.447 |

Equal-budget result:

- Hierarchical improves service in seven of eight scenarios.
- `mixed_stress` is no longer below old production; it improves by `+0.0028`.
- `route_disruption_congestion` remains slightly below old production by `-0.0037`, which is inside the long-run gate's service regression tolerance.
- Premium and route no-work remain dramatically improved.
- Equal-budget long-run gate passes with no fatal failures.

## Gate Warnings

Equal-budget gate warnings were nonfatal:

- top-action concentration increased in `baseline_normal`, `high_holding_cost`, `lead_time_volatility`, `mixed_stress`, `premium_sla_pressure`, and `route_disruption_congestion`
- `route_disruption_congestion` lateness above old production: `0.008` vs `0.000`
- mixed-success route-failure steps increased in `demand_spike_volatility`, `high_holding_cost`, `mixed_stress`, `route_disruption_congestion`, and `vehicle_scarcity_capacity_shock`

These remain production watches, not blockers.

## Route Disruption Deep Dive

`route_disruption_congestion` service:

- old production mean/sd/min/median/max: `0.904 / 0.049 / 0.822 / 0.902 / 0.994`
- hierarchical mean/sd/min/median/max: `0.901 / 0.016 / 0.876 / 0.900 / 0.942`
- paired delta H-old: `-0.003693`
- exact paired sign-flip p-value: `0.760`
- paired bootstrap 95% interval: `[-0.0271, 0.0183]`

Operational counters:

- lateness: `0.000000 -> 0.008256`
- dispatch rate: `0.550868 -> 0.498611`
- dispatch success: `0.861557 -> 0.999320`
- no-work per-step: `0.150694 -> 0.000347`
- routefail per-step: `0.000521 -> 0.001389`
- no_vehicle per-step: `0.000000 -> 0.000000`
- already-assigned per-step: `0.237847 -> 0.260069`

Action `32`:

- old production: `39` attempts, `0.006771` per step
- hierarchical: `2737` attempts, `0.475174` per step
- no-current action `32`: `1 -> 0`
- no-unassigned action `32`: `1 -> 0`
- failed-noop action `32`: `1 -> 0`
- route-failure action `32`: `0 -> 5`
- no-vehicle action `32`: `0 -> 0`
- already-assigned action `32`: `30 -> 1444`

Worst service episodes:

- old production: `0.822`, `0.850`, `0.852`
- hierarchical: `0.876`, `0.878`, `0.883`

Interpretation:

Route action `32` concentration is real and should remain monitored. The fair service delta is not statistically meaningful and is inside gate tolerance. Hierarchical also improves the old production route no-work and dispatch-success failures by a large margin. The only material residual watch is small route lateness plus policy concentration, not a production-risk service regression.

## Mixed Stress Deep Dive

`mixed_stress` service:

- old production mean/sd/min/median/max: `0.940 / 0.014 / 0.919 / 0.937 / 0.966`
- hierarchical mean/sd/min/median/max: `0.942 / 0.011 / 0.925 / 0.944 / 0.966`
- paired delta H-old: `+0.002758`
- exact paired sign-flip p-value: `0.300`
- paired bootstrap 95% interval: `[-0.0022, 0.0077]`

Operational counters:

- lateness: `0.001721 -> 0.000178`
- dispatch rate: `0.943403 -> 0.986632`
- dispatch success: `0.996681 -> 0.999293`
- no-work per-step: `0.006250 -> 0.001389`
- routefail per-step: `0.001042 -> 0.001736`
- no_vehicle per-step: `0.050868 -> 0.050868`
- already-assigned per-step: `0.588368 -> 0.659201`

Action `24`:

- old production: `0` attempts, `0.000000` per step
- hierarchical: `3342` attempts, `0.580208` per step
- no-current action `24`: `0 -> 0`
- no-unassigned action `24`: `0 -> 0`
- failed-noop action `24`: `0 -> 0`
- route-failure action `24`: `0 -> 10`
- no-vehicle action `24`: `0 -> 121`
- already-assigned action `24`: `0 -> 2010`

Worst service episodes:

- old production: `0.919`, `0.923`, `0.923`
- hierarchical: `0.925`, `0.929`, `0.930`

Interpretation:

The equal-budget run clears the prior mixed-stress service watch. Hierarchical is slightly better on service, better on lateness, better on no-work, and better on dispatch success. Action `24` concentration is real, but it is not paired with no-current/no-unassigned/failed-noop evidence. It remains a monitoring watch, not a gated-extension trigger.

## Decision

`HIERARCHICAL_V1_EQUAL_BUDGET_RESIDUAL_WATCHES_ACCEPTABLE`

Rationale:

- Equal-budget gate passed with exit code `0`.
- Both evals have `8/8 PASS`, `160` rows, and zero hard blockers.
- `mixed_stress` no longer regresses against old production under equal budget.
- `route_disruption_congestion` service delta is only `-0.0037`, inside gate tolerance, and statistically indistinguishable from zero in the paired test.
- Old production's route/premium no-work and dispatch-success weaknesses remain materially improved by hierarchical v1.
- Action concentration is real but not accompanied by hard-blocker or no-work explosion evidence.
- No hidden runtime/checkpoint/reward bug is suspected.

Do not start 3M now. No gated 1.5M/2M extension is recommended from this evidence. The safe next step is production monitoring of the remaining nonfatal watches: route lateness, route action `32`, mixed action `24`, top-action concentration, and mixed-success route-failure steps.

## Protected No-Mutation Proof

Protected hashes after eval/gate:

- `models/registry/active_models.json`: `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
- `models/registry/models.jsonl`: `935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`
- old production comparator joint: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`
- hierarchical production joint: `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE`

Protected path profiles:

| Path | File count | Total bytes | Status |
| --- | ---: | ---: | --- |
| `models/baselines` | `2692` | `7392576274` | unchanged |
| `db` | `7` | `28330` | unchanged |
| `models/checkpoints` | `787` | `4920830666` | unchanged |
| `models/eval` | `128` | `388267659` | changed only by the two approved fresh equal-budget eval dirs |

Process scan:

```text
NO_TRAIN_EVAL_GATE_PROCESS
```

## Dashboard Status

`docs/00_PROJECT_DASHBOARD.md` was updated because residual-watch status changed after the approved equal-budget eval.

## Independent Review

Reviewer verdict:

```text
EQUAL_BUDGET_RESIDUAL_WATCHES_ACCEPTABLE
```

Reviewer rationale summary:

- Both fresh eval dirs contain required files, 8 scenarios, 160 episode rows, 20 episodes per scenario, and 8/8 PASS scenario verdicts with hard-blocker telemetry zero.
- Gate result is PASS with `fatal_failures=[]` and recorded exit code `0`.
- Reviewer independently reproduced route disruption service delta `-0.003693`, paired p `0.7597`, and action `32` attempts `39 -> 2737` without no-current/no-unassigned/failed-noop growth.
- Reviewer independently reproduced mixed stress service delta `+0.002758`, paired p `0.2997`, and action `24` attempts `0 -> 3342` without no-current/no-unassigned/failed-noop growth.
- Route lateness and action concentration remain real watches, but not blockers.
- Protected no-mutation proof is coherent.

## Final Classification

`HIERARCHICAL_V1_EQUAL_BUDGET_RESIDUAL_WATCHES_ACCEPTABLE`
