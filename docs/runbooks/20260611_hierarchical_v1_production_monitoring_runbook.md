# Hierarchical V1 Production Monitoring Runbook

Date: 2026-06-11

Final classification: `HIERARCHICAL_V1_PRODUCTION_MONITORING_RUNBOOK_READY`

## Scope

This runbook covers operational monitoring and real-world validation follow-up
for the currently promoted hierarchical v1 1M production model.

Hard boundaries:

- Do not train from this runbook.
- Do not run offline eval from this runbook.
- Do not mutate registry, production, baselines, DB, checkpoints, or existing
  eval outputs without separate explicit approval.
- Do not start 3M, 5M, 10M, or 100M from monitoring evidence alone.

## Current Production Model

- Logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Runtime family: `torch_joint`
- DQN architecture: `hierarchical_v1`
- Hierarchical initialization: `flat_teacher_distillation_v1`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation dimension: `73`
- Continuous action dimension: `5`
- External discrete action count: `48`
- Training step: `1000000`
- Exact resume capable: `true`

Production directory:

`models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Production artifacts:

| Artifact | SHA256 |
| --- | --- |
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` |
| `production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` |

Active registry ids:

| Role | Status | Registry id |
| --- | --- | --- |
| PPO `continuous_control` | active | `17ba1d28-0054-4f7c-ae9a-34cd305ebb89` |
| DQN `tactical_dispatch` | active | `f87e10d6-479f-44fc-99d1-6925bc9cb346` |
| PPO `continuous_control` | candidate | `3befa11c-e853-4d0f-a951-c2fbbb2a9898` |
| DQN `tactical_dispatch` | candidate | `71af6701-ac3a-40f2-bdd9-cc80868d5a1c` |

## Accepted Residual Watches

The following watches remain active but accepted:

- `route_disruption_congestion`: action `32` concentration.
- `mixed_stress`: action `24` concentration.
- top-action concentration warnings across several scenarios.
- mixed-success route-failure warnings.

Action decode:

- action `24` = `dispatch + shortest + secondary_fleet + none`
- action `32` = `dispatch + low_congestion + secondary_fleet + none`

Equal-budget monitoring baselines:

| Watch | Old production | Hierarchical v1 1M | Interpretation |
| --- | ---: | ---: | --- |
| `route_disruption_congestion` service | `0.904` | `0.901` | slight `-0.0037` delta, inside gate tolerance |
| `route_disruption_congestion` action `32` | `39` attempts, `0.006771` per step | `2737` attempts, `0.475174` per step | real concentration shift |
| `mixed_stress` service | `0.940` | `0.942` | equal-budget regression cleared |
| `mixed_stress` action `24` | `0` attempts, `0.000000` per step | `3342` attempts, `0.580208` per step | real concentration shift |

## Why The Watches Are Accepted

The residual watches are accepted for monitoring because the fair comparator run
did not show a production-risk regression:

- Equal-budget old production and hierarchical production evals used the same
  scenario dir, `20` episodes per scenario, seed `42`, CPU, and deterministic
  inference.
- Both evals had `8/8` scenario verdicts PASS and `160` episode rows.
- Equal-budget long-run gate decision was `PASS` with exit code `0`.
- Hard blockers were zero.
- There was no no-current, no-unassigned, or failed-noop explosion on action
  `24` or action `32`.
- Route-disruption no-work and premium no-work remained materially improved
  versus old production.
- Amazon Last Mile small-sample analysis supports partial route-side
  plausibility: actual driver sequences were generally efficient versus a
  greedy travel-time proxy, and directed travel-time asymmetry was nontrivial.

The watches are not closed. They remain operational calibration questions,
especially for secondary-fleet and no-reorder economics.

## Monitoring Metrics

Track these metrics at scenario, time bucket, customer segment, and action level
where the production system can provide the dimensions:

| Metric | Purpose |
| --- | --- |
| service by scenario or operating regime | Detect SLA/service drift. |
| lateness mean and tail | Detect hidden degradation masked by service average. |
| dispatch success | Confirm dispatch attempts remain executable. |
| no-current by action | Catch empty-work dispatch attempts. |
| no-unassigned by action | Catch dispatch attempts with no eligible work. |
| failed-noop by action | Catch invalid no-op behavior. |
| route failure by action | Catch brittle route selection. |
| no-vehicle by action | Catch fleet infeasibility or capacity mismatch. |
| already-assigned by action | Catch duplicate assignment or lock contention. |
| top action concentration | Detect over-concentrated policy behavior. |
| action `24` rate | Monitor shortest + secondary + no-reorder preference. |
| action `32` rate | Monitor low-congestion + secondary + no-reorder preference. |
| secondary_fleet rate | Validate overflow fleet usage against cost and availability. |
| reorder none rate | Validate no-reorder preference against stockout/backlog outcomes. |

## Warning Thresholds

Use these as operational warnings, not automatic training triggers.

Immediate warning:

- Any no-current, no-unassigned, or failed-noop count appears on action `24` or
  action `32` in production telemetry.
- Service falls below the configured threshold for the corresponding scenario
  or operating regime.
- A runtime check returns non-`torch_joint`, invalid discrete action outside
  `0..47`, non-finite continuous values, or SB3 fallback calls.

Material drift warning:

- Action `24` or action `32` concentration rises by at least `10` percentage
  points above the equal-budget baseline in the comparable regime.
- Action `24` in mixed-stress-like traffic materially exceeds the `0.580208`
  per-step equal-budget baseline and is paired with service, lateness,
  route-failure, or no-vehicle degradation.
- Action `32` in route-disruption-like traffic materially exceeds the
  `0.475174` per-step equal-budget baseline and is paired with service,
  lateness, route-failure, or already-assigned degradation.
- Route failures by action double from the equal-budget watch baseline or grow
  together with service/lateness regression.
- No-vehicle on action `24` materially exceeds the equal-budget baseline of
  `121` mixed-stress no-vehicle rows, or action `32` develops no-vehicle rows
  where the equal-budget baseline was zero.
- Secondary-fleet usage conflicts with real availability, acceptance, cost, or
  cancellation data.
- Reorder-none usage conflicts with stockout, backlog, or emergency-order
  evidence.

Escalation warning:

- The same watch crosses a material drift threshold in two consecutive
  monitoring windows.
- A watch crosses a material drift threshold and a business KPI degrades in the
  same window.
- Company data contradicts the simulator economics for secondary fleet or
  reorder-none decisions.

## Real-World Data Contract

Company data needed for full validation:

| Field group | Required fields |
| --- | --- |
| Order identity | `order_id`, `customer_id`, `customer_location_id`, `created_at` |
| Location | `latitude`, `longitude`, `zone_id`, depot/station id |
| Promise/SLA | `promised_window_start`, `promised_window_end`, SLA tier |
| Dispatch lifecycle | decision timestamp, dispatch attempt timestamp, acceptance timestamp, pickup timestamp, delivery timestamp |
| Outcome | delivery status, delivered on time, lateness, cancellation flag |
| Failure taxonomy | dispatch failure reason, failed attempt reason, no-vehicle reason, route failure reason, duplicate/lock conflict reason |
| Fleet | `vehicle_id`, `carrier_id`, `fleet_type`, vehicle capacity, availability at decision time, acceptance/cancellation rates |
| Route | planned route id, planned route type, actual route id, planned distance/time, actual distance/time, congestion or disruption flag, reroute flag |
| Inventory | SKU/site inventory at decision time, stockout flag, backlog, reorder event, replenishment lead time |
| Cost | primary/secondary fleet cost, route cost, lateness penalty, stockout penalty, holding cost, reorder cost |

Minimum join keys:

- `order_id` for order, dispatch, delivery, and inventory joins.
- `vehicle_id` or `carrier_id` for fleet availability and cost joins.
- route id or route segment id for planned/actual route comparison.
- timestamp fields with a shared timezone and clear event semantics.

## Public-Data Validation Status

Amazon Last Mile small-sample analysis is ready and supports route-side method
validation only.

What the small sample supports:

- It contains `13` routes and `3,129` packages.
- Route, package, travel-time matrix, actual sequence, and invalid-sequence
  score files are present for all `13` routes.
- Actual driver sequences can be compared with greedy travel-time proxies.
- Actual/greedy travel-time ratio mean was `0.967`, with `9/13` routes no worse
  than the greedy proxy.
- Directed travel-time asymmetry was nontrivial, supporting a
  reliability/congestion-style proxy for action `32`.
- The sample supports partial route-side plausibility for shortest-like action
  `24` and low-congestion-like action `32`.

What remains unvalidated without company data:

- Whether secondary fleet should be used at the observed rate.
- Whether secondary fleet is cost-effective or capacity-realistic.
- Whether reorder `none` is correct under mixed stress.
- Whether real dispatch failures, carrier acceptance, and cancellation rates
  match the simulator.
- Whether route/action concentration is statistically typical under the
  company's own stress regimes.

## Operating Checklist

Daily or per monitoring window:

1. Confirm active registry still points to hierarchical v1 1M PPO/DQN artifacts.
2. Confirm production checkpoint path remains
   `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`.
3. Check service, lateness, dispatch success, and hard-failure counters by
   operating regime.
4. Check action `24` and action `32` rates against the equal-budget baselines.
5. Check no-current, no-unassigned, failed-noop, route-failure, no-vehicle, and
   already-assigned counters by action.
6. Check secondary-fleet rate and reorder-none rate against real fleet,
   inventory, and cost facts.
7. Record any warning threshold crossing with the exact window, denominator,
   affected route/fleet/reorder mix, and business KPI impact.

Weekly or after a warning:

1. Reconcile route-choice behavior with available route telemetry.
2. Reconcile secondary-fleet usage with carrier cost, acceptance, and capacity.
3. Reconcile reorder-none usage with stockout, backlog, and emergency-order
   outcomes.
4. Decide whether the finding is telemetry noise, accepted policy behavior, a
   runtime/registry issue, or a research trigger.

## Rollback Procedure

Rollback requires separate explicit approval unless an operational incident
policy already grants emergency authority.

1. Restore active registry to the prior production active pair:
   - `ppo:continuous_control -> models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\ppo_torch_joint_final_balanced_retention_ft_200k.pt`
   - `dqn:tactical_dispatch -> models\checkpoints\joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608\dqn_torch_joint_final_balanced_retention_ft_200k.pt`
2. Preserve candidate and active rows in `models/registry/models.jsonl` for
   audit unless a separate cleanup plan is approved.
3. Quarantine
   `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611` only
   with explicit approval.
4. Do not mutate baselines.
5. Do not mutate DB rows.
6. Run a read-only registry, runtime, production, baseline, DB, and process
   audit after rollback.

## When To Open New Research

Open new research only when monitoring or real-world validation creates a
specific, reproducible question:

- monitored service/lateness regression crosses the warning threshold;
- action `24` or `32` concentration rises materially and coincides with
  operational degradation;
- no-current, no-unassigned, failed-noop, route-failure, or no-vehicle counters
  reveal a concrete action-quality issue;
- company data shows secondary-fleet economics or reorder-none behavior is
  miscalibrated;
- public or company route data contradicts the route-choice proxy assumptions.

Do not start blind 3M training. If more training becomes justified, start with a
bounded gated `1.5M` or `2M` extension plan with explicit residual-watch gates,
fresh output dirs, exact resume, and protected no-mutation checks.

## Non-Goals

- No baseline update.
- No DB cleanup.
- No registry mutation.
- No production overwrite.
- No source checkpoint mutation.
- No offline eval from this runbook.
- No 3M, 5M, 10M, or 100M run from this runbook.

## Source Documents

- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/runs/20260611_hierarchical_v1_residual_watch_statistical_assurance.md`
- `docs/runs/20260611_hierarchical_v1_production_assurance_audit.md`
- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_public_route_proxy_validation_readiness.md`
- `docs/runs/20260611_amazon_last_mile_small_sample_analysis.md`

## Final Classification

`HIERARCHICAL_V1_PRODUCTION_MONITORING_RUNBOOK_READY`
