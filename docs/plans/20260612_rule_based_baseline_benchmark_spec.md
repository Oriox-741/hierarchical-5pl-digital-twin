# Rule-Based Baseline Benchmark Spec

Date: 2026-06-13

Scope: design/spec only. This document does not implement or run a benchmark. It does not train, run offline eval, run a long-run gate, install dependencies, download data, mutate registry, mutate production, mutate baselines, mutate DB, mutate checkpoints, edit existing eval outputs, create training configs, ingest private company data, or prepare/send a firm-facing data request.

Final planned classification: `RULE_BASED_BASELINE_SPEC_READY`

## Executive Summary

The first benchmark path after the Step 6 no-run benchmark protocol should be transparent rule-based logistics baselines. These baselines answer a practical thesis/advisor question:

> Does the learned hierarchical v1 policy beat simple dispatch, route, fleet, and reorder rules that an operator can understand?

This spec defines the baselines and comparison metrics only. A later implementation must be separately approved and must not run simulator eval unless the user explicitly approves a fresh benchmark/eval output path.

Current production model to compare later:

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- runtime: `torch_joint`
- architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- observation dimension: `73`
- PPO continuous action dimension: `5`
- DQN external discrete action count: `48`
- status: active registry plus copy-only production promoted
- ladder: `250k PASS`, `500k PASS`, `1M PASS`
- equal-budget residual-watch gate: `PASS`
- ops bundle: `OPS_BUNDLE_CLEAN`
- accepted watches: action `24` and action `32` concentration

## Why Rule-Based Baselines Are First

Rule-based baselines come first because they are:

- advisor-readable;
- implementable without external solver dependencies;
- compatible with the existing `env_5pl` and 48-action contract;
- useful before heavier OR-Tools, public-dataset, multi-seed, or ablation work;
- a direct answer to "Yapay zeka neye gore iyi?"

They should be deliberately simple. A strong rule baseline makes the RL claim more meaningful. A weak rule baseline still provides a necessary sanity check and helps identify which scenario families are genuinely hard.

## What This Benchmark Will Prove Later

If separately implemented and run under approval, this benchmark can show:

- whether hierarchical v1 improves service/lateness/dispatch quality against transparent operator heuristics under the same simulator scenarios;
- whether action `24` and action `32` concentration is better or worse than simple route/fleet/reorder rules;
- which scenario families need stronger baselines or public benchmark follow-up;
- whether the learned policy is using the action space in a way that beats obvious static choices.

## What This Benchmark Will Not Prove

Even if run later, this benchmark will not prove:

- real-world logistics optimality;
- live TMS/WMS/ERP readiness;
- company-specific secondary-fleet economics;
- company-specific reorder economics;
- public route-benchmark SOTA;
- that action `24` or `32` are economically optimal outside the simulator contract.

Company data remains necessary for fleet cost, carrier acceptance, inventory/reorder economics, and dispatch failure taxonomy.

## Existing Action Contract

The rule baselines must emit the existing external DQN action ids. They must not add new actions or change `env_5pl` physics.

External discrete action composition:

`2 dispatch x 3 route x 2 fleet mode x 4 reorder = 48`

Decode/encode order:

1. dispatch: `hold=0`, `dispatch=1`
2. route: `shortest=0`, `low_congestion=1`, `high_resilience=2`
3. mode: `secondary_fleet=0`, `primary_fleet=1`
4. reorder: `none=0`, `conservative=1`, `aggressive=2`, `emergency=3`

Important ids:

| Action id | Decode | Use in this spec |
| ---: | --- | --- |
| `0` | hold + shortest + secondary_fleet + none | default safe hold |
| `4` | hold + shortest + primary_fleet + none | primary-flavored hold, semantically hold |
| `8` | hold + low_congestion + secondary_fleet + none | low-congestion hold, semantically hold |
| `12` | hold + low_congestion + primary_fleet + none | primary low-congestion hold, semantically hold |
| `24` | dispatch + shortest + secondary_fleet + none | watched mixed-stress action |
| `28` | dispatch + shortest + primary_fleet + none | FIFO/EDD primary default |
| `29` | dispatch + shortest + primary_fleet + conservative | conservative reorder dispatch |
| `31` | dispatch + shortest + primary_fleet + emergency | emergency stockout-prevention dispatch |
| `32` | dispatch + low_congestion + secondary_fleet + none | watched route-disruption action |
| `36` | dispatch + low_congestion + primary_fleet + none | low-congestion primary dispatch |
| `40` | dispatch + high_resilience + secondary_fleet + none | high-resilience secondary dispatch |
| `44` | dispatch + high_resilience + primary_fleet + none | high-resilience primary dispatch |

Hold actions must be treated carefully. In `hierarchical_v1`, route/mode labels are masked for hold decisions. A rule implementation should prefer one canonical hold action, likely `0`, unless a test explicitly checks encode/decode coverage.

## Baseline 1: FIFO + Shortest + Primary + None

Operational logic:

- If at least one dispatchable order and one primary vehicle are available, dispatch the oldest eligible order.
- Use shortest route.
- Use primary fleet.
- Do not request DQN reorder override.
- If there is no dispatchable work or no vehicle, hold.

Required state signals:

- pending order creation/arrival time;
- assigned/delivered status;
- primary vehicle availability and capacity;
- route network shortest path availability;
- inventory/stockout signal only for reporting, not action selection.

Dispatch rule:

- Choose oldest eligible unassigned order by order timestamp.

Route rule:

- Always `shortest`.

Fleet rule:

- Prefer `primary_fleet`; do not fallback unless this baseline is extended.

Reorder rule:

- Always `none`.

Likely action ids:

- dispatch: `28`
- hold: `0`

Expected strengths:

- strong under `baseline_normal`;
- simple and stable;
- low action concentration ambiguity;
- easy advisor explanation.

Expected weaknesses:

- may fail under vehicle scarcity if primary fleet is unavailable;
- may fail route disruption because shortest can become unreliable;
- may ignore urgent/premium orders;
- may under-react to stockout pressure.

Scenarios where performs well:

- `baseline_normal`
- possibly `high_holding_cost` if no reorder is safe

Scenarios where may fail:

- `premium_sla_pressure`
- `route_disruption_congestion`
- `vehicle_scarcity_capacity_shock`
- `mixed_stress`

## Baseline 2: Earliest Due Date + Shortest + Primary + None

Operational logic:

- If dispatchable work and primary vehicle are available, dispatch the order with earliest due/promised deadline.
- Use shortest route and primary fleet.
- Do not request reorder override.
- Hold when no useful dispatch is possible.

Required state signals:

- promised due time or lateness risk;
- pending unassigned orders;
- primary vehicle availability and capacity;
- shortest route availability.

Dispatch rule:

- Choose eligible unassigned order with minimum due time or highest lateness risk.

Route rule:

- Always `shortest`.

Fleet rule:

- Prefer `primary_fleet`.

Reorder rule:

- Always `none`.

Likely action ids:

- dispatch: `28`
- hold: `0`

Expected strengths:

- better SLA/lateness behavior than FIFO;
- understandable deadline-first benchmark;
- useful in premium and lead-time pressure comparisons.

Expected weaknesses:

- can starve older non-urgent orders;
- still vulnerable to route disruption;
- still ignores secondary fleet when primary is constrained;
- no proactive inventory response.

Scenarios where performs well:

- `premium_sla_pressure`
- `lead_time_volatility`
- `baseline_normal`

Scenarios where may fail:

- `route_disruption_congestion`
- `vehicle_scarcity_capacity_shock`
- `mixed_stress`

## Baseline 3: Premium-First + Shortest + Primary/Secondary + None

Operational logic:

- Prioritize premium/urgent orders first.
- Use shortest route.
- Prefer primary fleet for premium work if available; fallback to secondary fleet if primary cannot dispatch and secondary can.
- Do not request reorder override.
- Hold if no useful dispatch is possible.

Required state signals:

- premium/urgent order flag;
- due time/lateness risk;
- primary and secondary vehicle availability/capacity;
- pending unassigned orders;
- shortest route availability.

Dispatch rule:

- Premium/urgent eligible orders first, then earliest due, then FIFO tie-break.

Route rule:

- Always `shortest`.

Fleet rule:

- `primary_fleet` when feasible.
- `secondary_fleet` fallback only when primary is unavailable and useful dispatch exists.

Reorder rule:

- Always `none`.

Likely action ids:

- primary dispatch: `28`
- secondary dispatch: `24`
- hold: `0`

Expected strengths:

- strong SLA intuition;
- direct comparator for premium behavior;
- tests whether action `24` secondary-fleet use is merely a fallback rule or a learned broader strategy.

Expected weaknesses:

- secondary-fleet economics remain unvalidated without company data;
- may overuse shortest route under congestion;
- no inventory response.

Scenarios where performs well:

- `premium_sla_pressure`
- `mixed_stress` if secondary fleet is a useful capacity fallback

Scenarios where may fail:

- `high_holding_cost`
- `route_disruption_congestion`
- real-world cost comparison without company data

## Baseline 4: Low-Congestion Under Disruption

Operational logic:

- Use normal FIFO or earliest-due dispatch selection.
- Switch route from shortest to low-congestion when route disruption, congestion, or delay multiplier pressure is high.
- Prefer primary fleet by default; optionally use secondary if primary unavailable.
- Do not request reorder override.

Required state signals:

- route disruption probability/pressure;
- congestion or arc delay proxy;
- pending unassigned orders;
- available primary and secondary vehicles;
- route candidate scores if exposed by the future adapter.

Dispatch rule:

- Earliest due among eligible orders when useful dispatch exists.

Route rule:

- `low_congestion` if route-disruption pressure exceeds threshold.
- Otherwise `shortest`.

Fleet rule:

- Primary default; secondary fallback only if primary unavailable and the baseline variant allows fallback.

Reorder rule:

- Always `none`.

Likely action ids:

- low-congestion secondary dispatch: `32`
- low-congestion primary dispatch: `36`
- shortest primary dispatch: `28`
- hold: `0`

Expected strengths:

- direct transparent comparator for action `32`;
- should perform better than shortest-only under `route_disruption_congestion`;
- tests whether learned route concentration is a sensible route-stress response.

Expected weaknesses:

- threshold choice can be arbitrary;
- low-congestion may increase distance or cost;
- does not validate secondary-fleet or no-reorder economics.

Scenarios where performs well:

- `route_disruption_congestion`
- `mixed_stress`

Scenarios where may fail:

- `baseline_normal` if it over-switches;
- `high_holding_cost` if route choice ignores inventory pressure.

## Baseline 5: Conservative Stock-Threshold Reorder

Operational logic:

- Dispatch normally using FIFO or earliest due.
- If inventory coverage or safety-stock gap crosses a conservative threshold, use conservative reorder action.
- Otherwise no reorder.
- Route and fleet remain simple defaults.

Required state signals:

- inventory coverage;
- safety stock target gap;
- stockout risk;
- pending orders;
- primary vehicle availability.

Dispatch rule:

- FIFO or earliest due among eligible orders.

Route rule:

- `shortest`.

Fleet rule:

- `primary_fleet`.

Reorder rule:

- `conservative` when inventory pressure is moderate and holding-cost pressure is not prohibitive.
- `none` otherwise.

Likely action ids:

- dispatch + shortest + primary + conservative: `29`
- dispatch + shortest + primary + none: `28`
- hold + none: `0`

Expected strengths:

- understandable inventory sanity baseline;
- useful for lead-time and stockout-pressure comparison;
- avoids emergency overuse.

Expected weaknesses:

- current `env_5pl` treats DQN conservative override similarly to no override in `_apply_hub_discrete_action`, so future implementation must document exactly what conservative means in action telemetry versus physical replenishment;
- may be too weak under severe stockout;
- may over-order under high holding cost if threshold is poorly chosen.

Scenarios where performs well:

- `lead_time_volatility`
- `baseline_normal`

Scenarios where may fail:

- `high_holding_cost`
- `mixed_stress`

## Baseline 6: Emergency Stockout-Prevention Reorder

Operational logic:

- If stockout risk or inventory shortfall is severe, issue emergency reorder.
- Dispatch only when useful work exists; otherwise hold with emergency reorder variant only if a future implementation explicitly supports non-dispatch reorder-only actions.
- Route/fleet defaults remain simple.

Required state signals:

- stockout risk;
- inventory shortfall;
- safety stock gap;
- lead-time delay pressure;
- pending orders;
- vehicle availability.

Dispatch rule:

- Earliest due if dispatchable; otherwise hold.

Route rule:

- `shortest`.

Fleet rule:

- `primary_fleet`.

Reorder rule:

- `emergency` when severe stockout/shortfall threshold is crossed.
- `none` otherwise.

Likely action ids:

- dispatch + shortest + primary + emergency: `31`
- dispatch + shortest + primary + none: `28`
- hold: `0` by default; any reorder-only hold variant must be justified and tested because hold route/mode labels are semantically neutral.

Expected strengths:

- protects service under inventory shortage;
- clear comparator for stockout-pressure behavior.

Expected weaknesses:

- emergency reorder is costly;
- can be bad in high holding-cost scenarios;
- may create unrealistic claims without company cost data.

Scenarios where performs well:

- `lead_time_volatility`
- demand surge regimes with stockout risk

Scenarios where may fail:

- `high_holding_cost`
- `baseline_normal`

## Baseline 7: Vehicle-Scarcity Primary-First/Secondary-Fallback

Operational logic:

- Prefer primary fleet when primary capacity exists.
- If primary fleet is unavailable but secondary fleet can dispatch useful work, fallback to secondary.
- Use earliest due or FIFO dispatch selection.
- Use shortest route unless route disruption pressure is high in an optional combined variant.
- Do not request reorder override.

Required state signals:

- primary vehicle availability and remaining capacity;
- secondary vehicle availability and remaining capacity;
- pending unassigned orders;
- order capacity requirement;
- route feasibility.

Dispatch rule:

- Earliest due or FIFO among eligible orders.

Route rule:

- `shortest` in the base variant.
- Optional low-congestion combined variant must be named separately.

Fleet rule:

- `primary_fleet` if feasible.
- `secondary_fleet` only as fallback.

Reorder rule:

- `none`.

Likely action ids:

- primary dispatch: `28`
- secondary fallback dispatch: `24`
- hold: `0`

Expected strengths:

- clear comparator for secondary-fleet usage;
- useful under `vehicle_scarcity_capacity_shock`;
- tests whether action `24` is an intelligent fallback or over-concentrated.

Expected weaknesses:

- secondary-fleet cost/reliability cannot be validated without company data;
- may miss low-congestion route adaptation;
- may over-dispatch if vehicle availability is stale.

Scenarios where performs well:

- `vehicle_scarcity_capacity_shock`
- `mixed_stress`

Scenarios where may fail:

- `route_disruption_congestion`
- cost-sensitive real-world settings

## Baseline 8: High-Holding No-Overstock Rule

Operational logic:

- Avoid reorder when holding cost pressure is high and stockout risk is controlled.
- Dispatch useful work with FIFO or earliest due.
- Use shortest route and primary fleet.
- Hold if dispatch would be infeasible or no useful work exists.

Required state signals:

- holding-cost pressure;
- inventory coverage;
- stockout risk;
- safety stock gap;
- pending unassigned orders;
- primary vehicle availability.

Dispatch rule:

- FIFO or earliest due among eligible orders.

Route rule:

- `shortest`.

Fleet rule:

- `primary_fleet`.

Reorder rule:

- force `none` when holding-cost pressure is high and stockout risk is below threshold;
- allow conservative/emergency only in separately named stockout override variant.

Likely action ids:

- dispatch + shortest + primary + none: `28`
- hold: `0`

Expected strengths:

- direct sanity baseline for `high_holding_cost`;
- prevents trivial overstock behavior;
- advisor-readable economic intuition.

Expected weaknesses:

- may under-protect service if stockout pressure is misread;
- cannot validate true holding-cost accounting without company finance data;
- does not solve route disruption or vehicle scarcity.

Scenarios where performs well:

- `high_holding_cost`
- `baseline_normal`

Scenarios where may fail:

- `lead_time_volatility`
- `mixed_stress`

## Comparison Metrics

Future benchmark reports should use the existing monitoring/eval vocabulary:

- service level;
- lateness and true lateness pressure;
- dispatch rate;
- dispatch success per attempt;
- no-current dispatch steps;
- no-unassigned dispatch steps;
- failed-noop dispatch steps;
- route failure per step;
- no-vehicle per step;
- already-assigned per step;
- delivered per step;
- action concentration and top action share;
- action `24` rate;
- action `32` rate;
- secondary-fleet rate;
- reorder-none rate;
- conservative/aggressive/emergency reorder rate;
- scenario verdict;
- hard blocker status.

Every metric must include denominator semantics. Raw counts must not be compared across unequal episode budgets without normalization.

## Scenario Expectations

| Scenario | Baselines expected to be strongest | Baselines expected to be weaker |
| --- | --- | --- |
| `baseline_normal` | FIFO shortest primary; EDD shortest primary | low-congestion if over-triggered; emergency reorder |
| `demand_spike_volatility` | EDD; premium-first fallback; conservative reorder if stock pressure exists | no-overstock if stockout grows |
| `high_holding_cost` | no-overstock; FIFO shortest primary none | emergency reorder; aggressive reorder |
| `lead_time_volatility` | conservative stock-threshold; emergency stockout prevention when severe | no-overstock if shortfall grows |
| `mixed_stress` | premium-first fallback; vehicle-scarcity fallback; low-congestion variant | single static FIFO |
| `premium_sla_pressure` | EDD; premium-first | FIFO-only |
| `route_disruption_congestion` | low-congestion under disruption | shortest-only |
| `vehicle_scarcity_capacity_shock` | primary-first/secondary-fallback | primary-only FIFO |

## Expected Output Artifacts If Implemented Later

Implementation-only skeleton, if approved later:

- `src/eval/rule_based_baselines.py`
- `tests/eval/test_rule_based_baselines.py`
- `docs/runs/20260613_rule_based_baseline_skeleton_report.md`

Benchmark/eval run, only if separately approved later:

- fresh baseline output directory outside existing eval artifacts or under an explicitly approved fresh benchmark path;
- JSON report with one row per baseline/scenario;
- CSV summary for advisor tables;
- markdown report comparing hierarchical v1 against each baseline;
- protected no-mutation proof;
- explicit statement that the run is simulator-only.

## Stop Conditions

Stop before implementation or benchmark if:

- the future rule requires changing `env_5pl` physics;
- the future rule requires changing observation/action contract `73/48`;
- output paths would edit existing eval outputs;
- dependency installation is proposed without approval;
- training, long-run gate, or offline eval would start without approval;
- protected hashes drift before a future run;
- private company data is requested or ingested;
- the rule cannot distinguish "hold because no useful work" from "failed dispatch."

## Approval Requirements Before Implementation Or Run

Separate approval is required for:

- writing implementation code;
- adding tests;
- running any simulator benchmark/eval;
- writing fresh benchmark/eval outputs;
- running a long-run gate;
- installing dependencies;
- downloading datasets;
- ingesting any company data.

This spec alone authorizes no benchmark execution.

## Non-Goals

- No training.
- No offline eval.
- No long-run gate.
- No benchmark run.
- No dependency install.
- No dataset download.
- No registry mutation.
- No production mutation.
- No baseline mutation.
- No DB mutation.
- No checkpoint mutation.
- No existing eval-output edit.
- No training config creation.
- No private company-data ingest.
- No firm-facing data request.
- No claim that rule baselines prove real-world optimum.

