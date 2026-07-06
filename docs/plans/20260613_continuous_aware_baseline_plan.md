# Continuous-Aware Rule Baseline Plan

Date: 2026-06-13

Status: planning package only. This plan does not authorize training, offline eval, long-run gate, benchmark execution, dependency installation, dataset download, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output edits, private company-data ingestion, or SOTA claims.

Recommended classification: `CONTINUOUS_AWARE_BASELINE_PLAN_READY`

## Executive Summary

The current rule-based baseline benchmark is useful, but it is not yet a full PPO+DQN-equivalent comparator. The discrete side is meaningful: each rule emits a legal DQN action id in the external `0..47` contract. The continuous side is fixed: `src/eval/rule_based_baseline_arena.py` uses `neutral_continuous_action()`, which returns a normalized zero vector of length 5 for every rule decision.

Important source-truth finding: normalized zero is not a physical no-op. `ActionProjector` maps `[0, 0, 0, 0, 0]` to midpoint physical controls:

- `reorder_fraction = 0.5`
- `dispatch_intensity = 0.5`
- `speed_multiplier = 1.0`
- `safety_stock_multiplier = 1.5` with the default max multiplier `2.0`
- `capacity_buffer_fraction = 0.25` with the default max buffer `0.50`

In joint mode, `env_5pl` applies continuous controls with `allow_dispatch=False`, so PPO does not directly dispatch pending orders. However, the continuous vector still affects planned replenishment, safety stock, capacity buffer restoration, vehicle speed, and DQN macro-dispatch budget. Therefore the current benchmark should be described as a fixed continuous-control comparator, not a truly continuous-neutral full baseline.

Recommended next implementation: add a continuous action mode seam to the rule-based arena. The first skeleton should support mode selection and tests only. It must not run the benchmark. It should keep current behavior as the default for backwards-compatible interpretation, then add heuristic and PPO-assisted options behind explicit configuration.

## Source Findings

Current rule baseline path:

- `src/eval/rule_based_baselines.py` defines pure deterministic discrete rule policies.
- `src/eval/rule_based_baseline_arena.py` adapts those rules into joint actions.
- `neutral_continuous_action()` returns `np.zeros((CONTINUOUS_ACTION_DIM,), dtype=np.float32)`.
- `build_joint_action_from_rule_decision()` always emits:

```python
{
    "continuous": neutral_continuous_action(),
    "discrete": validate_rule_action_id(decision.action_id),
}
```

Source-truth continuous dimensions and physical fields:

- `CONTINUOUS_ACTION_DIM = 5`
- `PhysicalAction.reorder_fraction`
- `PhysicalAction.dispatch_intensity`
- `PhysicalAction.speed_multiplier`
- `PhysicalAction.safety_stock_multiplier`
- `PhysicalAction.capacity_buffer_fraction`

Projection formulas from `src/act/action_projector.py`:

- `reorder_fraction = (raw[0] + 1) * 0.5`
- `dispatch_intensity = (raw[1] + 1) * 0.5`
- `speed_multiplier = 1.0 + raw[2] * max_speed_delta_fraction`
- `safety_stock_multiplier = 1.0 + ((raw[3] + 1) * 0.5) * (max_safety_stock_multiplier - 1.0)`
- `capacity_buffer_fraction = ((raw[4] + 1) * 0.5) * max_capacity_buffer_fraction`

Joint-mode application in `env_5pl`:

- continuous physical action is applied first with `allow_dispatch=False`;
- `dispatch_intensity` becomes `macro_dispatch_release`;
- macro dispatch budget is `1 + int(release * max_extra_dispatch_budget)`;
- discrete DQN action then dispatches/holds, selects route, fleet mode, and reorder override.

## Continuous Mode Design

### Mode 1: `neutral_continuous`

Definition:

- Existing behavior.
- Fixed normalized vector: `[0, 0, 0, 0, 0]`.
- Should be documented internally as `neutral_normalized_midpoint`, because it is midpoint physical control, not physical no-op.

Fairness:

- Good for backward compatibility with the completed benchmark.
- Deterministic and simple.
- Does not represent a learned PPO policy.
- Does not isolate discrete-only behavior as cleanly as the word "neutral" suggests.

Advisor interpretation:

- "Rule policies were evaluated with a fixed continuous-control vector, so the benchmark primarily compares tactical discrete rules."

### Mode 2: `heuristic_continuous`

Definition:

- Deterministic rules produce the 5-dimensional normalized PPO vector from current state/context signals.
- Uses no future leakage.
- Uses only signals exposed through observation features, environment snapshot helpers, scenario stress state, or rule context fields.

Required input signals:

- Stock pressure: `stockout_risk`, `inventory_coverage`, `safety_stock_gap`, backlog/inventory shortfall if exposed.
- Demand/SLA pressure: dispatchable order count, pending work pressure, urgent ratio, lateness risk, premium SLA pressure.
- Route disruption/congestion: route disruption pressure, congestion pressure, route candidate scores/gaps.
- Vehicle scarcity: primary/secondary availability, feasible dispatch ratio, vehicle availability pressure, capacity shock pressure.
- Lead time / holding pressure: lead-time volatility, supplier delay pressure, holding cost pressure, stockout penalty pressure.

Target physical behavior:

- `reorder_fraction`: high when stockout/lead-time/safety-stock pressure is high; low when holding pressure is high and inventory risk is low.
- `dispatch_intensity`: high when useful dispatch opportunity, urgent demand, or SLA pressure is high; lower when feasible dispatch ratio is poor.
- `speed_multiplier`: above 1.0 only when lateness/urgent/route disruption pressure justifies speed; near 1.0 otherwise.
- `safety_stock_multiplier`: high under stockout/safety-stock/lead-time pressure; suppressed under high holding cost with low stockout risk.
- `capacity_buffer_fraction`: high under capacity shock, vehicle scarcity, route disruption, or flow pressure.

Recommended bounded policy sketch:

```text
inventory_pressure = max(stockout_risk, safety_stock_gap, 1 - inventory_coverage, lead_time_pressure)
flow_pressure = max(useful_dispatch_opportunity, urgent_ratio, lateness_risk, premium_sla_pressure)
route_pressure = max(route_disruption_pressure, congestion_pressure)
scarcity_pressure = max(vehicle_availability_pressure, capacity_shock_pressure, 1 - feasible_dispatch_ratio)

reorder_fraction_target = clamp(inventory_pressure * (1 - 0.6 * holding_cost_pressure), 0, 1)
dispatch_intensity_target = clamp(max(flow_pressure, useful_dispatch_opportunity) * feasible_dispatch_ratio, 0, 1)
speed_multiplier_target = clamp(1.0 + 0.25 * max(lateness_risk, urgent_ratio, route_pressure), 0.65, 1.35)
safety_stock_multiplier_target = clamp(1.0 + inventory_pressure * (1 - 0.5 * holding_cost_pressure), 1.0, max_safety_stock_multiplier)
capacity_buffer_target = max_capacity_buffer_fraction * clamp(max(scarcity_pressure, route_pressure, flow_pressure), 0, 1)
```

Implementation note:

- Convert physical targets back to normalized `[-1, 1]` using the inverse of `ActionProjector`.
- Keep inverse helper tested and projector-consistent.
- Never hand-tune per scenario after seeing benchmark outcomes unless the report labels it as tuned and non-comparable.

Fairness:

- More PPO-like than fixed zero because it uses state-sensitive continuous controls.
- Still a human-designed heuristic, not a learned PPO.
- Fair as a transparent operations baseline when compared on same scenarios, seeds, and episode budgets.

### Mode 3: `ppo_assisted_continuous`

Definition:

- Use the production PPO continuous action provider read-only, paired with rule-based discrete action ids.
- The discrete tactical policy remains rule-based; the continuous operational tone comes from the learned PPO.

Implementation stance:

- The first skeleton should not load production checkpoints by default.
- It should define an injectable `ContinuousActionProvider` interface.
- Unit tests should use a fake provider that returns a legal 5-dimensional vector.
- A later explicitly approved benchmark can wire this provider to `load_torch_joint_policy` or `PolicyService` read-only.

Fairness:

- Strong diagnostic hybrid.
- Answers: "If the learned PPO continuous controls are held constant, do simple rule tactical choices still lag the learned DQN tactical policy?"
- Not a pure rule baseline.
- Must be reported separately from `heuristic_continuous` and `neutral_continuous`.

Advisor interpretation:

- "PPO-assisted rule baseline isolates the tactical DQN contribution by pairing learned continuous controls with transparent discrete rules."

Overclaim warning:

- It cannot be used to say a pure operator heuristic equals the learned system, because part of the learned system is still present.

### Mode 4: `oracle_diagnostic_continuous` (optional, non-deployable)

Definition:

- Uses hindsight, future episode metrics, or privileged scenario labels to choose continuous controls.

Use:

- Diagnostic upper-bound only.
- Must be disabled by default.
- Must be excluded from advisor-facing fair-comparison tables unless clearly marked.

Fairness:

- Not deployable.
- Not comparable to production or real-world execution.
- Useful only to test whether continuous control is the bottleneck.

## Recommended Skeleton Architecture

Future file scope:

- `src/eval/rule_based_baseline_arena.py`
- `tests/eval/test_rule_based_baseline_arena.py`
- `docs/runs/20260613_continuous_aware_baseline_skeleton_report.md`
- `docs/00_PROJECT_DASHBOARD.md` links/status only if useful

New concepts:

- `ContinuousBaselineMode` enum:
  - `neutral_continuous`
  - `heuristic_continuous`
  - `ppo_assisted_continuous`
  - `oracle_diagnostic_continuous`
- `ContinuousActionProvider` protocol/callable for PPO-assisted mode.
- `ContinuousRuleContext`, either reusing `RuleBasedDecisionContext` plus optional observation raw feature map, or extending the context with fields already available in env/snapshot.
- `build_continuous_action(mode, context, provider=None, projector_config=None) -> np.ndarray`
- `build_joint_action_from_rule_decision(decision, continuous_mode=..., continuous_context=..., provider=None)`

Default:

- Preserve current default as `neutral_continuous` to avoid silently changing historical interpretation.

## Tests Required In Future Skeleton

TDD required. Add failing tests before implementation:

1. `neutral_continuous` returns length 5, finite, bounded normalized values and equals the existing zero vector.
2. `neutral_continuous` projection semantics test documents midpoint physical controls, using `ActionProjector`.
3. `heuristic_continuous` returns length 5, finite, bounded normalized values for low pressure and high pressure contexts.
4. `heuristic_continuous` increases reorder/safety-stock targets under inventory pressure.
5. `heuristic_continuous` suppresses reorder posture under high holding pressure when inventory risk is low.
6. `heuristic_continuous` increases dispatch intensity under useful work/SLA pressure but gates by feasibility.
7. `heuristic_continuous` increases capacity buffer under vehicle scarcity/capacity shock.
8. `ppo_assisted_continuous` uses the injected provider and does not load checkpoints in unit tests.
9. `ppo_assisted_continuous` rejects provider output with wrong shape, non-finite values, or values outside normalized bounds.
10. `oracle_diagnostic_continuous` is marked non-deployable in metadata and is disabled unless explicitly allowed.
11. Existing discrete action validation for 0..47 remains unchanged.
12. Action 24 and action 32 remain legal external action ids through `DiscreteActionMapper`.
13. Fake-env rollout still feeds `EpisodeMetricAccumulator.observe` without writing benchmark outputs.
14. No code writes under `models/eval` or other protected output roots.

## Future Report Metadata

Any later benchmark report must include:

- `continuous_mode`
- `continuous_mode_description`
- `continuous_action_dim = 5`
- `continuous_action_source`
  - `fixed_zero_normalized`
  - `heuristic_current_state_only`
  - `production_ppo_read_only`
  - `oracle_hindsight_non_deployable`
- `continuous_vector_bounds = [-1, 1]`
- `physical_projection_fields`
- `ppo_assisted_checkpoint` only if a later approved run loads it read-only
- `fairness_caveat`
- `deployable = true/false`

## Interpretation Ladder

Recommended future benchmark table should separate modes:

1. `neutral_continuous`: tactical-rule sanity check.
2. `heuristic_continuous`: full transparent heuristic baseline.
3. `ppo_assisted_continuous`: learned-continuous + rule-discrete diagnostic.
4. `oracle_diagnostic_continuous`: non-deployable upper-bound diagnostic.

Do not collapse these into one "rule baseline" score. Their fairness claims differ.

## Stop Conditions

Stop before implementation or benchmark if:

- future code needs to change `env_5pl` physics;
- future code changes obs/action contract `73/5/48`;
- future code loads production checkpoints without explicit approval;
- future code writes benchmark outputs without explicit approval;
- output paths would edit existing eval/benchmark outputs;
- any mode uses future episode outcomes without `oracle_diagnostic_continuous` labeling;
- PPO-assisted mode cannot prove read-only checkpoint loading;
- tests cannot prove vectors are bounded, finite, and 5-dimensional.

## Recommended Next Step

Create the skeleton only:

```text
/goal Read docs/goals/20260613_execute_continuous_aware_baseline_skeleton_goal.txt and execute it exactly.
```

That future goal must not run a benchmark. It should only add the mode seam, bounded vector helpers, tests, and a skeleton report.
