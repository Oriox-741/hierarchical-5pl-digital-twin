# MDP Contract

The Gymnasium-facing MDP is a fixed-horizon rolling operations problem. It exposes dense normalized logistics observations and accepts atomic joint actions containing both PPO continuous controls and DQN discrete controls.

## Horizon

- `max_steps = 288`
- `decision_interval_seconds = 300`
- one episode represents one simulated operational day
- full delivery of currently visible orders is an operational milestone, not a terminal success state

## Observation Inputs

- asset telemetry and positions
- shipment status and ETA
- inventory levels by node and product
- congestion and disruption coefficients
- capacity utilization
- Safety Potential
- rolling demand and order-flow state

## Joint Action

The environment accepts:

```text
{
  "continuous": PPO vector with 5 normalized controls,
  "discrete": DQN integer action
}
```

PPO controls are applied first to reshape the strategic capacity/safety envelope. DQN controls are then projected through the updated safety context and applied tactically in the same step.

Each accepted joint action produces one synchronized transition stream containing the shared observation, both role actions, the next observation, reward components, termination flags, and safety/projection metadata. The PyTorch joint trainer consumes this exact transition for both PPO rollout learning and DQN replay learning, so the strategic and tactical policies never train against stale opponent behavior.

## PPO Continuous Controls

- reorder fraction
- dispatch intensity
- speed multiplier
- safety-stock multiplier
- capacity-buffer fraction

PPO owns planned inventory posture. `reorder_fraction` represents planned replenishment intensity and
`safety-stock multiplier` represents the target safety-stock buffer. Planned replenishment must be
tracked separately from emergency procurement so inventory credit is assigned to the strategic
controller rather than hidden inside tactical override actions.

PPO speed control is baseline-relative. A speed action sets the vehicle's operating speed relative to
its stable baseline speed for that decision cycle; it must not multiply the previously adjusted speed.
This prevents physically impossible compounding velocity drift across repeated decisions.

## DQN Discrete Controls

- dispatch or hold
- route choice
- transport mode
- reorder trigger level

## Inventory Ownership Audit

The pre-refactor MDP exposed redundant replenishment control. PPO carried continuous
`reorder_fraction` and `safety-stock multiplier` controls, while DQN carried a discrete
`ReorderDecision` ladder. Both paths reached the same inventory mutation helper, so planned
inventory posture, emergency procurement, and safety-stock target changes were not cleanly
attributable. This weakened credit assignment and allowed PPO to collapse toward near-zero
planned replenishment while DQN absorbed emergency inventory behavior.

DQN owns tactical execution and emergency override. `NONE` and `CONSERVATIVE` do not create
additional stock beyond PPO's planned replenishment posture. `AGGRESSIVE` and `EMERGENCY` represent
explicit override procurement paths and must be traceable as higher-cost operational events. Emergency
procurement can be justified by severe stockout or backlog risk, but it is never free inventory.

## Episode End

- `terminated`: catastrophic safety/business failure.
- `truncated`: normal fixed-horizon completion at step `288`.
