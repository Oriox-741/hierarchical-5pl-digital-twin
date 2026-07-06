# System Architecture Specification

The framework is a 5PL control tower built around a continuous `Sense -> Think -> Act -> Learn` loop. The current executable architecture is a synchronized joint PPO+DQN system trained from scratch under a fixed 288-step rolling operations MDP.

The final training architecture uses one joint transition collector: each environment step records the shared observation, PPO continuous action, DQN discrete action, next observation, reward decomposition, and safety/projection metadata as a single atomic transition. PPO and DQN update from that same rollout stream, preserving role-isolated reward components while writing two audit rows per joint transition that share one `joint_action_id`.

## Layer Responsibilities

- `Sense`: ingest physical and synthetic telemetry into PostgreSQL, TimescaleDB, and PostGIS.
- `Think`: coordinate PPO strategic continuous control and DQN tactical discrete control.
- `Act`: execute atomic joint actions inside the SimPy/Gymnasium digital twin.
- `Learn`: persist state-action-reward traces, reward decomposition, policy diagnostics, and model registry metadata.

## Control Tower Contract

- Default action mode: `joint`.
- Episode horizon: `288` steps.
- Decision interval: `300` simulated seconds.
- Demand model: rolling stochastic order arrivals throughout the episode.
- Termination: catastrophic safety failure only.
- Truncation: normal fixed-horizon completion at step `288`.

## Agent Roles

- PPO strategic controller: capacity buffers, safety stock, speed discipline, and macro resource readiness.
- DQN tactical controller: dispatch/hold, route choice, transport mode, and reorder category.

## Architectural Constraint

Only open-source components are allowed. Proprietary simulation or optimization tooling is out of scope.
