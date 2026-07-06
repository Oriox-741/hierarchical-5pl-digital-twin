# Reward Contract

The active reward contract is decomposed for joint PPO+DQN co-adaptation. The scalar environment reward remains available, but training and audit traces must preserve role-isolated components so PPO is credited for strategic envelope quality and DQN is credited for tactical feasibility.

## Components

- `global`: network service level, throughput, resilience, and Safety Potential.
- `ppo_local`: capacity readiness, inventory health, speed/cost discipline, and buffer quality.
- `dqn_local`: dispatch feasibility, route/mode transport efficiency, and emergency-action discipline.

## Timing Rule

Rolling demand generated at the end of step `T` is masked from pending-order penalties until the agents have one full decision cycle to observe and react at step `T+1`.

## Economic Discipline

The reward penalizes:

- excessive transport cost
- unjustified emergency reorders
- unnecessary premium fleet sourcing
- unsafe or brittle route choices
- backlog, stockout, and capacity overflow

## Training Diagnostics

Joint training must log:

- mean global reward
- mean PPO local reward
- mean DQN local reward
- PPO action variance
- DQN action entropy
- projection and blocking outcomes

Trace persistence writes two audit rows per joint transition: one PPO row and one DQN row. Both rows share the same `joint_action_id`, episode, step, observation, next observation, and global reward; each row stores its own role-local reward while also retaining `ppo_local`, `dqn_local`, and `global` inside `info.reward_components`.
