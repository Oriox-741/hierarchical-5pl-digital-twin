# Master Development Plan: Open-Source Autonomous 5PL Digital Twin Framework

## 1. Executive Summary

This plan uses only open-source technologies and follows the required continuous `Sense -> Think -> Act -> Learn` architecture.

The target system is a cognitive 5PL meta-organization: a self-optimizing digital twin network that replaces human-mediated 4PL control towers with autonomous perception, simulation, decision, execution, and learning loops.

The architecture will use:

- **Sense:** PostgreSQL, TimescaleDB, PostGIS, BRIN/GiST indexes, hypertables, declarative partitioning, asyncpg.
- **Act:** Python, SimPy discrete-event simulation, Gymnasium environment wrappers.
- **Think:** raw PyTorch synchronized joint MARL, with PPO for continuous strategic controls and DQN for discrete tactical decisions trained from the same transition stream.
- **Learn:** synthetic traces, state-action-reward storage, Safety Potential reward shaping, role-isolated reward diagnostics, `.pt` checkpoint registry, and continual adaptation.

The aborted Stable-Baselines3 staggered co-training loop is retained only under `src/archive_aborted_sb3/` for thesis autopsy and control-group comparison. The active final trainer is `src.learn.train_joint_torch`, which eliminates stale-opponent policy lag by collecting one atomic joint transition per environment step and updating PPO and DQN from that shared data stream.

Operations-research models such as **LRP-MPPD-2E**, VRPTW, dynamic berth allocation, and multi-echelon inventory optimization will define constraints, feasibility rules, reward terms, and benchmark scenarios rather than becoming proprietary solver dependencies.

## 2. Phased Breakdown

### Phase 1: Architectural Foundation and Mathematical Contracts

**Layer mapping:** All layers

**Objectives**

- Formalize the project as a modular `Sense -> Think -> Act -> Learn` system.
- Translate the source documents' theory into implementation contracts.
- Define core logistics entities: hubs, depots, customers, vehicles, products, orders, arcs, disruptions, inventory echelons.
- Define mathematical baselines:
  - LRP-MPPD-2E for two-echelon multi-product routing, pickup, delivery, hub activation, vehicle assignment, capacity, time windows, and flow conservation.
  - MEIO for safety stock, lead-time uncertainty, demand volatility, and bullwhip mitigation.
  - Safety Potential for proactive risk suppression.

**Technologies**

- Python project structure.
- Markdown specs.
- PostgreSQL-compatible SQL design documents.

**Components to define**

- `docs/specs/01_system_architecture.md`
- `docs/specs/02_mathematical_contracts.md`
- `docs/specs/03_mdp_contract.md`
- `docs/specs/04_reward_contract.md`
- `docs/specs/05_database_contract.md`

### Phase 2: Sense Layer Database Core

**Layer mapping:** Sense

**Objectives**

- Build the spatiotemporal persistence backbone.
- Store real IIoT telemetry and synthetic digital-twin traces.
- Support high-volume inserts and low-latency state polling for RL agents.

**Technologies**

- PostgreSQL.
- TimescaleDB hypertables.
- PostGIS geometry/geography columns.
- Declarative range partitioning.
- BRIN indexes for append-only time-series data.
- GiST indexes for spatial and spatiotemporal queries.
- UNLOGGED staging tables for transient observations.

**Key components/scripts**

- `db/extensions.sql`: enable TimescaleDB and PostGIS.
- `db/schema_sense.sql`: telemetry, device, asset, route, inventory, event, and trace tables.
- `db/partitions.sql`: hypertable setup, chunk policy, retention/archive policy.
- `db/indexes.sql`: BRIN on time columns, GiST on geometry, composite indexes for asset trajectory queries.
- `db/staging.sql`: transient agent-state and latest-observation tables.

### Phase 3: Sense Layer Async Ingestion and Snapshot Services

**Layer mapping:** Sense

**Objectives**

- Implement high-throughput ingestion without ORM overhead.
- Support physical IIoT streams and synthetic trace ingestion.
- Provide fast latest-state snapshots to the Act and Think layers.

**Technologies**

- Python.
- asyncpg.
- PostgreSQL binary COPY / prepared statements.

**Key components/scripts**

- `src/sense/db_pool.py`: asyncpg pool lifecycle.
- `src/sense/telemetry_ingest.py`: append-only telemetry ingestion.
- `src/sense/state_snapshot.py`: latest network-state retrieval.
- `src/sense/synthetic_stream.py`: synthetic telemetry generator for early development.
- `src/sense/trace_writer.py`: state-action-reward trace persistence.

### Phase 4: Act Layer Digital Twin Simulation Core

**Layer mapping:** Act

**Objectives**

- Build the open-source discrete-event digital twin.
- Model hubs, vehicles, inventory, orders, queues, capacity, transit time, disruptions, and recovery.
- Encode OR constraints from LRP-MPPD-2E and MEIO as simulation rules.

**Technologies**

- Python.
- SimPy.

**Key components/scripts**

- `src/act/sim_engine.py`: SimPy environment orchestration.
- `src/act/entities.py`: hubs, customers, vehicles, products, orders.
- `src/act/resources.py`: warehouse capacity, fleet capacity, dock/berth capacity.
- `src/act/routing_physics.py`: dynamic transit time, congestion, route feasibility.
- `src/act/inventory_dynamics.py`: multi-echelon inventory updates.
- `src/act/disruptions.py`: node failure, arc delay, demand shock, capacity loss.

### Phase 5: Act Layer Gymnasium Environment and Actuator

**Layer mapping:** Act + Think bridge

**Objectives**

- Wrap the SimPy digital twin in a Gymnasium-compatible environment.
- Standardize `reset()`, `step()`, observation, action, reward, termination, and truncation behavior.
- Translate normalized RL outputs into physical logistics actions.
- Run the active training contract through `FivePLDigitalTwinEnv(action_mode="joint", max_steps=288)`, where each decision step accepts one atomic joint action containing both continuous PPO controls and a discrete DQN tactical action.

**Technologies**

- Gymnasium.
- SimPy.
- NumPy-compatible observation/action arrays.

**Key components/scripts**

- `src/act/env_5pl.py`: custom `gymnasium.Env` and final 288-step fixed-horizon joint transition collector.
- `src/act/observation_builder.py`: converts DB/simulation state into flat observation vectors.
- `src/act/action_projector.py`: maps PPO `[-1, 1]` outputs to reorder quantities, speed changes, dispatch rates, capacity buffers.
- `src/act/discrete_action_mapper.py`: maps DQN actions to dispatch, route, mode, or reorder decisions.
- `src/act/safety_projector.py`: blocks or projects unsafe actions before execution.

### Phase 6: Think Layer Reward, Feasibility, and Synchronized Joint Agents

**Layer mapping:** Think

**Objectives**

- Implement custom non-linear reward mechanisms.
- Train PPO and DQN as synchronized role-specialized agents from shared joint transitions.
- Separate strategic continuous controls and tactical discrete decisions while preventing stale-opponent training.

**Technologies**

- PyTorch.
- Gymnasium.

**Reward design**

- Penalize operational cost, holding cost, transit cost, backlog, late delivery, failed service level, unsafe routing, and excessive cash conversion cycle.
- Reward service-level attainment, throughput, resilience, and stable inventory positioning.
- Add **Safety Potential** shaping to penalize rising risk before visible failure occurs.

**Key components/scripts**

- `src/think/rewards.py`: weighted multi-objective reward and Safety Potential terms.
- `src/think/feasibility.py`: capacity, time-window, flow, and inventory feasibility checks.
- `src/think/joint_policies.py`: PyTorch feature extractors, PPO actor/critic, DQN Q-network, target-network utilities, and `.pt` checkpoint helpers.
- `src/think/train_ppo.py`: deprecated compatibility shim pointing to the final joint trainer.
- `src/think/train_dqn.py`: deprecated compatibility shim pointing to the final joint trainer.
- `src/think/policies.py`: legacy policy extensions retained for historical SB3 evaluation utilities.
- `src/think/evaluate_policy.py`: benchmark trained agents against heuristics.

### Phase 7: Learn Layer Training Trace Store and Feedback Loop

**Layer mapping:** Learn

**Objectives**

- Persist episodes as reusable learning assets.
- Support simulation-generated experience, evaluation, replay-style analysis, and continual adaptation.
- Build the loop: synchronized joint transition -> reward audit -> PPO rollout update and DQN replay update -> `.pt` policy registry -> re-simulation.
- Persist two role rows per joint transition, with PPO and DQN rows sharing the same `joint_action_id`.

**Technologies**

- PostgreSQL / TimescaleDB.
- asyncpg.
- PyTorch.

**Key components/scripts**

- `src/learn/episode_store.py`: writes and reads episode traces.
- `src/learn/replay_dataset.py`: builds training/evaluation datasets from traces.
- `src/learn/curriculum.py`: generates harder scenarios over time.
- `src/learn/fine_tune.py`: adapts trained policies to new nodes or changed demand/cost profiles.
- `src/learn/joint_buffers.py`: synchronized PPO rollout and compact DQN replay buffers.
- `src/learn/joint_metrics.py`: role reward, action variance, action entropy, and safety/projection diagnostics.
- `src/learn/train_joint_torch.py`: final raw PyTorch synchronized PPO+DQN MARL trainer.
- `src/learn/model_registry.py`: stores model metadata, metrics, and active policy pointers.

### Phase 8: Closed-Loop Autonomous Orchestration

**Layer mapping:** Sense + Think + Act + Learn

**Objectives**

- Connect all layers into a continuous runtime loop.
- Read current state from Sense.
- Run synchronized PyTorch policy inference in Think.
- Execute/project decisions in Act.
- Persist outcomes and update learning assets in Learn.

**Technologies**

- Python asyncio.
- asyncpg.
- SimPy.
- Gymnasium.
- PyTorch `.pt` policy artifacts.

**Key components/scripts**

- `src/orchestration/runtime_loop.py`: main autonomous loop.
- `src/orchestration/policy_service.py`: loads active PPO/DQN `.pt` models and runs inference.
- `src/orchestration/decision_audit.py`: records observation, action, reward, and safety projection details.
- `src/orchestration/scenario_runner.py`: executes benchmark and disruption scenarios.

### Phase 9: Multi-Agent 5PL Expansion

**Layer mapping:** Think + Act + Learn

**Objectives**

- Extend from one decision agent to a multi-agent meta-organization.
- Model vehicles, hubs, suppliers, and distribution centers as decentralized actors.
- Use shared policies for homogeneous agents and centralized training concepts where needed.

**Technologies**

- PyTorch.
- Gymnasium-compatible multi-agent wrappers.
- SimPy.
- PostgreSQL trace storage.

**Key components/scripts**

- `src/think/multi_agent.py`: shared-policy and centralized-critic training scaffolds.
- `src/act/multi_agent_env.py`: multi-agent environment wrapper.
- `src/learn/self_play.py`: disruption proposer, logistics solver, verifier loop.
- `src/learn/counterfactual_metrics.py`: agent contribution and credit-assignment analysis.

### Phase 10: Validation, Benchmarking, and Hardening

**Layer mapping:** All layers

**Objectives**

- Prove that each layer works alone and inside the full loop.
- Compare autonomous policies against deterministic heuristics.
- Validate safety, stability, and performance before real IIoT integration.

**Test scenarios**

- High-volume telemetry ingestion with partition pruning.
- BRIN time-window query benchmark.
- GiST/PostGIS trajectory lookup benchmark.
- Deterministic SimPy reset reproducibility.
- LRP-MPPD-2E constraint violations: capacity overflow, missed customer, time-window breach.
- MEIO scenarios: stockout, excess stock, demand shock, lead-time variance.
- Synchronized joint PPO+DQN convergence under the 288-step rolling MDP.
- PPO local reward stability, DQN local reward stability, and global reward co-adaptation.
- `.pt` checkpoint save/load and model-registry activation.
- Verification that no active training path uses the archived SB3 staggered loop.
- Safety Potential early-intervention scenario.
- Closed-loop `Sense -> Think -> Act -> Learn` episode persistence.
- Crash recovery for UNLOGGED transient state by refreshing from simulation or sensors.

## 3. Important Interfaces and Data Contracts

- **Telemetry record:** `device_id`, `asset_id`, `recorded_at`, `location`, `state_vector`, `sensor_type`, `source`.
- **Observation vector:** inventory levels, in-transit orders, demand forecasts, node capacities, vehicle states, congestion coefficients, disruption coefficients, Safety Potential indicators.
- **Atomic joint action:** `{"continuous": PPO vector in [-1, 1], "discrete": DQN integer action}` applied within the same environment step.
- **Continuous PPO role:** normalized controls mapped to order quantity, dispatch intensity, speed adjustment, safety stock buffer, or capacity allocation.
- **Discrete DQN role:** route choice, mode choice, dispatch/no-dispatch, reorder trigger, disruption response class.
- **Role-isolated rewards:** `global`, `ppo_local`, and `dqn_local`, plus train rewards derived from configured global/local weights.
- **Trace record:** two rows per joint transition, one PPO row and one DQN row, sharing `joint_action_id`, `episode_id`, `step_id`, observation, next observation, and reward-component metadata.
- **Gymnasium contract:** `reset()` returns initial observation and info; `step(action)` advances SimPy to the next decision point and returns observation, reward, terminated, truncated, info.

## 4. Suggested Modular Folder Structure

```text
5PL-Autonomous-Ecosystem/
├── README.md
├── requirements.txt
├── docs/
│   ├── source/
│   │   ├── AI-Assisted Digital Twin Architecture Blueprint.docx
│   │   └── Autonomous 5PL AI Ecosystem Research last.docx
│   └── specs/
│       ├── 01_system_architecture.md
│       ├── 02_mathematical_contracts.md
│       ├── 03_mdp_contract.md
│       ├── 04_reward_contract.md
│       └── 05_database_contract.md
├── db/
│   ├── extensions.sql
│   ├── schema_sense.sql
│   ├── staging.sql
│   ├── indexes.sql
│   ├── partitions.sql
│   └── seed_scenarios.sql
├── configs/
│   ├── simulation.json
│   ├── training_joint.json
│   └── reward_weights.json
├── src/
│   ├── sense/
│   │   ├── db_pool.py
│   │   ├── telemetry_ingest.py
│   │   ├── state_snapshot.py
│   │   ├── synthetic_stream.py
│   │   └── trace_writer.py
│   ├── act/
│   │   ├── sim_engine.py
│   │   ├── entities.py
│   │   ├── resources.py
│   │   ├── routing_physics.py
│   │   ├── inventory_dynamics.py
│   │   ├── disruptions.py
│   │   ├── env_5pl.py
│   │   ├── action_projector.py
│   │   └── safety_projector.py
│   ├── think/
│   │   ├── rewards.py
│   │   ├── feasibility.py
│   │   ├── joint_policies.py
│   │   ├── policies.py
│   │   ├── train_ppo.py
│   │   ├── train_dqn.py
│   │   └── evaluate_policy.py
│   ├── learn/
│   │   ├── episode_store.py
│   │   ├── replay_dataset.py
│   │   ├── curriculum.py
│   │   ├── fine_tune.py
│   │   ├── joint_buffers.py
│   │   ├── joint_metrics.py
│   │   ├── train_joint_torch.py
│   │   ├── model_registry.py
│   │   └── self_play.py
│   ├── orchestration/
│   │   ├── runtime_loop.py
│   │   ├── policy_service.py
│   │   ├── decision_audit.py
│   │   └── scenario_runner.py
│   └── shared/
│       ├── types.py
│       ├── constants.py
│       └── metrics.py
├── tests/
│   ├── sense/
│   ├── act/
│   ├── think/
│   ├── learn/
│   └── integration/
└── models/
    ├── checkpoints/
    ├── registry/
    └── reports/
```

## Assumptions and Defaults

- Development begins with synthetic digital-twin data, then physical IIoT adapters are added later.
- PostgreSQL is the single source of operational truth; transient state may use UNLOGGED tables when durability is not required.
- PPO is the strategic continuous controller; DQN is the tactical discrete controller.
- SimPy is the only simulation engine; no proprietary simulation tooling is used.
- Safety Potential is implemented both as reward shaping and as actuator-side action projection.
- `src.learn.train_joint_torch` is the active synchronized trainer; `.pt` checkpoints are the active model artifacts.
- The historical SB3 staggered loop is aborted and archived; historical isolated checkpoints are retained only for control-group analysis.
