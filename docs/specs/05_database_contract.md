# Database Contract

The Sense database is the PostgreSQL source of truth for operational state, high-velocity telemetry, and joint training traces.

## Required Extensions

- TimescaleDB for hypertables, chunks, compression, and retention.
- PostGIS for geometry and geography query paths.

## Required Tables

- `sense_devices`
- `sense_assets`
- `sense_routes`
- `iiot_container_telemetry`
- `ops_shipment_states`
- `sense_inventory_levels`
- `sense_network_events`
- `sense_episode_traces`
- `transient_agent_state`
- `latest_asset_observations`
- `latest_shipment_observations`

## Telemetry Identity

`iiot_container_telemetry.telemetry_id` must be `BIGSERIAL` so append-heavy IIoT telemetry receives automatic monotonic row identifiers while TimescaleDB partitions by `recorded_at`.

## Joint Trace Fields

`sense_episode_traces` stores:

- `policy_id`
- `agent_id`
- `agent_role`
- `team_id`
- `joint_action_id`
- `local_reward`
- `global_reward`
- `ppo_local_reward`
- `dqn_local_reward`
- `info` JSONB with `reward_components`

## Indexing Requirements

- BRIN indexes on append-heavy time columns.
- GiST indexes on PostGIS geometry/geography columns.
- GIN index on trace `info` JSONB for reward-component and safety diagnostics.
- Composite lookup indexes for policy, role, team, joint action, asset, device, shipment, route, and inventory query paths.

## Hypertable Strategy

One-day chunks align with the 288-step fixed horizon at 300 seconds per decision cycle.
