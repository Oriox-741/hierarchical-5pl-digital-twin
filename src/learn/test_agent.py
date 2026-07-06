"""Large-scale deterministic evaluation for the fine-tuned DQN 5PL agent."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from src.act.disruptions import DisruptionEvent, DisruptionKind
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.act.routing_physics import make_arc_from_points
from src.act.sim_engine import DigitalTwinSimulation


DEFAULT_MODEL_PATH = Path("models/baselines/historical_isolated/dqn_5pl_finetuned.zip")
DEFAULT_CONFIG = Path("configs/training_joint.json")
DEFAULT_EPISODES = 100
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = int(
    json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))["shared_global_parameters"]["max_steps"]
)


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    episode: int
    seed: int
    total_reward: float
    episode_length: int
    service_level: float
    safety_potential: float
    total_transport_cost: float
    disruption_score: float
    delivered_orders: int
    late_orders: int
    failed_orders: int
    projected_actions: int
    blocked_actions: int
    terminated: bool
    truncated: bool
    route_decisions: Counter[str] = field(default_factory=Counter)
    mode_decisions: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    episodes: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    episode_length_mean: float
    episode_length_std: float
    service_level_mean: float
    service_level_std: float
    safety_potential_mean: float
    safety_potential_std: float
    transport_cost_mean: float
    transport_cost_std: float
    disruption_score_mean: float
    delivered_orders_mean: float
    late_orders_mean: float
    failed_orders_mean: float
    projected_action_rate: float
    blocked_action_rate: float
    termination_rate: float
    truncation_rate: float
    route_decisions: Counter[str]
    mode_decisions: Counter[str]


def build_vec_env(seed: int, *, max_steps: int = DEFAULT_MAX_STEPS) -> DummyVecEnv:
    """Build a one-environment VecEnv matching the fine-tuned DQN action space."""
    def make_env() -> FivePLDigitalTwinEnv:
        config = EnvironmentConfig(
            action_mode="discrete",
            random_seed=seed,
            max_steps=max_steps,
        )
        env = FivePLDigitalTwinEnv(config)
        default_factory = env.simulation_factory

        def stress_factory(episode_seed: int | None) -> DigitalTwinSimulation:
            effective_seed = seed if episode_seed is None else int(episode_seed)
            simulation = default_factory(effective_seed)
            _apply_seeded_stressors(simulation, effective_seed)
            return simulation

        env.simulation_factory = stress_factory
        env.simulation = stress_factory(seed)
        env._previous_snapshot = env.simulation.snapshot()
        return env

    return DummyVecEnv(
        [
            make_env
        ]
    )


def load_dqn_model(model_path: Path, env: DummyVecEnv) -> DQN:
    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned DQN checkpoint not found: {model_path}")
    return DQN.load(model_path, env=env)


def _apply_seeded_stressors(simulation: DigitalTwinSimulation, seed: int) -> None:
    rng = random.Random(seed)

    for customer in simulation.customers.values():
        demand_multiplier = rng.uniform(0.55, 2.85)
        service_time_multiplier = rng.uniform(0.60, 2.20)
        customer.service_time *= service_time_multiplier
        for product_id, demand in list(customer.demand_units.items()):
            customer.demand_units[product_id] = max(1.0, demand * demand_multiplier)

    for order in simulation.orders.values():
        order.release_time = rng.uniform(0.0, 900.0)
        order.due_time = rng.uniform(1_800.0, 24_000.0)
        order.metadata["stress_seed"] = seed
        demand_multiplier = rng.uniform(0.55, 2.85)
        for line in order.lines:
            line.quantity_units = max(1.0, line.quantity_units * demand_multiplier)

    for position in simulation.inventory_network.positions.values():
        position.on_hand_units *= rng.uniform(0.45, 1.60)
        position.safety_stock_units *= rng.uniform(0.50, 1.75)
        position.demand_mean *= rng.uniform(0.65, 2.25)
        position.demand_variance *= rng.uniform(0.75, 3.00)
        position.lead_time_mean *= rng.uniform(0.65, 2.40)
        position.lead_time_variance *= rng.uniform(0.75, 3.20)

    for vehicle in simulation.vehicles.values():
        vehicle.nominal_speed_mps *= rng.uniform(0.65, 1.35)
        vehicle.fixed_usage_cost *= rng.uniform(0.70, 1.90)
        vehicle.transport_cost_per_meter *= rng.uniform(0.70, 2.25)
        vehicle.capacity_units *= rng.uniform(0.70, 1.35)

    _add_seeded_route_alternatives(simulation, rng)
    _seed_route_conditions(simulation, rng)
    _schedule_seeded_disruptions(simulation, rng)


def _add_seeded_route_alternatives(simulation: DigitalTwinSimulation, rng: random.Random) -> None:
    customers = list(simulation.customers.values())
    if len(customers) < 2:
        return

    for origin in customers:
        for destination in customers:
            if origin.customer_id == destination.customer_id:
                continue
            arc_key = (origin.customer_id, destination.customer_id)
            if arc_key in simulation.route_network.arcs:
                continue
            simulation.add_route_arc(
                make_arc_from_points(
                    origin_node_id=origin.customer_id,
                    destination_node_id=destination.customer_id,
                    origin=origin.location,
                    destination=destination.location,
                    nominal_speed_mps=rng.uniform(9.0, 19.0),
                    capacity_units=rng.uniform(60.0, 260.0),
                    metadata={"risk_score": rng.uniform(0.0, 1.0), "stress_alternative": True},
                )
            )


def _seed_route_conditions(simulation: DigitalTwinSimulation, rng: random.Random) -> None:
    for origin_node_id, destination_node_id in simulation.route_network.arcs:
        simulation.routing_physics.set_congestion(origin_node_id, destination_node_id, rng.uniform(0.0, 0.95))
        if rng.random() < 0.45:
            simulation.routing_physics.set_arc_delay_multiplier(
                origin_node_id,
                destination_node_id,
                rng.uniform(1.05, 3.50),
            )


def _schedule_seeded_disruptions(simulation: DigitalTwinSimulation, rng: random.Random) -> None:
    route_keys = list(simulation.route_network.arcs.keys())
    for origin_node_id, destination_node_id in rng.sample(route_keys, k=min(len(route_keys), rng.randint(1, 3))):
        simulation.schedule_disruption(
            DisruptionEvent(
                event_id=uuid4(),
                kind=DisruptionKind.ARC_DELAY,
                starts_at=rng.uniform(300.0, 7_200.0),
                duration=rng.uniform(900.0, 10_800.0),
                severity=rng.uniform(0.20, 0.90),
                target_id=origin_node_id,
                payload={
                    "origin_node_id": str(origin_node_id),
                    "destination_node_id": str(destination_node_id),
                },
            )
        )

    capacity_targets = list(simulation.capacity_states)
    if capacity_targets:
        target_id = rng.choice(capacity_targets)
        simulation.schedule_disruption(
            DisruptionEvent(
                event_id=uuid4(),
                kind=DisruptionKind.CAPACITY_LOSS,
                starts_at=rng.uniform(600.0, 8_400.0),
                duration=rng.uniform(1_200.0, 12_000.0),
                severity=rng.uniform(0.15, 0.70),
                target_id=target_id,
            )
        )

    inventory_targets = [position.node_id for position in simulation.inventory_network.positions.values()]
    if inventory_targets:
        target_id = rng.choice(inventory_targets)
        simulation.schedule_disruption(
            DisruptionEvent(
                event_id=uuid4(),
                kind=DisruptionKind.DEMAND_SHOCK,
                starts_at=rng.uniform(300.0, 6_000.0),
                duration=rng.uniform(1_200.0, 9_000.0),
                severity=rng.uniform(0.15, 0.85),
                target_id=target_id,
            )
        )


def evaluate_agent(
    model: DQN,
    env: DummyVecEnv,
    *,
    episodes: int,
    seed: int,
) -> list[EpisodeMetrics]:
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    metrics: list[EpisodeMetrics] = []
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        base_config = env.get_attr("config")[0]
        if isinstance(base_config, EnvironmentConfig):
            base_config.random_seed = episode_seed
            env.set_attr("config", base_config)
        env.seed(episode_seed)
        observation = env.reset()

        total_reward = 0.0
        episode_length = 0
        projected_actions = 0
        blocked_actions = 0
        terminated = False
        truncated = False
        latest_info: dict[str, Any] = {}
        route_decisions: Counter[str] = Counter()
        mode_decisions: Counter[str] = Counter()

        while True:
            action, _ = model.predict(cast(np.ndarray, observation), deterministic=True)
            observation, rewards, dones, infos = env.step(action)

            reward = float(np.asarray(rewards, dtype=np.float64).reshape(-1)[0])
            done = bool(np.asarray(dones, dtype=np.bool_).reshape(-1)[0])
            info = _first_info(infos)
            latest_info = info

            total_reward += reward
            episode_length += 1

            if bool(info.get("projected", False)):
                projected_actions += 1
            if bool(info.get("blocked", False)):
                blocked_actions += 1

            projected_action = info.get("projected_action")
            if isinstance(projected_action, dict):
                route = projected_action.get("route")
                mode = projected_action.get("mode")
                if route is not None:
                    route_decisions[str(route)] += 1
                if mode is not None:
                    mode_decisions[str(mode)] += 1

            if done:
                terminated = not bool(info.get("TimeLimit.truncated", False))
                truncated = bool(info.get("TimeLimit.truncated", False))
                break

        snapshot = _snapshot(latest_info)
        metrics.append(
            EpisodeMetrics(
                episode=episode_index + 1,
                seed=episode_seed,
                total_reward=total_reward,
                episode_length=episode_length,
                service_level=_float_snapshot(snapshot, "service_level"),
                safety_potential=_float_snapshot(snapshot, "network_safety_potential"),
                total_transport_cost=_float_snapshot(snapshot, "total_transport_cost"),
                disruption_score=_float_snapshot(snapshot, "disruption_score"),
                delivered_orders=_count_orders(snapshot, "delivered"),
                late_orders=_count_late_orders(snapshot),
                failed_orders=_count_failed_events(snapshot),
                projected_actions=projected_actions,
                blocked_actions=blocked_actions,
                terminated=terminated,
                truncated=truncated,
                route_decisions=route_decisions,
                mode_decisions=mode_decisions,
            )
        )
    return metrics


def aggregate(metrics: list[EpisodeMetrics]) -> AggregateMetrics:
    if not metrics:
        raise ValueError("Cannot aggregate an empty metrics list.")

    rewards = np.array([item.total_reward for item in metrics], dtype=np.float64)
    lengths = np.array([item.episode_length for item in metrics], dtype=np.float64)
    service_levels = np.array([item.service_level for item in metrics], dtype=np.float64)
    safety_values = np.array([item.safety_potential for item in metrics], dtype=np.float64)
    costs = np.array([item.total_transport_cost for item in metrics], dtype=np.float64)
    disruption_scores = np.array([item.disruption_score for item in metrics], dtype=np.float64)
    total_steps = max(int(lengths.sum()), 1)

    route_counter: Counter[str] = Counter()
    mode_counter: Counter[str] = Counter()
    for item in metrics:
        route_counter.update(item.route_decisions)
        mode_counter.update(item.mode_decisions)

    return AggregateMetrics(
        episodes=len(metrics),
        reward_mean=float(rewards.mean()),
        reward_std=float(rewards.std(ddof=0)),
        reward_min=float(rewards.min()),
        reward_max=float(rewards.max()),
        episode_length_mean=float(lengths.mean()),
        episode_length_std=float(lengths.std(ddof=0)),
        service_level_mean=float(service_levels.mean()),
        service_level_std=float(service_levels.std(ddof=0)),
        safety_potential_mean=float(safety_values.mean()),
        safety_potential_std=float(safety_values.std(ddof=0)),
        transport_cost_mean=float(costs.mean()),
        transport_cost_std=float(costs.std(ddof=0)),
        disruption_score_mean=float(disruption_scores.mean()),
        delivered_orders_mean=float(np.mean([item.delivered_orders for item in metrics])),
        late_orders_mean=float(np.mean([item.late_orders for item in metrics])),
        failed_orders_mean=float(np.mean([item.failed_orders for item in metrics])),
        projected_action_rate=sum(item.projected_actions for item in metrics) / total_steps,
        blocked_action_rate=sum(item.blocked_actions for item in metrics) / total_steps,
        termination_rate=sum(1 for item in metrics if item.terminated) / len(metrics),
        truncation_rate=sum(1 for item in metrics if item.truncated) / len(metrics),
        route_decisions=route_counter,
        mode_decisions=mode_counter,
    )


def print_report(
    *,
    model_path: Path,
    episodes: int,
    seed: int,
    aggregate_metrics: AggregateMetrics,
    episode_metrics: list[EpisodeMetrics],
) -> None:
    print("# Fine-Tuned DQN 5PL Agent Evaluation")
    print()
    print("## Run Configuration")
    print(f"- Model Path: `{model_path}`")
    print(f"- Episodes: `{episodes}`")
    print(f"- Initial Seed: `{seed}`")
    print("- Action Mode: `discrete`")
    print("- Deterministic Policy: `True`")
    print()
    print("## Aggregate Performance")
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Mean Reward | {aggregate_metrics.reward_mean:.6f} |")
    print(f"| Reward Std Dev | {aggregate_metrics.reward_std:.6f} |")
    print(f"| Min Reward | {aggregate_metrics.reward_min:.6f} |")
    print(f"| Max Reward | {aggregate_metrics.reward_max:.6f} |")
    print(f"| Mean Episode Length | {aggregate_metrics.episode_length_mean:.3f} |")
    print(f"| Episode Length Std Dev | {aggregate_metrics.episode_length_std:.3f} |")
    print(f"| Mean Service Level | {aggregate_metrics.service_level_mean:.6f} |")
    print(f"| Service Level Std Dev | {aggregate_metrics.service_level_std:.6f} |")
    print(f"| Mean Safety Potential | {aggregate_metrics.safety_potential_mean:.6f} |")
    print(f"| Safety Potential Std Dev | {aggregate_metrics.safety_potential_std:.6f} |")
    print(f"| Mean Transport Cost | {aggregate_metrics.transport_cost_mean:.6f} |")
    print(f"| Transport Cost Std Dev | {aggregate_metrics.transport_cost_std:.6f} |")
    print(f"| Mean Disruption Score | {aggregate_metrics.disruption_score_mean:.6f} |")
    print(f"| Mean Delivered Orders | {aggregate_metrics.delivered_orders_mean:.3f} |")
    print(f"| Mean Late Orders | {aggregate_metrics.late_orders_mean:.3f} |")
    print(f"| Mean Failed Orders | {aggregate_metrics.failed_orders_mean:.3f} |")
    print(f"| Projected Action Rate | {aggregate_metrics.projected_action_rate:.6f} |")
    print(f"| Blocked Action Rate | {aggregate_metrics.blocked_action_rate:.6f} |")
    print(f"| Termination Rate | {aggregate_metrics.termination_rate:.6f} |")
    print(f"| Truncation Rate | {aggregate_metrics.truncation_rate:.6f} |")
    print()
    print("## Tactical Decision Distribution")
    print("| Decision Family | Decision | Count | Share |")
    print("|---|---|---:|---:|")
    _print_counter_rows("Route", aggregate_metrics.route_decisions)
    _print_counter_rows("Mode", aggregate_metrics.mode_decisions)
    print()
    print("## Episode Sample")
    print("| Episode | Seed | Steps | Reward | Service Level | Safety Potential | Cost | Terminated | Truncated |")
    print("|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for item in episode_metrics[: min(10, len(episode_metrics))]:
        print(
            f"| {item.episode} | {item.seed} | {item.episode_length} | "
            f"{item.total_reward:.6f} | {item.service_level:.6f} | "
            f"{item.safety_potential:.6f} | {item.total_transport_cost:.6f} | "
            f"{str(item.terminated)} | {str(item.truncated)} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test the fine-tuned DQN 5PL logistics orchestration agent."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be greater than zero.")

    env = build_vec_env(args.seed, max_steps=int(args.max_steps))
    try:
        model = load_dqn_model(args.model_path, env)
        metrics = evaluate_agent(
            model,
            env,
            episodes=int(args.episodes),
            seed=int(args.seed),
        )
    finally:
        env.close()

    print_report(
        model_path=args.model_path,
        episodes=int(args.episodes),
        seed=int(args.seed),
        aggregate_metrics=aggregate(metrics),
        episode_metrics=metrics,
    )


def _first_info(infos: Any) -> dict[str, Any]:
    if isinstance(infos, (list, tuple)) and infos:
        first = infos[0]
        return dict(first) if isinstance(first, dict) else {}
    return {}


def _snapshot(info: dict[str, Any]) -> dict[str, Any]:
    value = info.get("snapshot", {})
    return dict(value) if isinstance(value, dict) else {}


def _float_snapshot(snapshot: dict[str, Any], key: str) -> float:
    value = snapshot.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0.0
    return float(value)


def _count_orders(snapshot: dict[str, Any], status: str) -> int:
    orders = snapshot.get("orders", {})
    if not isinstance(orders, dict):
        return 0
    return sum(
        1
        for order in orders.values()
        if isinstance(order, dict) and str(order.get("status", "")).lower() == status
    )


def _count_late_orders(snapshot: dict[str, Any]) -> int:
    orders = snapshot.get("orders", {})
    if not isinstance(orders, dict):
        return 0
    return sum(1 for order in orders.values() if isinstance(order, dict) and bool(order.get("is_late", False)))


def _count_failed_events(snapshot: dict[str, Any]) -> int:
    events = snapshot.get("event_log", [])
    if not isinstance(events, list):
        return 0
    return sum(
        1
        for event in events
        if isinstance(event, dict) and str(event.get("event", "")).lower() == "route_feasibility_failed"
    )


def _print_counter_rows(family: str, counter: Counter[str]) -> None:
    total = sum(counter.values())
    if total <= 0:
        print(f"| {family} | none_observed | 0 | 0.000000 |")
        return
    for decision, count in counter.most_common():
        print(f"| {family} | {decision} | {count} | {count / total:.6f} |")


if __name__ == "__main__":
    main()
