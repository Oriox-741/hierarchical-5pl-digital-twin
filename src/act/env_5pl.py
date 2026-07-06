"""Gymnasium environment wrapping the SimPy 5PL digital twin."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.act.action_projector import CONTINUOUS_ACTION_DIM, ActionProjector, PhysicalAction
from src.act.discrete_action_mapper import (
    DISCRETE_ACTION_COUNT,
    DispatchDecision,
    DiscreteActionMapper,
    DiscreteLogisticsAction,
    ModeDecision,
    ReorderDecision,
    RouteDecision,
)
from src.act.entities import Customer, Hub, Order, OrderLine, Product, RouteArc, Vehicle, VehicleTier
from src.act.inventory_dynamics import InventoryPosition
from src.act.observation_builder import OBSERVATION_DIM, ObservationBuilder
from src.act.routing_physics import make_arc_from_points
from src.act.safety_projector import SafetyProjector
from src.act.sim_engine import DigitalTwinSimulation, SimulationConfig
from src.shared.metrics import clamp
from src.shared.types import AssetKind, GeoPoint, JsonDict, ShipmentStatus
from src.think.rewards import RewardModel


ActionMode = Literal["continuous", "discrete", "joint"]
InfoMode = Literal["compact", "full"]
SimulationFactory = Callable[[int | None], DigitalTwinSimulation]
ActionInput = np.ndarray | int | Mapping[str, Any] | Sequence[Any]
DUE_SOON_HORIZON_SECONDS = 7_200.0
LATENESS_GRACE_HORIZON_SECONDS = 28_800.0


@dataclass(frozen=True, slots=True)
class OrderFlowMetrics:
    eligible_orders: int
    delivered_orders: int
    pending_orders: int
    unassigned_orders: int
    in_transit_orders: int
    healthy_in_transit_orders: int
    urgent_unassigned_orders: int
    unassigned_backlog_pressure: float
    in_transit_pressure: float
    due_soon_pressure: float
    true_lateness_pressure: float
    healthy_in_transit_ratio: float
    urgent_unassigned_ratio: float
    actionable_lateness_risk: float
    actionable_flow_pressure: float

    def as_reward_components(self) -> dict[str, float]:
        return {
            "order_flow_eligible_orders": float(self.eligible_orders),
            "order_flow_delivered_orders": float(self.delivered_orders),
            "order_flow_pending_orders": float(self.pending_orders),
            "order_flow_unassigned_orders": float(self.unassigned_orders),
            "order_flow_in_transit_orders": float(self.in_transit_orders),
            "order_flow_healthy_in_transit_orders": float(self.healthy_in_transit_orders),
            "order_flow_urgent_unassigned_orders": float(self.urgent_unassigned_orders),
            "unassigned_backlog_pressure": float(self.unassigned_backlog_pressure),
            "in_transit_pressure": float(self.in_transit_pressure),
            "due_soon_pressure": float(self.due_soon_pressure),
            "true_lateness_pressure": float(self.true_lateness_pressure),
            "healthy_in_transit_ratio": float(self.healthy_in_transit_ratio),
            "urgent_unassigned_ratio": float(self.urgent_unassigned_ratio),
            "actionable_lateness_risk": float(self.actionable_lateness_risk),
            "actionable_flow_pressure": float(self.actionable_flow_pressure),
        }


@dataclass(slots=True)
class EnvironmentConfig:
    action_mode: ActionMode = "joint"
    info_mode: InfoMode = "full"
    decision_interval: float = 300.0
    max_steps: int = 288
    random_seed: int | None = None
    service_level_target: float = 0.95
    db_snapshots: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    rolling_demand_enabled: bool = True
    rolling_demand_probability: float = 0.35
    rolling_demand_max_orders_per_step: int = 2
    rolling_demand_min_units: float = 8.0
    rolling_demand_max_units: float = 65.0
    urgent_order_probability: float = 0.25
    urgent_due_window_multiplier: float = 1.0
    transport_cost_penalty_scale: float = 150.0
    emergency_reorder_penalty: float = 0.30
    emergency_credit_weight: float = 0.30
    emergency_base_penalty: float = 0.30
    emergency_cost_weight: float = 0.30
    emergency_abuse_penalty: float = 0.30
    hold_penalty_weight: float = 0.18
    delay_weight: float = 0.22
    planned_replenishment_unit_cost: float = 0.05
    aggressive_replenishment_unit_cost: float = 0.15
    emergency_replenishment_unit_cost: float = 0.50
    primary_fleet_penalty: float = 0.10
    high_resilience_route_penalty: float = 0.08
    low_congestion_route_penalty: float = 0.03
    global_transport_cost_penalty_weight: float = 0.18
    global_backlog_reduction_weight: float = 0.10
    global_lateness_reduction_weight: float = 0.12
    speed_cost_penalty_weight: float = 0.30
    unjustified_speed_penalty_weight: float = 0.20
    speed_justification_credit_weight: float = 0.08
    planned_replenishment_cost_weight: float = 0.12
    planned_replenishment_credit_weight: float = 0.18
    excess_inventory_penalty_weight: float = 0.15
    stockout_penalty_multiplier: float = 1.0
    inventory_shortfall_penalty_multiplier: float = 1.0
    safety_stock_gap_credit_weight: float = 0.16
    capacity_opportunity_cost: float = 0.10
    reorder_neglect_weight: float = 0.18
    safety_stock_neglect_weight: float = 0.16
    capacity_neglect_weight: float = 0.10
    low_speed_lateness_weight: float = 0.12
    flow_capacity_neglect_weight: float = 0.10
    flow_enablement_weight: float = 0.08
    dispatch_progress_weight: float = 0.05
    infeasible_dispatch_penalty_weight: float = 0.04
    max_extra_dispatch_budget: int = 4
    feasibility_gated_macro_budget: bool = True
    idle_penalty_weight: float = 0.25
    speed_cost_coefficient: float = 1.25
    speed_risk_coefficient: float = 0.75


class FivePLDigitalTwinEnv(gym.Env[np.ndarray, Any]):
    """Strict Gymnasium wrapper around the SimPy digital twin."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        *,
        simulation_factory: SimulationFactory | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        if self.config.info_mode not in {"compact", "full"}:
            raise ValueError("info_mode must be 'compact' or 'full'.")
        if self.config.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if not 0.0 <= self.config.rolling_demand_probability <= 1.0:
            raise ValueError("rolling_demand_probability must be within [0, 1].")
        if self.config.rolling_demand_max_orders_per_step <= 0:
            raise ValueError("rolling_demand_max_orders_per_step must be positive.")
        if self.config.max_extra_dispatch_budget < 0:
            raise ValueError("max_extra_dispatch_budget must not be negative.")
        if self.config.rolling_demand_min_units <= 0.0:
            raise ValueError("rolling_demand_min_units must be positive.")
        if self.config.rolling_demand_max_units < self.config.rolling_demand_min_units:
            raise ValueError("rolling_demand_max_units must be >= rolling_demand_min_units.")
        if self.config.urgent_due_window_multiplier <= 0.0:
            raise ValueError("urgent_due_window_multiplier must be positive.")
        if self.config.stockout_penalty_multiplier < 0.0:
            raise ValueError("stockout_penalty_multiplier must not be negative.")
        if self.config.inventory_shortfall_penalty_multiplier < 0.0:
            raise ValueError("inventory_shortfall_penalty_multiplier must not be negative.")
        if self.config.global_transport_cost_penalty_weight < 0.0:
            raise ValueError("global_transport_cost_penalty_weight must not be negative.")
        if self.config.planned_replenishment_cost_weight < 0.0:
            raise ValueError("planned_replenishment_cost_weight must not be negative.")
        if self.config.speed_cost_coefficient < 0.0:
            raise ValueError("speed_cost_coefficient must not be negative.")
        if self.config.speed_risk_coefficient < 0.0:
            raise ValueError("speed_risk_coefficient must not be negative.")
        if self.config.reorder_neglect_weight < 0.0:
            raise ValueError("reorder_neglect_weight must not be negative.")
        if self.config.safety_stock_neglect_weight < 0.0:
            raise ValueError("safety_stock_neglect_weight must not be negative.")
        if self.config.capacity_neglect_weight < 0.0:
            raise ValueError("capacity_neglect_weight must not be negative.")
        if self.config.low_speed_lateness_weight < 0.0:
            raise ValueError("low_speed_lateness_weight must not be negative.")
        if self.config.flow_capacity_neglect_weight < 0.0:
            raise ValueError("flow_capacity_neglect_weight must not be negative.")
        if self.config.infeasible_dispatch_penalty_weight < 0.0:
            raise ValueError("infeasible_dispatch_penalty_weight must not be negative.")
        if self.config.idle_penalty_weight < 0.0:
            raise ValueError("idle_penalty_weight must not be negative.")
        for name, value in {
            "emergency_credit_weight": self.config.emergency_credit_weight,
            "emergency_base_penalty": self.config.emergency_base_penalty,
            "emergency_cost_weight": self.config.emergency_cost_weight,
            "emergency_abuse_penalty": self.config.emergency_abuse_penalty,
            "hold_penalty_weight": self.config.hold_penalty_weight,
            "delay_weight": self.config.delay_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must not be negative.")

        self.observation_builder = ObservationBuilder()
        self.action_projector = ActionProjector()
        self.discrete_mapper = DiscreteActionMapper()
        self.safety_projector = SafetyProjector()
        self.reward_model = RewardModel()
        self.simulation_factory = simulation_factory or self._default_simulation_factory
        self.scenario_stress_state = self._default_scenario_stress_state()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        if self.config.action_mode == "continuous":
            self.action_space: spaces.Space[Any] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(CONTINUOUS_ACTION_DIM,),
                dtype=np.float32,
            )
        elif self.config.action_mode == "discrete":
            self.action_space = spaces.Discrete(DISCRETE_ACTION_COUNT)
        elif self.config.action_mode == "joint":
            self.action_space = spaces.Dict(
                {
                    "continuous": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(CONTINUOUS_ACTION_DIM,),
                        dtype=np.float32,
                    ),
                    "discrete": spaces.Discrete(DISCRETE_ACTION_COUNT),
                }
            )
        else:
            raise ValueError(f"unsupported action_mode: {self.config.action_mode}")

        self.simulation: DigitalTwinSimulation = self.simulation_factory(self.config.random_seed)
        self._apply_physics_contract()
        self._step_count = 0
        self._previous_snapshot: JsonDict = self.simulation.snapshot()
        self._step_order_flow_metrics_cache: dict[tuple[int, tuple[str, ...]], OrderFlowMetrics] | None = None
        self._step_event_float_sum_cache: dict[tuple[int, int, int, float, str], dict[str, float]] | None = None
        self._previous_inaction_pressure = 0.0
        self._recent_infeasible_dispatch_attempts: list[float] = []
        self._last_dispatch_diagnostics = _empty_dispatch_diagnostics()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self.config.random_seed
        self.simulation = self.simulation_factory(effective_seed)
        self._apply_physics_contract()
        self._step_count = 0
        self._previous_snapshot = self.simulation.snapshot()
        self._step_order_flow_metrics_cache = None
        self._step_event_float_sum_cache = None
        self._previous_inaction_pressure = 0.0
        self._recent_infeasible_dispatch_attempts = []
        self._last_dispatch_diagnostics = _empty_dispatch_diagnostics()
        observation = self._observation(snapshot=self._previous_snapshot)
        return observation, self._info(projected=False, blocked=False, reasons=(), snapshot=self._previous_snapshot)

    def step(self, action: ActionInput) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        previous_snapshot = self._previous_snapshot

        projection_info = self._apply_action(action)
        self.simulation.run_until_next_decision()
        generated_order_ids = self._generate_rolling_demand()
        generated_orders = len(generated_order_ids)
        projection_info["rolling_demand_orders"] = generated_orders
        projection_info["masked_order_ids"] = tuple(str(order_id) for order_id in generated_order_ids)

        current_snapshot = self.simulation.snapshot()
        observation = self._observation(snapshot=current_snapshot)
        self._step_order_flow_metrics_cache = {}
        self._step_event_float_sum_cache = {}
        try:
            reward, reward_components = self._reward(previous_snapshot, current_snapshot, projection_info)
        finally:
            self._step_order_flow_metrics_cache = None
            self._step_event_float_sum_cache = None
        terminated = self._terminated(current_snapshot)
        truncated = self._step_count >= self.config.max_steps
        self._previous_snapshot = current_snapshot

        info = self._info(
            projected=projection_info["projected"],
            blocked=projection_info["blocked"],
            reasons=tuple(projection_info["reasons"]),
            projected_action=projection_info["action"],
            raw_action=projection_info.get("raw_action"),
            rolling_demand_orders=generated_orders,
            termination_reason=self._termination_reason(current_snapshot) if terminated else None,
            reward_components=reward_components,
            snapshot=current_snapshot,
        )
        return observation, reward, terminated, truncated, info

    def render(self) -> str:
        snapshot = self.simulation.snapshot()
        return (
            f"time={snapshot['time']:.2f} "
            f"service_level={snapshot['service_level']:.3f} "
            f"safety_potential={snapshot['network_safety_potential']:.3f} "
            f"disruption_score={snapshot['disruption_score']:.3f}"
        )

    def close(self) -> None:
        return None

    def _apply_physics_contract(self) -> None:
        """Apply economic physics knobs to the active simulation instance."""
        self.simulation.routing_physics.speed_cost_coefficient = self.config.speed_cost_coefficient
        self.simulation.routing_physics.speed_risk_coefficient = self.config.speed_risk_coefficient

    def _apply_action(self, action: ActionInput) -> dict[str, Any]:
        safety_context = self._safety_context()
        if self.config.action_mode == "continuous":
            physical = self.action_projector.project(np.asarray(action, dtype=np.float32))
            result = self.safety_projector.project_continuous(physical, **safety_context)
            if not isinstance(result.action, PhysicalAction):
                raise TypeError("continuous safety projection returned a non-physical action.")
            physical_info = self._apply_physical_action(result.action)
            return {
                "projected": result.projected,
                "blocked": result.blocked,
                "reasons": result.reasons,
                "raw_action": np.asarray(action, dtype=np.float32).reshape(-1).tolist(),
                "action": result.action.as_dict(),
                "continuous_projected": result.projected,
                "continuous_blocked": result.blocked,
                "continuous_reasons": result.reasons,
                "discrete_projected": False,
                "discrete_blocked": False,
                "discrete_reasons": (),
                "distance_cost_delta": 0.0,
                **physical_info,
            }
        if self.config.action_mode == "discrete":
            discrete = self.discrete_mapper.map(_coerce_discrete_action(action))
            result = self.safety_projector.project_discrete(discrete, **safety_context)
            if not isinstance(result.action, DiscreteLogisticsAction):
                raise TypeError("discrete safety projection returned a non-discrete action.")
            discrete_info = self._apply_discrete_action(result.action)
            return {
                "projected": result.projected,
                "blocked": result.blocked,
                "reasons": result.reasons,
                "raw_action": _coerce_discrete_action(action),
                "action": result.action.as_dict(),
                "continuous_projected": False,
                "continuous_blocked": False,
                "continuous_reasons": (),
                "discrete_projected": result.projected,
                "discrete_blocked": result.blocked,
            "discrete_reasons": result.reasons,
            **discrete_info,
            "distance_cost_delta": 0.0,
        }

        continuous_input, discrete_input = self._split_joint_action(action)
        raw_continuous = np.asarray(continuous_input, dtype=np.float32).reshape(-1)
        raw_discrete = _coerce_discrete_action(discrete_input)
        physical = self.action_projector.project(raw_continuous)
        continuous_result = self.safety_projector.project_continuous(physical, **safety_context)
        if not isinstance(continuous_result.action, PhysicalAction):
            raise TypeError("joint continuous safety projection returned a non-physical action.")
        physical_info = self._apply_physical_action(continuous_result.action, allow_dispatch=False)
        macro_dispatch_release = clamp(continuous_result.action.dispatch_intensity, 0.0, 1.0)
        macro_dispatch_budget = self._macro_dispatch_budget(macro_dispatch_release)

        aligned_safety_context = self._safety_context()
        discrete = self.discrete_mapper.map(raw_discrete)
        discrete_result = self.safety_projector.project_discrete(discrete, **aligned_safety_context)
        if not isinstance(discrete_result.action, DiscreteLogisticsAction):
            raise TypeError("joint discrete safety projection returned a non-discrete action.")
        discrete_info = self._apply_discrete_action(
            discrete_result.action,
            macro_dispatch_release=macro_dispatch_release,
            macro_dispatch_budget=macro_dispatch_budget,
        )

        reasons = tuple(f"ppo:{reason}" for reason in continuous_result.reasons) + tuple(
            f"dqn:{reason}" for reason in discrete_result.reasons
        )
        return {
            "projected": continuous_result.projected or discrete_result.projected,
            "blocked": continuous_result.blocked or discrete_result.blocked,
            "reasons": reasons,
            "raw_action": {
                "continuous": raw_continuous.tolist(),
                "discrete": raw_discrete,
            },
            "action": {
                "continuous": continuous_result.action.as_dict(),
                "discrete": discrete_result.action.as_dict(),
            },
            "pre_safety_continuous_action": physical.as_dict(),
            "continuous_projected": continuous_result.projected,
            "continuous_blocked": continuous_result.blocked,
            "continuous_reasons": continuous_result.reasons,
            "discrete_projected": discrete_result.projected,
            "discrete_blocked": discrete_result.blocked,
            "discrete_reasons": discrete_result.reasons,
            "safety_context_before_ppo": dict(safety_context),
            "safety_context_after_ppo": dict(aligned_safety_context),
            **physical_info,
            **discrete_info,
            "distance_cost_delta": 0.0,
        }

    def _split_joint_action(self, action: ActionInput) -> tuple[Any, Any]:
        if isinstance(action, Mapping):
            if "continuous" in action and "discrete" in action:
                return action["continuous"], action["discrete"]
            if "ppo" in action and "dqn" in action:
                return action["ppo"], action["dqn"]
            raise ValueError("joint action mapping must contain continuous/discrete or ppo/dqn keys.")
        if isinstance(action, Sequence) and not isinstance(action, (str, bytes, bytearray)):
            values = list(action)
            if len(values) == 2:
                return values[0], values[1]
        raise TypeError("joint action must be a mapping or two-item sequence.")

    def _macro_dispatch_budget(self, macro_dispatch_release: float) -> int:
        release = clamp(macro_dispatch_release, 0.0, 1.0)
        return 1 + int(release * self.config.max_extra_dispatch_budget)

    def _apply_physical_action(self, action: PhysicalAction, *, allow_dispatch: bool = True) -> dict[str, float]:
        replenishment_info = self._apply_hub_continuous_action(action)
        self._apply_vehicle_continuous_action(action, allow_dispatch=allow_dispatch)
        return replenishment_info

    def _apply_hub_continuous_action(self, action: PhysicalAction) -> dict[str, float]:
        """Apply continuous control-tower decisions that affect hub inventory/capacity roles."""
        for state in self.simulation.capacity_states.values():
            state.restore(action.capacity_buffer_fraction)

        ordered_units, received_units, useful_units, cost = self._apply_inventory_replenishment(
            reorder_fraction=action.reorder_fraction,
            safety_stock_multiplier=action.safety_stock_multiplier,
            receipt_fraction=0.0,
            source="ppo_planned",
            unit_cost=self.config.planned_replenishment_unit_cost,
            update_safety_stock=True,
        )
        return {
            "planned_replenishment_units": ordered_units,
            "planned_replenishment_received_units": received_units,
            "planned_replenishment_useful_units": useful_units,
            "planned_replenishment_cost": cost,
        }

    def _apply_vehicle_continuous_action(self, action: PhysicalAction, *, allow_dispatch: bool = True) -> None:
        """Apply continuous control-tower decisions that affect fleet movement roles."""
        for vehicle in self.simulation.vehicles.values():
            baseline_speed = _vehicle_baseline_speed_mps(vehicle)
            vehicle.nominal_speed_mps = max(0.1, baseline_speed * action.speed_multiplier)
            vehicle.metadata["last_speed_multiplier"] = action.speed_multiplier

        if allow_dispatch and action.dispatch_intensity > 0.0:
            self._dispatch_pending_orders(
                max_orders=max(1, round(action.dispatch_intensity * 5.0)),
                route_decision=RouteDecision.SHORTEST,
                mode_decision=ModeDecision.SECONDARY_FLEET,
            )

    def _apply_discrete_action(
        self,
        action: DiscreteLogisticsAction,
        *,
        macro_dispatch_release: float = 0.0,
        macro_dispatch_budget: int = 1,
    ) -> dict[str, float]:
        replenishment_info = self._apply_hub_discrete_action(action)
        self._last_dispatch_diagnostics = _empty_dispatch_diagnostics()
        requested_dispatch = action.dispatch == DispatchDecision.DISPATCH
        raw_dispatch_budget = max(1, int(macro_dispatch_budget))
        feasibility_metrics = self._dispatch_feasibility_metrics(action)
        feasible_dispatch_ratio = _safe_float(feasibility_metrics.get("feasible_dispatch_ratio"), 0.0)
        feasible_dispatch_cap_before_surge = 1 + int(
            clamp(feasible_dispatch_ratio, 0.0, 1.0) * self.config.max_extra_dispatch_budget
        )
        surge_metrics = self._demand_surge_dispatch_metrics(
            requested_dispatch=requested_dispatch,
            raw_dispatch_budget=raw_dispatch_budget,
            feasible_dispatch_cap=feasible_dispatch_cap_before_surge,
            feasibility_metrics=feasibility_metrics,
        )
        feasible_dispatch_cap_after_surge = int(surge_metrics["feasible_macro_dispatch_cap_after_surge"])
        dispatch_budget_before_surge = (
            min(raw_dispatch_budget, feasible_dispatch_cap_before_surge)
            if self.config.feasibility_gated_macro_budget
            else raw_dispatch_budget
        )
        dispatch_budget = dispatch_budget_before_surge
        if self.config.feasibility_gated_macro_budget:
            demand_floor = int(surge_metrics["demand_surge_macro_floor"])
            if bool(surge_metrics["demand_surge_release_active"]):
                dispatch_budget = max(
                    dispatch_budget_before_surge,
                    min(demand_floor, feasible_dispatch_cap_after_surge),
                )
            else:
                dispatch_budget = min(raw_dispatch_budget, feasible_dispatch_cap_after_surge)
        dispatched_orders = self._apply_vehicle_discrete_action(action, max_dispatch_orders=dispatch_budget)
        dispatch_diagnostics = dict(self._last_dispatch_diagnostics)
        dispatch_diagnostics.update(feasibility_metrics)
        dispatch_diagnostics.update(surge_metrics)
        dispatch_diagnostics["macro_dispatch_release"] = clamp(macro_dispatch_release, 0.0, 1.0)
        dispatch_diagnostics["raw_macro_dispatch_budget"] = float(raw_dispatch_budget)
        dispatch_diagnostics["feasible_macro_dispatch_cap"] = float(feasible_dispatch_cap_after_surge)
        dispatch_diagnostics["macro_dispatch_budget"] = float(dispatch_budget)
        dispatch_diagnostics["dqn_dispatch_requested"] = 1.0 if requested_dispatch else 0.0
        infeasible_attempt = self._infeasible_dispatch_attempt(
            requested_dispatch=requested_dispatch,
            dispatch_diagnostics=dispatch_diagnostics,
        )
        if requested_dispatch:
            self._recent_infeasible_dispatch_attempts.append(infeasible_attempt)
            self._recent_infeasible_dispatch_attempts = self._recent_infeasible_dispatch_attempts[-20:]
        dispatch_diagnostics["infeasible_dispatch_attempt"] = float(infeasible_attempt)
        dispatch_diagnostics["infeasible_dispatch_attempt_window_count"] = float(
            len(self._recent_infeasible_dispatch_attempts)
        )
        dispatch_diagnostics["repeated_infeasible_dispatch_attempt_rate"] = _mean_or_zero(
            self._recent_infeasible_dispatch_attempts
        )
        dispatch_diagnostics["dispatch_success_rate"] = clamp(
            _safe_float(dispatch_diagnostics.get("dispatch_success_count"), 0.0) / max(float(dispatch_budget), 1.0),
            0.0,
            1.0,
        )
        dispatch_diagnostics["no_vehicle_available_rate"] = clamp(
            _safe_float(dispatch_diagnostics.get("dispatch_no_vehicle_available"), 0.0)
            / max(_safe_float(dispatch_diagnostics.get("dispatchable_order_count"), 0.0), 1.0),
            0.0,
            1.0,
        )
        dispatchable_orders = _safe_float(dispatch_diagnostics.get("dispatchable_order_count"), 0.0)
        already_assigned = _safe_float(dispatch_diagnostics.get("dispatch_already_assigned_count"), 0.0)
        dispatch_diagnostics["already_assigned_rate"] = clamp(
            already_assigned / max(dispatchable_orders + already_assigned, 1.0),
            0.0,
            1.0,
        )
        return {
            **replenishment_info,
            "dqn_dispatched_orders": float(dispatched_orders),
            **dispatch_diagnostics,
        }

    def _apply_hub_discrete_action(self, action: DiscreteLogisticsAction) -> dict[str, float]:
        if action.reorder in (ReorderDecision.NONE, ReorderDecision.CONSERVATIVE):
            return {
                "dqn_replenishment_override_units": 0.0,
                "dqn_replenishment_override_received_units": 0.0,
                "dqn_replenishment_override_useful_units": 0.0,
                "dqn_replenishment_override_cost": 0.0,
            }

        unit_cost = (
            self.config.emergency_replenishment_unit_cost
            if action.reorder == ReorderDecision.EMERGENCY
            else self.config.aggressive_replenishment_unit_cost
        )
        ordered_units, received_units, useful_units, cost = self._apply_inventory_replenishment(
            reorder_fraction=self._reorder_fraction(action.reorder),
            safety_stock_multiplier=1.0,
            receipt_fraction=0.0,
            source=f"dqn_{action.reorder.name.lower()}_override",
            unit_cost=unit_cost,
            update_safety_stock=False,
        )
        return {
            "dqn_replenishment_override_units": ordered_units,
            "dqn_replenishment_override_received_units": received_units,
            "dqn_replenishment_override_useful_units": useful_units,
            "dqn_replenishment_override_cost": cost,
        }

    def _apply_vehicle_discrete_action(self, action: DiscreteLogisticsAction, *, max_dispatch_orders: int = 1) -> int:
        if action.dispatch == DispatchDecision.DISPATCH:
            return self._dispatch_pending_orders(
                max_orders=max(1, int(max_dispatch_orders)),
                route_decision=action.route,
                mode_decision=action.mode,
            )
        return 0

    def _dispatch_feasibility_metrics(self, action: DiscreteLogisticsAction) -> dict[str, float]:
        dispatchable_orders = [
            order
            for order in self.simulation.orders.values()
            if _is_dispatchable_order(order)
        ]
        preferred_tier = self._preferred_vehicle_tier(action.mode)
        minimum_required_units = (
            min((max(0.0, order.total_units) for order in dispatchable_orders), default=0.0)
            if dispatchable_orders
            else 0.0
        )
        available_vehicles = [
            vehicle
            for vehicle in self.simulation.vehicles.values()
            if vehicle.active
            and vehicle.tier == preferred_tier
            and vehicle.remaining_units >= minimum_required_units
            and vehicle.remaining_units > 0.0
        ]
        available_vehicle_count = float(len(available_vehicles))
        all_available_vehicles = [
            vehicle
            for vehicle in self.simulation.vehicles.values()
            if vehicle.active
            and vehicle.remaining_units >= minimum_required_units
            and vehicle.remaining_units > 0.0
        ]
        all_available_vehicle_count = float(len(all_available_vehicles))
        dispatchable_order_count = float(len(dispatchable_orders))
        feasible_dispatch_ratio = (
            clamp(min(available_vehicle_count, dispatchable_order_count) / max(dispatchable_order_count, 1.0), 0.0, 1.0)
            if dispatchable_order_count > 0.0
            else 0.0
        )
        vehicle_availability_pressure = (
            1.0 - clamp(available_vehicle_count / max(dispatchable_order_count, 1.0), 0.0, 1.0)
            if dispatchable_order_count > 0.0
            else 0.0
        )
        return {
            "available_vehicle_count": available_vehicle_count,
            "available_vehicle_count_all": all_available_vehicle_count,
            "dispatchable_order_count": dispatchable_order_count,
            "feasible_dispatch_opportunity": 1.0
            if available_vehicle_count > 0.0 and dispatchable_order_count > 0.0
            else 0.0,
            "feasible_dispatch_opportunity_all": 1.0
            if all_available_vehicle_count > 0.0 and dispatchable_order_count > 0.0
            else 0.0,
            "feasible_dispatch_ratio": feasible_dispatch_ratio,
            "vehicle_availability_pressure": vehicle_availability_pressure,
        }

    def _demand_surge_dispatch_metrics(
        self,
        *,
        requested_dispatch: bool,
        raw_dispatch_budget: int,
        feasible_dispatch_cap: int,
        feasibility_metrics: Mapping[str, float],
    ) -> dict[str, float]:
        dispatchable_orders = _safe_float(feasibility_metrics.get("dispatchable_order_count"), 0.0)
        available_any = _safe_float(feasibility_metrics.get("available_vehicle_count_all"), 0.0)
        available_preferred = _safe_float(feasibility_metrics.get("available_vehicle_count"), 0.0)
        physical_cap = int(min(max(available_any, 0.0), max(dispatchable_orders, 0.0)))
        hard_cap = max(1, min(1 + self.config.max_extra_dispatch_budget, physical_cap)) if physical_cap > 0 else 1
        backlog_pressure = _smoothstep_unit(
            (dispatchable_orders - max(available_any * 3.0, 3.0)) / max(available_any * 8.0, 1.0)
        )
        rolling_probability_pressure = _smoothstep_unit(
            (self.config.rolling_demand_probability - 0.35) / 0.35
        )
        rolling_volume_pressure = _smoothstep_unit(
            (float(self.config.rolling_demand_max_orders_per_step) - 2.0) / 2.0
        )
        urgent_probability_pressure = _smoothstep_unit(
            (self.config.urgent_order_probability - 0.25) / 0.50
        )
        unit_midpoint = (self.config.rolling_demand_min_units + self.config.rolling_demand_max_units) / 2.0
        unit_pressure = _smoothstep_unit((unit_midpoint - 36.5) / 61.0)
        demand_pressure = clamp(
            max(
                rolling_probability_pressure,
                rolling_volume_pressure,
                urgent_probability_pressure,
                unit_pressure,
            ),
            0.0,
            1.0,
        )
        surge_pressure = clamp(demand_pressure * backlog_pressure, 0.0, 1.0)
        demand_surge_macro_floor = 1.0
        demand_surge_release_active = 0.0
        feasible_cap_after_surge = float(feasible_dispatch_cap)
        if (
            requested_dispatch
            and surge_pressure >= 0.35
            and available_any > 1.0
            and dispatchable_orders > available_any
        ):
            demand_surge_macro_floor = float(min(hard_cap, max(2, int(round(available_any)))))
            feasible_cap_after_surge = float(max(feasible_dispatch_cap, int(demand_surge_macro_floor)))
            demand_surge_release_active = 1.0 if feasible_cap_after_surge > feasible_dispatch_cap else 0.0

        return {
            "demand_surge_release_active": float(demand_surge_release_active),
            "demand_surge_macro_floor": float(demand_surge_macro_floor),
            "backlog_pressure": float(backlog_pressure),
            "demand_surge_pressure": float(surge_pressure),
            "macro_dispatch_budget_before_surge": float(
                min(raw_dispatch_budget, feasible_dispatch_cap)
                if self.config.feasibility_gated_macro_budget
                else raw_dispatch_budget
            ),
            "macro_dispatch_budget_after_surge": float(
                max(
                    min(raw_dispatch_budget, feasible_dispatch_cap),
                    min(int(demand_surge_macro_floor), int(feasible_cap_after_surge)),
                )
                if self.config.feasibility_gated_macro_budget and demand_surge_release_active
                else (
                    min(raw_dispatch_budget, int(feasible_cap_after_surge))
                    if self.config.feasibility_gated_macro_budget
                    else raw_dispatch_budget
                )
            ),
            "feasible_macro_dispatch_cap_before_surge": float(feasible_dispatch_cap),
            "feasible_macro_dispatch_cap_after_surge": float(feasible_cap_after_surge),
            "demand_surge_available_vehicle_count": float(available_any),
            "demand_surge_preferred_vehicle_count": float(available_preferred),
        }

    @staticmethod
    def _infeasible_dispatch_attempt(
        *,
        requested_dispatch: bool,
        dispatch_diagnostics: Mapping[str, float],
    ) -> float:
        if not requested_dispatch:
            return 0.0
        if _safe_float(dispatch_diagnostics.get("dispatch_success_count"), 0.0) > 0.0:
            return 0.0
        if _safe_float(dispatch_diagnostics.get("dispatchable_order_count"), 0.0) <= 0.0:
            return 0.0
        failure_count = (
            _safe_float(dispatch_diagnostics.get("dispatch_no_vehicle_available"), 0.0)
            + _safe_float(dispatch_diagnostics.get("dispatch_already_assigned_count"), 0.0)
            + _safe_float(dispatch_diagnostics.get("dispatch_route_failure"), 0.0)
            + _safe_float(dispatch_diagnostics.get("dispatch_capacity_rejected"), 0.0)
        )
        return 1.0 if failure_count > 0.0 else 0.0

    def _apply_inventory_replenishment(
        self,
        *,
        reorder_fraction: float,
        safety_stock_multiplier: float,
        receipt_fraction: float,
        source: str,
        unit_cost: float,
        update_safety_stock: bool,
    ) -> tuple[float, float, float, float]:
        total_ordered_units = 0.0
        total_received_units = 0.0
        total_useful_units = 0.0
        for position in self.simulation.inventory_network.positions.values():
            if update_safety_stock:
                target = position.target_safety_stock() * safety_stock_multiplier
                position.safety_stock_units = max(position.safety_stock_units, target)
            reorder_units = position.reorder_quantity() * reorder_fraction
            if reorder_units > 0.0:
                inventory_gap = max(
                    0.0,
                    position.backlog_units
                    + position.reserved_units
                    + position.safety_stock_units
                    - position.available_units,
                )
                received_units = reorder_units * receipt_fraction
                in_transit_units = max(0.0, reorder_units - received_units)
                position.on_hand_units += received_units
                position.in_transit_units += in_transit_units
                total_ordered_units += reorder_units
                total_received_units += received_units
                total_useful_units += min(reorder_units, inventory_gap)
        total_cost = total_ordered_units * unit_cost
        if total_ordered_units > 0.0:
            self.simulation.state.total_transport_cost += total_cost
            self.simulation.state.event_log.append(
                {
                    "time": self.simulation.env.now,
                    "event": "inventory_replenishment",
                    "source": source,
                    "ordered_units": total_ordered_units,
                    "received_units": total_received_units,
                    "in_transit_units": max(0.0, total_ordered_units - total_received_units),
                    "useful_units": total_useful_units,
                    "unit_cost": unit_cost,
                    "cost": total_cost,
                }
            )
        return total_ordered_units, total_received_units, total_useful_units, total_cost

    @staticmethod
    def _reorder_fraction(reorder: ReorderDecision) -> float:
        return {
            ReorderDecision.NONE: 0.0,
            ReorderDecision.CONSERVATIVE: 0.25,
            ReorderDecision.AGGRESSIVE: 0.60,
            ReorderDecision.EMERGENCY: 1.0,
        }[reorder]

    def _dispatch_pending_orders(
        self,
        *,
        max_orders: int,
        route_decision: RouteDecision,
        mode_decision: ModeDecision,
    ) -> int:
        dispatched = 0
        reserved_vehicle_ids: set[UUID] = set()
        preferred_tier = self._preferred_vehicle_tier(mode_decision)
        diagnostics = _empty_dispatch_diagnostics()
        diagnostics["dispatch_attempted"] = 1.0 if max_orders > 0 else 0.0
        eligible_unassigned_orders = [
            order
            for order in self.simulation.orders.values()
            if _is_dispatchable_order(order)
        ]
        if not eligible_unassigned_orders:
            diagnostics["dispatch_no_unassigned_orders"] = 1.0
        diagnostics["dispatch_already_assigned_count"] = float(
            sum(
                1
                for order in self.simulation.orders.values()
                if order.assigned_vehicle_id is not None and order.delivery_time is None
            )
        )
        for order in self.simulation.orders.values():
            if not _is_dispatchable_order(order):
                continue
            vehicle = self._select_vehicle(
                order.total_units,
                excluded_vehicle_ids=reserved_vehicle_ids,
                preferred_tier=preferred_tier,
            )
            if vehicle is None:
                diagnostics["dispatch_no_vehicle_available"] += 1.0
                continue
            try:
                path = self._select_route_path(
                    order.origin_node_id,
                    order.destination_node_id,
                    route_decision=route_decision,
                )
                feasibility = self.simulation.evaluate_order_route(order, vehicle, path, start_time=self.simulation.env.now)
                if not feasibility.feasible:
                    order.status = ShipmentStatus.DELAYED
                    order.assigned_vehicle_id = None
                    order.pickup_time = None
                    diagnostics["dispatch_route_failure"] += 1.0
                    diagnostics["route_feasibility_failed_count"] += 1.0
                    diagnostics["failed_route_vehicle_rollback_count"] += 1.0
                    diagnostics["customer_revisit_blocked_count"] += float(
                        any(str(violation).startswith("customer_revisited") for violation in feasibility.violations)
                    )
                    self.simulation.state.failed_orders += 1
                    self.simulation.state.event_log.append(
                        {
                            "time": self.simulation.env.now,
                            "event": "route_feasibility_failed",
                            "order_id": str(order.order_id),
                            "vehicle_id": str(vehicle.vehicle_id),
                            "violations": list(feasibility.violations),
                            "failed_route_vehicle_rollback_count": 1,
                        }
                    )
                    continue
                reserved_vehicle_ids.add(vehicle.vehicle_id)
                order.metadata["route_decision"] = route_decision.name.lower()
                order.metadata["mode_decision"] = mode_decision.name.lower()
                vehicle.metadata["last_mode_decision"] = mode_decision.name.lower()
                vehicle.metadata["last_route_decision"] = route_decision.name.lower()
                self.simulation.state.event_log.append(
                    {
                        "time": self.simulation.env.now,
                        "event": "dispatch_decision",
                        "order_id": str(order.order_id),
                        "vehicle_id": str(vehicle.vehicle_id),
                        "vehicle_tier": vehicle.tier.value,
                        "route_decision": route_decision.name.lower(),
                        "mode_decision": mode_decision.name.lower(),
                        "path": [str(node_id) for node_id in path],
                    }
                )
                self.simulation.dispatch_order(order.order_id, vehicle.vehicle_id, path)
                dispatched += 1
            except (KeyError, ValueError):
                diagnostics["dispatch_route_failure"] += 1.0
                continue
            if dispatched >= max_orders:
                break
        diagnostics["dispatch_success_count"] = float(dispatched)
        self._last_dispatch_diagnostics = diagnostics
        return dispatched

    def _select_vehicle(
        self,
        required_units: float,
        *,
        excluded_vehicle_ids: set[UUID] | None = None,
        preferred_tier: VehicleTier | None = None,
    ) -> Vehicle | None:
        excluded = excluded_vehicle_ids or set()
        candidates = [
            vehicle
            for vehicle in self.simulation.vehicles.values()
            if vehicle.active
            and vehicle.vehicle_id not in excluded
            and vehicle.remaining_units >= required_units
        ]
        if preferred_tier is not None:
            preferred = [vehicle for vehicle in candidates if vehicle.tier == preferred_tier]
            if preferred:
                return max(preferred, key=lambda vehicle: vehicle.remaining_units)
        return max(candidates, key=lambda vehicle: vehicle.remaining_units, default=None)

    @staticmethod
    def _preferred_vehicle_tier(mode_decision: ModeDecision) -> VehicleTier:
        if mode_decision == ModeDecision.PRIMARY_FLEET:
            return VehicleTier.PRIMARY
        return VehicleTier.SECONDARY

    def _select_route_path(
        self,
        origin_node_id: UUID,
        destination_node_id: UUID,
        *,
        route_decision: RouteDecision,
    ) -> list[UUID]:
        if route_decision == RouteDecision.SHORTEST:
            return self.simulation.route_network.shortest_path(origin_node_id, destination_node_id)
        return self._weighted_route_path(origin_node_id, destination_node_id, route_decision=route_decision)

    def _weighted_route_path(
        self,
        origin_node_id: UUID,
        destination_node_id: UUID,
        *,
        route_decision: RouteDecision,
    ) -> list[UUID]:
        if origin_node_id == destination_node_id:
            return [origin_node_id]

        adjacency: dict[UUID, set[UUID]] = {}
        for origin, destination in self.simulation.route_network.arcs:
            adjacency.setdefault(origin, set()).add(destination)
            adjacency.setdefault(destination, set())

        distances: dict[UUID, float] = {origin_node_id: 0.0}
        previous: dict[UUID, UUID] = {}
        unvisited = set(adjacency)

        while unvisited:
            current = min(unvisited, key=lambda node: distances.get(node, float("inf")))
            current_distance = distances.get(current, float("inf"))
            if current == destination_node_id or current_distance == float("inf"):
                break
            unvisited.remove(current)

            for neighbor in adjacency.get(current, ()):
                arc = self.simulation.route_network.get_arc(current, neighbor)
                if not arc.active:
                    continue
                candidate = current_distance + self._route_decision_cost(arc, route_decision=route_decision)
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = current

        if destination_node_id not in distances:
            raise ValueError(f"no active path from {origin_node_id} to {destination_node_id}")

        path = [destination_node_id]
        while path[-1] != origin_node_id:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def _route_decision_cost(self, arc: RouteArc, *, route_decision: RouteDecision) -> float:
        key = (arc.origin_node_id, arc.destination_node_id)
        congestion = clamp(self.simulation.routing_physics.congestion_by_arc.get(key, 0.0), 0.0, 1.0)
        delay_multiplier = max(1.0, self.simulation.routing_physics.arc_delay_multiplier.get(key, 1.0))
        if route_decision == RouteDecision.LOW_CONGESTION:
            return arc.distance_m * (1.0 + (4.0 * congestion) + (0.25 * (delay_multiplier - 1.0)))
        if route_decision == RouteDecision.HIGH_RESILIENCE:
            risk_score = float(arc.metadata.get("risk_score", 0.0))
            capacity_penalty = 0.0 if arc.capacity_units == float("inf") else 1.0 / max(arc.capacity_units, 1.0)
            return arc.distance_m * (
                1.0
                + (2.0 * congestion)
                + (1.5 * (delay_multiplier - 1.0))
                + (3.0 * clamp(risk_score, 0.0, 1.0))
                + capacity_penalty
            )
        return arc.distance_m

    def _observation(self, *, snapshot: Mapping[str, Any] | None = None) -> np.ndarray:
        return self.observation_builder.build(
            self.simulation,
            snapshot=snapshot,
            db_snapshots=self.config.db_snapshots,
            scenario_stress_state=self.scenario_stress_state,
        ).vector

    def _default_scenario_stress_state(self) -> dict[str, float]:
        """Metadata-only scenario stress signals for observation features."""

        return {
            "holding_cost_multiplier": 1.0,
            "excess_inventory_penalty_multiplier": 1.0,
            "planned_replenishment_cost_multiplier": 1.0,
            "stockout_penalty_multiplier": float(self.config.stockout_penalty_multiplier),
            "inventory_shortfall_penalty_multiplier": float(self.config.inventory_shortfall_penalty_multiplier),
            "vehicle_availability_multiplier": 1.0,
            "fleet_capacity_multiplier": 1.0,
            "capacity_shock_severity": 0.0,
            "route_disruption_probability": 0.0,
            "congestion_multiplier": 1.0,
            "shortest_route_risk_multiplier": 1.0,
            "traversal_cost_multiplier": 1.0,
            "disruption_risk_multiplier": 1.0,
            "route_fleet_cost_multiplier": 1.0,
            "premium_sla_ratio": 0.0,
            "service_target": float(self.config.service_level_target),
            "urgent_due_window_multiplier": float(self.config.urgent_due_window_multiplier),
            "premium_lateness_penalty_multiplier": 1.0,
            "supplier_delay_probability": 0.0,
            "lead_time_mean_multiplier": 1.0,
            "lead_time_variance_multiplier": 1.0,
        }

    def _reward(
        self,
        previous: JsonDict,
        current: JsonDict,
        projection_info: Mapping[str, Any],
    ) -> tuple[float, dict[str, float]]:
        base_reward, base_components = self.reward_model.from_snapshots(
            previous,
            current,
            projected=bool(projection_info["projected"]),
            blocked=bool(projection_info["blocked"]),
        )
        masked_order_ids = _string_set(projection_info.get("masked_order_ids", ()))
        delivered_delta = max(0, _delivered_count(current) - _delivered_count(previous))
        cost_delta = max(
            0.0,
            float(current.get("total_transport_cost", 0.0)) - float(previous.get("total_transport_cost", 0.0)),
        )
        speed_cost_delta = self._event_float_sum_since(
            current,
            after_time=float(previous.get("time", 0.0)),
            event_name="route_arc_traversed",
            key="speed_cost",
        )
        distance_cost_delta = self._event_float_sum_since(
            current,
            after_time=float(previous.get("time", 0.0)),
            event_name="route_arc_traversed",
            key="distance_cost",
        )
        planned_replenishment_cost = _safe_float(projection_info.get("planned_replenishment_cost"), 0.0)
        dqn_replenishment_override_cost = _safe_float(
            projection_info.get("dqn_replenishment_override_cost"),
            0.0,
        )
        ppo_cost_delta = max(0.0, cost_delta - dqn_replenishment_override_cost)
        dqn_cost_delta = max(0.0, cost_delta - planned_replenishment_cost)
        service_gap = max(0.0, self.config.service_level_target - float(current["service_level"]))
        previous_order_flow_metrics = self._order_flow_metrics(previous, masked_order_ids=set())
        order_flow_metrics = self._order_flow_metrics(current, masked_order_ids=masked_order_ids)
        true_backlog_reduction, lateness_reduction, global_flow_credit = self._global_flow_progress_credit(
            previous_order_flow_metrics,
            order_flow_metrics,
        )
        global_reward = self._global_reward(
            current,
            delivered_delta=delivered_delta,
            service_gap=service_gap,
            masked_order_ids=masked_order_ids,
            cost_delta=cost_delta,
            global_flow_credit=global_flow_credit,
        )
        global_raw = self._global_reward_raw(
            current,
            delivered_delta=delivered_delta,
            service_gap=service_gap,
            masked_order_ids=masked_order_ids,
            cost_delta=cost_delta,
            global_flow_credit=global_flow_credit,
        )
        pending_work_pressure = self._pending_work_pressure(current, masked_order_ids=masked_order_ids)
        raw_pending_work_pressure = self._raw_pending_work_pressure(current, masked_order_ids=masked_order_ids)
        raw_lateness_risk = self._lateness_risk(current)
        global_idle_penalty = self._global_idle_penalty(
            pending_work_pressure=pending_work_pressure,
            delivered_delta=delivered_delta,
        )
        ppo_components = self._ppo_local_reward_components(
            current,
            projection_info,
            cost_delta=ppo_cost_delta,
            speed_cost_delta=speed_cost_delta,
            distance_cost_delta=distance_cost_delta,
        )
        ppo_local_reward = ppo_components["ppo_local"]
        dqn_components = self._dqn_local_reward_components(
            current,
            projection_info,
            delivered_delta=delivered_delta,
            cost_delta=dqn_cost_delta,
            masked_order_ids=masked_order_ids,
        )
        self._previous_inaction_pressure = dqn_components.get("current_inaction_pressure", 0.0)
        dqn_local_reward = dqn_components["dqn_local"]
        projection_penalty = 0.03 if bool(projection_info["projected"]) else 0.0
        blocked_penalty = 0.20 if bool(projection_info["blocked"]) else 0.0
        total = global_reward + ppo_local_reward + dqn_local_reward - projection_penalty - blocked_penalty
        return float(total), {
            "total": float(total),
            "global": float(global_reward),
            "global_raw": float(global_raw),
            "true_backlog_reduction": float(true_backlog_reduction),
            "lateness_reduction": float(lateness_reduction),
            "global_flow_credit": float(global_flow_credit),
            "pending_work_pressure": float(pending_work_pressure),
            "raw_pending_work_pressure": float(raw_pending_work_pressure),
            "raw_lateness_risk": float(raw_lateness_risk),
            "global_idle_penalty": float(global_idle_penalty),
            "ppo_local": float(ppo_local_reward),
            "dqn_local": float(dqn_local_reward),
            "base_reward_model": float(base_reward),
            "service_level": float(current.get("service_level", 0.0)),
            "network_safety_potential": float(current.get("network_safety_potential", 0.0)),
            "delivered_delta": float(delivered_delta),
            "transport_cost_delta": float(cost_delta),
            "distance_cost_delta": float(distance_cost_delta),
            "speed_cost_delta": float(speed_cost_delta),
            "ppo_cost_delta": float(ppo_cost_delta),
            "dqn_cost_delta": float(dqn_cost_delta),
            "planned_replenishment_units": _safe_float(projection_info.get("planned_replenishment_units"), 0.0),
            "planned_replenishment_received_units": _safe_float(
                projection_info.get("planned_replenishment_received_units"),
                0.0,
            ),
            "planned_replenishment_useful_units": _safe_float(
                projection_info.get("planned_replenishment_useful_units"),
                0.0,
            ),
            "planned_replenishment_cost": float(planned_replenishment_cost),
            "dqn_replenishment_override_units": _safe_float(
                projection_info.get("dqn_replenishment_override_units"),
                0.0,
            ),
            "dqn_replenishment_override_received_units": _safe_float(
                projection_info.get("dqn_replenishment_override_received_units"),
                0.0,
            ),
            "dqn_replenishment_override_useful_units": _safe_float(
                projection_info.get("dqn_replenishment_override_useful_units"),
                0.0,
            ),
            "dqn_replenishment_override_cost": float(dqn_replenishment_override_cost),
            "dqn_dispatched_orders": _safe_float(projection_info.get("dqn_dispatched_orders"), 0.0),
            "dispatch_attempted": _safe_float(projection_info.get("dispatch_attempted"), 0.0),
            "dispatch_success_count": _safe_float(projection_info.get("dispatch_success_count"), 0.0),
            "dispatch_no_unassigned_orders": _safe_float(projection_info.get("dispatch_no_unassigned_orders"), 0.0),
            "dispatch_no_vehicle_available": _safe_float(projection_info.get("dispatch_no_vehicle_available"), 0.0),
            "dispatch_route_failure": _safe_float(projection_info.get("dispatch_route_failure"), 0.0),
            "dispatch_capacity_rejected": _safe_float(projection_info.get("dispatch_capacity_rejected"), 0.0),
            "dispatch_already_assigned_count": _safe_float(
                projection_info.get("dispatch_already_assigned_count"),
                0.0,
            ),
            "available_vehicle_count": _safe_float(projection_info.get("available_vehicle_count"), 0.0),
            "available_vehicle_count_all": _safe_float(projection_info.get("available_vehicle_count_all"), 0.0),
            "dispatchable_order_count": _safe_float(projection_info.get("dispatchable_order_count"), 0.0),
            "feasible_dispatch_opportunity": _safe_float(
                projection_info.get("feasible_dispatch_opportunity"),
                0.0,
            ),
            "feasible_dispatch_opportunity_all": _safe_float(
                projection_info.get("feasible_dispatch_opportunity_all"),
                0.0,
            ),
            "feasible_dispatch_ratio": _safe_float(projection_info.get("feasible_dispatch_ratio"), 0.0),
            "vehicle_availability_pressure": _safe_float(
                projection_info.get("vehicle_availability_pressure"),
                0.0,
            ),
            "dispatch_feasibility_pressure": float(
                order_flow_metrics.actionable_flow_pressure
                * (1.0 - _safe_float(projection_info.get("feasible_dispatch_ratio"), 0.0))
            ),
            "infeasible_dispatch_attempt": _safe_float(
                projection_info.get("infeasible_dispatch_attempt"),
                0.0,
            ),
            "infeasible_dispatch_attempt_window_count": _safe_float(
                projection_info.get("infeasible_dispatch_attempt_window_count"),
                0.0,
            ),
            "repeated_infeasible_dispatch_attempt_rate": _safe_float(
                projection_info.get("repeated_infeasible_dispatch_attempt_rate"),
                0.0,
            ),
            "dispatch_success_rate": _safe_float(projection_info.get("dispatch_success_rate"), 0.0),
            "no_vehicle_available_rate": _safe_float(projection_info.get("no_vehicle_available_rate"), 0.0),
            "already_assigned_rate": _safe_float(projection_info.get("already_assigned_rate"), 0.0),
            "macro_dispatch_release": _safe_float(projection_info.get("macro_dispatch_release"), 0.0),
            "raw_macro_dispatch_budget": _safe_float(projection_info.get("raw_macro_dispatch_budget"), 1.0),
            "feasible_macro_dispatch_cap": _safe_float(projection_info.get("feasible_macro_dispatch_cap"), 1.0),
            "macro_dispatch_budget": _safe_float(projection_info.get("macro_dispatch_budget"), 1.0),
            "demand_surge_release_active": _safe_float(
                projection_info.get("demand_surge_release_active"),
                0.0,
            ),
            "demand_surge_macro_floor": _safe_float(projection_info.get("demand_surge_macro_floor"), 1.0),
            "backlog_pressure": _safe_float(projection_info.get("backlog_pressure"), 0.0),
            "demand_surge_pressure": _safe_float(projection_info.get("demand_surge_pressure"), 0.0),
            "macro_dispatch_budget_before_surge": _safe_float(
                projection_info.get("macro_dispatch_budget_before_surge"),
                1.0,
            ),
            "macro_dispatch_budget_after_surge": _safe_float(
                projection_info.get("macro_dispatch_budget_after_surge"),
                1.0,
            ),
            "feasible_macro_dispatch_cap_before_surge": _safe_float(
                projection_info.get("feasible_macro_dispatch_cap_before_surge"),
                1.0,
            ),
            "feasible_macro_dispatch_cap_after_surge": _safe_float(
                projection_info.get("feasible_macro_dispatch_cap_after_surge"),
                1.0,
            ),
            "dqn_dispatch_requested": _safe_float(projection_info.get("dqn_dispatch_requested"), 0.0),
            "pending_ratio_masked": self._pending_order_ratio(current, masked_order_ids=masked_order_ids),
            "projection_penalty": projection_penalty,
            "blocked_penalty": blocked_penalty,
            **ppo_components,
            **dqn_components,
            **order_flow_metrics.as_reward_components(),
            **{f"base_{name}": value for name, value in base_components.as_dict().items()},
        }

    def _terminated(self, snapshot: JsonDict) -> bool:
        catastrophic_risk = float(snapshot["network_safety_potential"]) >= 0.995
        return catastrophic_risk

    def _termination_reason(self, snapshot: JsonDict) -> str | None:
        if float(snapshot["network_safety_potential"]) >= 0.995:
            return "catastrophic_safety_potential"
        return None

    def _global_reward(
        self,
        current: Mapping[str, Any],
        *,
        delivered_delta: int,
        service_gap: float,
        masked_order_ids: set[str],
        cost_delta: float = 0.0,
        global_flow_credit: float = 0.0,
    ) -> float:
        raw = self._global_reward_raw(
            current,
            delivered_delta=delivered_delta,
            service_gap=service_gap,
            masked_order_ids=masked_order_ids,
            cost_delta=cost_delta,
            global_flow_credit=global_flow_credit,
        )
        pending_work_pressure = self._pending_work_pressure(current, masked_order_ids=masked_order_ids)
        idle_penalty = self._global_idle_penalty(
            pending_work_pressure=pending_work_pressure,
            delivered_delta=delivered_delta,
        )
        adjusted = raw - idle_penalty
        if delivered_delta == 0 and pending_work_pressure > 0.60:
            adjusted = min(adjusted, 0.45)
        return float(adjusted)

    def _global_reward_raw(
        self,
        current: Mapping[str, Any],
        *,
        delivered_delta: int,
        service_gap: float,
        masked_order_ids: set[str],
        cost_delta: float = 0.0,
        global_flow_credit: float = 0.0,
    ) -> float:
        service = clamp(float(current.get("service_level", 0.0)), 0.0, 1.0)
        safety_potential_value = clamp(float(current.get("network_safety_potential", 0.0)), 0.0, 1.0)
        disruption = clamp(float(current.get("disruption_score", 0.0)), 0.0, 1.0)
        pending_ratio = self._pending_work_pressure(current, masked_order_ids=masked_order_ids)
        cost_penalty = clamp(max(0.0, cost_delta) / self.config.transport_cost_penalty_scale, 0.0, 2.0)
        return float(
            (0.60 * service)
            + (0.12 * min(delivered_delta, 5))
            + (0.20 * (1.0 - safety_potential_value))
            + (0.08 * (1.0 - disruption))
            - (0.75 * service_gap)
            - (0.20 * pending_ratio)
            - (self.config.global_transport_cost_penalty_weight * cost_penalty)
            + global_flow_credit
        )

    def _global_flow_progress_credit(
        self,
        previous_metrics: OrderFlowMetrics,
        current_metrics: OrderFlowMetrics,
    ) -> tuple[float, float, float]:
        true_backlog_reduction = max(
            0.0,
            previous_metrics.unassigned_backlog_pressure - current_metrics.unassigned_backlog_pressure,
        )
        lateness_reduction = max(
            0.0,
            previous_metrics.true_lateness_pressure - current_metrics.true_lateness_pressure,
        )
        credit = (
            self.config.global_backlog_reduction_weight * true_backlog_reduction
            + self.config.global_lateness_reduction_weight * lateness_reduction
        )
        return float(true_backlog_reduction), float(lateness_reduction), float(credit)

    def _pending_work_pressure(
        self,
        current: Mapping[str, Any],
        *,
        masked_order_ids: set[str],
    ) -> float:
        return self._order_flow_metrics(current, masked_order_ids=masked_order_ids).actionable_flow_pressure

    def _raw_pending_work_pressure(
        self,
        current: Mapping[str, Any],
        *,
        masked_order_ids: set[str],
    ) -> float:
        return clamp(
            max(
                self._pending_order_ratio(current, masked_order_ids=masked_order_ids),
                self._backlog_age_pressure(current, masked_order_ids=masked_order_ids),
                self._urgent_pending_ratio(current),
                self._lateness_risk(current),
            ),
            0.0,
            1.0,
        )

    def _global_idle_penalty(self, *, pending_work_pressure: float, delivered_delta: int) -> float:
        no_delivery_indicator = 1.0 if delivered_delta == 0 else 0.0
        idle_gate = _smoothstep_unit((pending_work_pressure - 0.20) / 0.60)
        return float(self.config.idle_penalty_weight * no_delivery_indicator * idle_gate)

    def _event_float_sum_since(
        self,
        snapshot: Mapping[str, Any],
        *,
        after_time: float,
        event_name: str,
        key: str,
    ) -> float:
        events = snapshot.get("event_log", ())
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
            return 0.0
        cache = self._step_event_float_sum_cache
        cache_key = (id(snapshot), id(events), len(events), float(after_time), str(event_name))
        if cache is None:
            return _event_float_sum_since(
                snapshot,
                after_time=after_time,
                event_name=event_name,
                key=key,
            )
        sums = cache.get(cache_key)
        if sums is None:
            sums = _event_float_sums_since(events, after_time=float(after_time), event_name=str(event_name))
            cache[cache_key] = sums
        return float(sums.get(str(key), 0.0))

    def _order_flow_metrics(
        self,
        snapshot: Mapping[str, Any],
        *,
        masked_order_ids: set[str] | None = None,
    ) -> OrderFlowMetrics:
        excluded = masked_order_ids or set()
        cache = self._step_order_flow_metrics_cache
        cache_key = (id(snapshot), tuple(sorted(str(order_id) for order_id in excluded)))
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        orders = snapshot.get("orders", {})
        current_time = _safe_float(snapshot.get("time"), 0.0)
        if not isinstance(orders, Mapping) or not orders:
            metrics = _empty_order_flow_metrics()
            if cache is not None:
                cache[cache_key] = metrics
            return metrics

        eligible_count = 0
        delivered_count = 0
        pending_count = 0
        unassigned_count = 0
        in_transit_count = 0
        healthy_in_transit_count = 0
        urgent_unassigned_count = 0
        due_soon_values: list[float] = []
        late_values: list[float] = []

        for order_id, order in orders.items():
            if str(order_id) in excluded or not isinstance(order, Mapping):
                continue
            eligible_count += 1
            status = str(order.get("status", ""))
            delivery_time = order.get("delivery_time")
            is_delivered = delivery_time is not None or status == ShipmentStatus.DELIVERED.value
            if is_delivered:
                delivered_count += 1
                continue

            pending_count += 1
            assigned_vehicle_id = order.get("assigned_vehicle_id")
            due_time = _safe_float(order.get("due_time"), current_time)
            remaining = due_time - current_time
            if assigned_vehicle_id is None:
                unassigned_count += 1
                if remaining <= DUE_SOON_HORIZON_SECONDS:
                    urgent_unassigned_count += 1
            else:
                in_transit_count += 1
                if remaining >= 0.0:
                    healthy_in_transit_count += 1

            if remaining > 0.0:
                due_soon_values.append(clamp(1.0 - (remaining / DUE_SOON_HORIZON_SECONDS), 0.0, 1.0))
            else:
                late_values.append(clamp((current_time - due_time) / LATENESS_GRACE_HORIZON_SECONDS, 0.0, 1.0))

        unassigned_backlog_pressure = clamp(unassigned_count / max(eligible_count, 1), 0.0, 1.0)
        in_transit_pressure = clamp(in_transit_count / max(eligible_count, 1), 0.0, 1.0)
        due_soon_pressure = _mean_or_zero(due_soon_values)
        true_lateness_pressure = _mean_or_zero(late_values)
        healthy_in_transit_ratio = clamp(healthy_in_transit_count / max(pending_count, 1), 0.0, 1.0)
        urgent_unassigned_ratio = clamp(urgent_unassigned_count / max(unassigned_count, 1), 0.0, 1.0)
        actionable_lateness_risk = clamp(
            max(true_lateness_pressure, due_soon_pressure * (1.0 - healthy_in_transit_ratio)),
            0.0,
            1.0,
        )
        actionable_flow_pressure = clamp(
            max(unassigned_backlog_pressure, actionable_lateness_risk, urgent_unassigned_ratio),
            0.0,
            1.0,
        )
        metrics = OrderFlowMetrics(
            eligible_orders=eligible_count,
            delivered_orders=delivered_count,
            pending_orders=pending_count,
            unassigned_orders=unassigned_count,
            in_transit_orders=in_transit_count,
            healthy_in_transit_orders=healthy_in_transit_count,
            urgent_unassigned_orders=urgent_unassigned_count,
            unassigned_backlog_pressure=unassigned_backlog_pressure,
            in_transit_pressure=in_transit_pressure,
            due_soon_pressure=due_soon_pressure,
            true_lateness_pressure=true_lateness_pressure,
            healthy_in_transit_ratio=healthy_in_transit_ratio,
            urgent_unassigned_ratio=urgent_unassigned_ratio,
            actionable_lateness_risk=actionable_lateness_risk,
            actionable_flow_pressure=actionable_flow_pressure,
        )
        if cache is not None:
            cache[cache_key] = metrics
        return metrics

    def _unassigned_backlog_pressure(
        self,
        snapshot: Mapping[str, Any],
        *,
        masked_order_ids: set[str] | None = None,
    ) -> float:
        return self._order_flow_metrics(snapshot, masked_order_ids=masked_order_ids).unassigned_backlog_pressure

    def _in_transit_pressure(
        self,
        snapshot: Mapping[str, Any],
        *,
        masked_order_ids: set[str] | None = None,
    ) -> float:
        return self._order_flow_metrics(snapshot, masked_order_ids=masked_order_ids).in_transit_pressure

    def _due_soon_pressure(
        self,
        snapshot: Mapping[str, Any],
        *,
        masked_order_ids: set[str] | None = None,
    ) -> float:
        return self._order_flow_metrics(snapshot, masked_order_ids=masked_order_ids).due_soon_pressure

    def _true_lateness_pressure(
        self,
        snapshot: Mapping[str, Any],
        *,
        masked_order_ids: set[str] | None = None,
    ) -> float:
        return self._order_flow_metrics(snapshot, masked_order_ids=masked_order_ids).true_lateness_pressure

    def _healthy_in_transit_ratio(
        self,
        snapshot: Mapping[str, Any],
        *,
        masked_order_ids: set[str] | None = None,
    ) -> float:
        return self._order_flow_metrics(snapshot, masked_order_ids=masked_order_ids).healthy_in_transit_ratio

    def _actionable_lateness_risk(
        self,
        snapshot: Mapping[str, Any],
        *,
        masked_order_ids: set[str] | None = None,
    ) -> float:
        return self._order_flow_metrics(snapshot, masked_order_ids=masked_order_ids).actionable_lateness_risk

    def _ppo_local_reward(
        self,
        current: Mapping[str, Any],
        projection_info: Mapping[str, Any],
        *,
        cost_delta: float,
        speed_cost_delta: float = 0.0,
        distance_cost_delta: float = 0.0,
    ) -> float:
        return self._ppo_local_reward_components(
            current,
            projection_info,
            cost_delta=cost_delta,
            speed_cost_delta=speed_cost_delta,
            distance_cost_delta=distance_cost_delta,
        )["ppo_local"]

    def _ppo_local_reward_components(
        self,
        current: Mapping[str, Any],
        projection_info: Mapping[str, Any],
        *,
        cost_delta: float,
        speed_cost_delta: float,
        distance_cost_delta: float = 0.0,
    ) -> dict[str, float]:
        action_payload = self._continuous_action_payload(projection_info)
        if not action_payload:
            return {"ppo_local": 0.0}

        capacity_readiness = 1.0 - self._macro_capacity_pressure()
        inventory_pressure = self._average_inventory_pressure(current)
        backlog_pressure = self._average_backlog_pressure(current)
        stockout_risk = clamp(
            self._stockout_risk(current) * self.config.stockout_penalty_multiplier,
            0.0,
            1.0,
        )
        inventory_coverage = self._inventory_coverage(current)
        inventory_shortfall = clamp(
            (1.0 - inventory_coverage) * self.config.inventory_shortfall_penalty_multiplier,
            0.0,
            1.0,
        )
        safety_stock_target_gap = self._safety_stock_target_gap()
        demand_volatility_pressure = self._demand_volatility_pressure()
        lead_time_volatility_pressure = self._lead_time_volatility_pressure()
        holding_cost_pressure = self._holding_cost_pressure()
        stockout_penalty_pressure = self._stockout_penalty_pressure()
        supplier_delay_pressure = self._scenario_supplier_delay_pressure()
        scenario_lead_time_volatility_pressure = self._scenario_lead_time_volatility_pressure()
        forecast_inventory_risk = clamp(
            max(
                safety_stock_target_gap,
                inventory_shortfall,
                demand_volatility_pressure,
                lead_time_volatility_pressure,
                supplier_delay_pressure,
                scenario_lead_time_volatility_pressure,
            ),
            0.0,
            1.0,
        )
        order_flow_metrics = self._order_flow_metrics(current, masked_order_ids=set())
        raw_lateness_risk = self._lateness_risk(current)
        urgent_ratio = order_flow_metrics.urgent_unassigned_ratio
        lateness_risk = order_flow_metrics.actionable_lateness_risk
        pending_work_pressure = self._pending_work_pressure(current, masked_order_ids=set())
        flow_pressure = clamp(max(pending_work_pressure, lateness_risk, urgent_ratio), 0.0, 1.0)
        disruption = clamp(float(current.get("disruption_score", 0.0)), 0.0, 1.0)
        speed_multiplier = _safe_float(action_payload.get("speed_multiplier"), 1.0)
        reorder_fraction = clamp(_safe_float(action_payload.get("reorder_fraction"), 0.0), 0.0, 1.0)
        safety_stock_multiplier = max(1.0, _safe_float(action_payload.get("safety_stock_multiplier"), 1.0))
        capacity_buffer_fraction = clamp(
            _safe_float(action_payload.get("capacity_buffer_fraction"), 0.0),
            0.0,
            self.action_projector.max_capacity_buffer_fraction,
        )
        normalized_safety_stock_action = clamp(safety_stock_multiplier - 1.0, 0.0, 1.0)
        normalized_capacity_buffer = clamp(
            capacity_buffer_fraction / max(self.action_projector.max_capacity_buffer_fraction, 1.0e-9),
            0.0,
            1.0,
        )
        normalized_speed = clamp((speed_multiplier - 0.65) / 0.70, 0.0, 1.0)

        speed_pressure = max(0.0, speed_multiplier - 1.0)
        speed_need = clamp(
            max(
                urgent_ratio,
                lateness_risk,
                min(backlog_pressure, urgent_ratio + lateness_risk),
                disruption,
            ),
            0.0,
            1.0,
        )
        speed_cost_denominator = max(distance_cost_delta, cost_delta, 1.0)
        speed_cost_penalty = clamp(speed_cost_delta / speed_cost_denominator, 0.0, 2.0)
        movement_exposure = 1.0 if distance_cost_delta > 0.0 and speed_cost_delta > 0.0 else 0.0
        speed_justification_credit = speed_pressure * speed_need * movement_exposure
        unjustified_speed_penalty = speed_pressure * (1.0 - speed_need)
        no_movement_speed_penalty = speed_pressure * speed_need * (1.0 - movement_exposure)

        planned_units = _safe_float(projection_info.get("planned_replenishment_units"), 0.0)
        planned_useful_units = _safe_float(projection_info.get("planned_replenishment_useful_units"), planned_units)
        planned_cost = _safe_float(projection_info.get("planned_replenishment_cost"), 0.0)
        inventory_risk_justification = clamp(
            max(
                stockout_risk,
                inventory_shortfall,
                safety_stock_target_gap,
                backlog_pressure,
                demand_volatility_pressure,
                lead_time_volatility_pressure,
                supplier_delay_pressure,
                scenario_lead_time_volatility_pressure,
                stockout_penalty_pressure,
            ),
            0.0,
            1.0,
        )
        reorder_need = max(
            stockout_risk,
            inventory_shortfall,
            safety_stock_target_gap,
            backlog_pressure,
            supplier_delay_pressure,
            scenario_lead_time_volatility_pressure,
            stockout_penalty_pressure,
        )
        useful_receipt_factor = clamp(
            planned_useful_units / max(planned_units, 1.0),
            0.0,
            1.0,
        )
        planned_replenishment_credit = reorder_fraction * reorder_need * useful_receipt_factor
        planned_replenishment_cost_penalty = clamp(
            planned_cost / self.config.transport_cost_penalty_scale,
            0.0,
            2.0,
        )
        low_risk = max(0.0, 1.0 - max(stockout_risk, backlog_pressure, inventory_pressure))
        excess_inventory_penalty = reorder_fraction * low_risk
        safety_stock_credit = (safety_stock_multiplier - 1.0) * max(safety_stock_target_gap, stockout_risk)
        safety_stock_overshoot_penalty = max(0.0, safety_stock_multiplier - 1.0) * low_risk
        high_inventory_posture = clamp(
            0.50 * reorder_fraction + 0.50 * normalized_safety_stock_action,
            0.0,
            1.0,
        )
        low_inventory_risk_factor = 1.0 - inventory_risk_justification
        cost_sensitive_inventory_penalty = (
            holding_cost_pressure
            * low_inventory_risk_factor
            * high_inventory_posture
        )

        capacity_pressure = self._macro_capacity_pressure()
        capacity_shock_pressure = self._capacity_shock_pressure()
        vehicle_availability_pressure = clamp(
            _safe_float(projection_info.get("vehicle_availability_pressure"), 0.0),
            0.0,
            1.0,
        )
        feasible_dispatch_ratio = clamp(
            _safe_float(projection_info.get("feasible_dispatch_ratio"), 1.0),
            0.0,
            1.0,
        )
        no_vehicle_available_rate = clamp(
            _safe_float(projection_info.get("no_vehicle_available_rate"), 0.0),
            0.0,
            1.0,
        )
        dispatch_feasibility_pressure = clamp(1.0 - feasible_dispatch_ratio, 0.0, 1.0)
        capacity_scarcity_pressure = clamp(
            max(
                capacity_shock_pressure,
                vehicle_availability_pressure,
                capacity_pressure,
                dispatch_feasibility_pressure,
                no_vehicle_available_rate,
            ),
            0.0,
            1.0,
        )
        buffer_need = clamp(max(capacity_pressure, backlog_pressure, disruption, capacity_scarcity_pressure), 0.0, 1.0)
        buffer_credit = capacity_buffer_fraction * buffer_need
        capacity_opportunity_cost = capacity_buffer_fraction * (1.0 - buffer_need)
        target_min_buffer_under_scarcity = min(0.15, self.action_projector.max_capacity_buffer_fraction)
        capacity_scarcity_buffer_penalty = (
            capacity_scarcity_pressure
            * clamp(
                (target_min_buffer_under_scarcity - capacity_buffer_fraction)
                / max(target_min_buffer_under_scarcity, 1.0e-9),
                0.0,
                1.0,
            )
        )
        reorder_neglect_penalty = (
            self.config.reorder_neglect_weight
            * forecast_inventory_risk
            * (1.0 - reorder_fraction)
        )
        safety_stock_neglect_penalty = (
            self.config.safety_stock_neglect_weight
            * forecast_inventory_risk
            * (1.0 - normalized_safety_stock_action)
        )
        capacity_neglect_penalty = (
            self.config.capacity_neglect_weight
            * capacity_pressure
            * (1.0 - normalized_capacity_buffer)
        )
        low_speed_lateness_penalty = (
            self.config.low_speed_lateness_weight
            * flow_pressure
            * max(0.0, 1.0 - speed_multiplier)
        )
        flow_capacity_neglect_penalty = (
            self.config.flow_capacity_neglect_weight
            * flow_pressure
            * (1.0 - normalized_capacity_buffer)
        )
        ppo_flow_enablement_credit = (
            self.config.flow_enablement_weight
            * flow_pressure
            * normalized_capacity_buffer
            * normalized_speed
        )

        ppo_local = (
            (0.25 * capacity_readiness)
            + (0.16 * (1.0 - inventory_pressure))
            + (0.20 * (1.0 - backlog_pressure))
            + (0.20 * buffer_credit)
            + (self.config.speed_justification_credit_weight * speed_justification_credit)
            + (self.config.planned_replenishment_credit_weight * planned_replenishment_credit)
            + (self.config.safety_stock_gap_credit_weight * safety_stock_credit)
            + ppo_flow_enablement_credit
            - (self.config.speed_cost_penalty_weight * speed_cost_penalty)
            - (self.config.unjustified_speed_penalty_weight * unjustified_speed_penalty)
            - (self.config.unjustified_speed_penalty_weight * no_movement_speed_penalty)
            - (self.config.planned_replenishment_cost_weight * planned_replenishment_cost_penalty)
            - (self.config.excess_inventory_penalty_weight * excess_inventory_penalty)
            - (self.config.excess_inventory_penalty_weight * safety_stock_overshoot_penalty)
            - (self.config.excess_inventory_penalty_weight * cost_sensitive_inventory_penalty)
            - (self.config.capacity_opportunity_cost * capacity_opportunity_cost)
            - (self.config.capacity_neglect_weight * capacity_scarcity_buffer_penalty)
            - reorder_neglect_penalty
            - safety_stock_neglect_penalty
            - capacity_neglect_penalty
            - low_speed_lateness_penalty
            - flow_capacity_neglect_penalty
        )
        return {
            "ppo_local": float(ppo_local),
            "stockout_risk": float(stockout_risk),
            "inventory_coverage": float(inventory_coverage),
            "inventory_shortfall": float(inventory_shortfall),
            "safety_stock_target_gap": float(safety_stock_target_gap),
            "demand_volatility_pressure": float(demand_volatility_pressure),
            "lead_time_volatility_pressure": float(lead_time_volatility_pressure),
            "holding_cost_pressure": float(holding_cost_pressure),
            "stockout_penalty_pressure": float(stockout_penalty_pressure),
            "supplier_delay_pressure": float(supplier_delay_pressure),
            "scenario_lead_time_volatility_pressure": float(scenario_lead_time_volatility_pressure),
            "forecast_inventory_risk": float(forecast_inventory_risk),
            "inventory_risk_justification": float(inventory_risk_justification),
            "flow_pressure": float(flow_pressure),
            "pending_work_pressure": float(pending_work_pressure),
            "urgent_ratio": float(urgent_ratio),
            "lateness_risk": float(lateness_risk),
            "raw_lateness_risk": float(raw_lateness_risk),
            "planned_replenishment_credit": float(planned_replenishment_credit),
            "planned_replenishment_units_reward_basis": float(planned_units),
            "planned_replenishment_useful_units_reward_basis": float(planned_useful_units),
            "planned_replenishment_cost_penalty": float(planned_replenishment_cost_penalty),
            "safety_stock_credit": float(safety_stock_credit),
            "excess_inventory_penalty": float(excess_inventory_penalty),
            "safety_stock_overshoot_penalty": float(safety_stock_overshoot_penalty),
            "high_inventory_posture": float(high_inventory_posture),
            "cost_sensitive_inventory_penalty": float(cost_sensitive_inventory_penalty),
            "speed_cost_penalty": float(speed_cost_penalty),
            "speed_justification_credit": float(speed_justification_credit),
            "unjustified_speed_penalty": float(unjustified_speed_penalty),
            "no_movement_speed_penalty": float(no_movement_speed_penalty),
            "capacity_opportunity_cost": float(capacity_opportunity_cost),
            "buffer_credit": float(buffer_credit),
            "capacity_pressure": float(capacity_pressure),
            "capacity_shock_pressure": float(capacity_shock_pressure),
            "vehicle_availability_pressure": float(vehicle_availability_pressure),
            "dispatch_feasibility_pressure": float(dispatch_feasibility_pressure),
            "no_vehicle_available_rate": float(no_vehicle_available_rate),
            "capacity_scarcity_pressure": float(capacity_scarcity_pressure),
            "target_min_buffer_under_scarcity": float(target_min_buffer_under_scarcity),
            "capacity_scarcity_buffer_penalty": float(capacity_scarcity_buffer_penalty),
            "normalized_speed": float(normalized_speed),
            "normalized_safety_stock_action": float(normalized_safety_stock_action),
            "normalized_capacity_buffer": float(normalized_capacity_buffer),
            "reorder_neglect_penalty": float(reorder_neglect_penalty),
            "safety_stock_neglect_penalty": float(safety_stock_neglect_penalty),
            "capacity_neglect_penalty": float(capacity_neglect_penalty),
            "low_speed_lateness_penalty": float(low_speed_lateness_penalty),
            "flow_capacity_neglect_penalty": float(flow_capacity_neglect_penalty),
            "ppo_flow_enablement_credit": float(ppo_flow_enablement_credit),
            **order_flow_metrics.as_reward_components(),
        }

    def _dqn_local_reward(
        self,
        current: Mapping[str, Any],
        projection_info: Mapping[str, Any],
        *,
        delivered_delta: int,
        cost_delta: float,
        masked_order_ids: set[str],
    ) -> float:
        return self._dqn_local_reward_components(
            current,
            projection_info,
            delivered_delta=delivered_delta,
            cost_delta=cost_delta,
            masked_order_ids=masked_order_ids,
        )["dqn_local"]

    def _dqn_local_reward_components(
        self,
        current: Mapping[str, Any],
        projection_info: Mapping[str, Any],
        *,
        delivered_delta: int,
        cost_delta: float,
        masked_order_ids: set[str],
    ) -> dict[str, float]:
        action_payload = self._discrete_action_payload(projection_info)
        if not action_payload:
            return {"dqn_local": 0.0}

        pending_ratio = self._pending_order_ratio(current, masked_order_ids=masked_order_ids)
        inventory_pressure = self._average_inventory_pressure(current)
        backlog_pressure = self._average_backlog_pressure(current)
        backlog_age_pressure = self._backlog_age_pressure(current, masked_order_ids=masked_order_ids)
        order_flow_metrics = self._order_flow_metrics(current, masked_order_ids=masked_order_ids)
        stockout_risk = clamp(
            self._stockout_risk(current) * self.config.stockout_penalty_multiplier,
            0.0,
            1.0,
        )
        inventory_coverage = self._inventory_coverage(current)
        safety_stock_target_gap = self._safety_stock_target_gap()
        raw_lateness_risk = self._lateness_risk(current)
        raw_urgent_ratio = self._urgent_pending_ratio(current)
        pending_work_pressure = self._pending_work_pressure(current, masked_order_ids=masked_order_ids)
        lateness_risk = order_flow_metrics.actionable_lateness_risk
        urgent_ratio = order_flow_metrics.urgent_unassigned_ratio
        disruption = clamp(float(current.get("disruption_score", 0.0)), 0.0, 1.0)
        congestion = self._network_congestion_score()
        scenario_route_disruption_pressure = self._scenario_route_disruption_pressure()
        route_cost_pressure = self._route_cost_pressure()
        premium_sla_pressure = self._premium_sla_pressure()
        safety_potential_value = clamp(float(current.get("network_safety_potential", 0.0)), 0.0, 1.0)
        route_stress = clamp(
            max(disruption, congestion, scenario_route_disruption_pressure, route_cost_pressure),
            0.0,
            1.0,
        )
        inventory_shortfall = clamp(
            (1.0 - inventory_coverage) * self.config.inventory_shortfall_penalty_multiplier,
            0.0,
            1.0,
        )
        crisis = clamp(
            max(
                stockout_risk,
                inventory_shortfall,
                safety_stock_target_gap,
                backlog_pressure,
                lateness_risk,
            ),
            0.0,
            1.0,
        )
        emergency_gate_input = clamp((crisis - 0.25) / 0.45, 0.0, 1.0)
        emergency_gate = _smoothstep_unit(emergency_gate_input)
        pre_crisis_pressure = clamp(
            max(
                pending_work_pressure,
                urgent_ratio,
                lateness_risk,
            ),
            0.0,
            1.0,
        )
        hold_gate = _smoothstep_unit((pre_crisis_pressure - 0.25) / 0.50)
        cost_per_delivery = cost_delta / max(delivered_delta, 1)

        reward = 0.0
        reward -= clamp(cost_per_delivery / self.config.transport_cost_penalty_scale, 0.0, 2.0) * 0.25

        dispatch = str(action_payload.get("dispatch", "hold"))
        dispatched_orders = _safe_float(projection_info.get("dqn_dispatched_orders"), 0.0)
        macro_dispatch_budget = max(1.0, _safe_float(projection_info.get("macro_dispatch_budget"), 1.0))
        dispatch_success_count = _safe_float(projection_info.get("dispatch_success_count"), dispatched_orders)
        current_dispatch_work_count = max(dispatched_orders, dispatch_success_count)
        current_dispatch_work_signal = (
            1.0 if dispatch == "dispatch" and current_dispatch_work_count > 0.0 else 0.0
        )
        successful_dispatch_count_capped = min(dispatched_orders, macro_dispatch_budget)
        dispatch_feasibility_credit = 0.0
        dispatch_progress_credit = 0.0
        dispatch_progress_quality_factor = 1.0
        dispatch_feasibility_quality_factor = 1.0
        demand_dispatch_quality_penalty = 0.0
        demand_service_collapse_penalty = 0.0
        demand_action_family_penalty = 0.0
        service_gap = clamp(self.config.service_level_target - float(current.get("service_level", 1.0)), 0.0, 1.0)
        service_collapse_pressure = _smoothstep_unit(service_gap / 0.35)
        lateness_collapse_pressure = _smoothstep_unit(order_flow_metrics.true_lateness_pressure / 0.25)
        demand_operational_degradation = clamp(
            max(service_collapse_pressure, lateness_collapse_pressure),
            0.0,
            1.0,
        )
        demand_dispatch_pressure = 0.0
        low_feasible_dispatch_factor = 0.0
        projected_dispatch_exposure = 0.0
        already_assigned_exposure = 0.0
        no_vehicle_exposure = 0.0
        action_family_concentration_proxy = 0.0
        unnecessary_dispatch_penalty = 0.0
        no_unassigned_dispatch_exposure = (
            1.0
            if dispatch == "dispatch"
            and _safe_float(projection_info.get("dispatch_no_unassigned_orders"), 0.0) > 0.0
            else 0.0
        )
        no_current_or_unassigned_dispatch_penalty = 0.0
        dqn_delivery_credit_blocked_no_current_dispatch_work = 0.0
        infeasible_dispatch_penalty = 0.0
        infeasible_dispatch_repeat_exposure = 0.0
        useful_dispatch_lateness_urgency_credit = 0.0
        useful_dispatch_opportunity_pressure = 0.0
        useful_dispatch_opportunity_missed = 0.0
        hold_under_lateness_pressure = 0.0
        hold_under_lateness_pressure_penalty = 0.0
        hold_timing_degradation_pressure = 0.0
        secondary_fleet_lateness_exposure = 0.0
        no_immediate_delivery_exposure = 1.0 if delivered_delta <= 0 else 0.0
        is_hold_action = 1.0 if dispatch == "hold" else 0.0
        no_dispatched_orders = 1.0 if dispatched_orders <= 0.0 else 0.0
        hold_penalty = self.config.hold_penalty_weight * hold_gate * is_hold_action * no_dispatched_orders
        current_inaction_pressure = pre_crisis_pressure * is_hold_action * no_dispatched_orders
        reward -= hold_penalty
        infeasible_dispatch_attempt = _safe_float(projection_info.get("infeasible_dispatch_attempt"), 0.0)
        repeated_infeasible_dispatch_attempt_rate = _safe_float(
            projection_info.get("repeated_infeasible_dispatch_attempt_rate"),
            0.0,
        )
        infeasible_dispatch_repeat_gate = _smoothstep_unit(
            (repeated_infeasible_dispatch_attempt_rate - 0.20) / 0.50
        )
        infeasible_dispatch_attempt_window_count = _safe_float(
            projection_info.get("infeasible_dispatch_attempt_window_count"),
            20.0,
        )
        infeasible_dispatch_repeat_exposure = _smoothstep_unit(
            (infeasible_dispatch_attempt_window_count - 1.0) / 3.0
        )
        vehicle_availability_pressure = _safe_float(projection_info.get("vehicle_availability_pressure"), 0.0)
        infeasible_dispatch_scarcity_relief = 0.50 * clamp(vehicle_availability_pressure, 0.0, 1.0)
        useful_dispatch_timing_pressure = clamp(
            max(
                order_flow_metrics.true_lateness_pressure,
                order_flow_metrics.actionable_lateness_risk,
                order_flow_metrics.due_soon_pressure,
                urgent_ratio,
                lateness_risk,
                premium_sla_pressure,
                scenario_route_disruption_pressure,
                route_cost_pressure,
            ),
            0.0,
            1.0,
        )
        useful_dispatch_opportunity_pressure = clamp(
            useful_dispatch_timing_pressure
            * max(
                order_flow_metrics.unassigned_backlog_pressure,
                order_flow_metrics.actionable_flow_pressure,
                pending_work_pressure,
            ),
            0.0,
            1.0,
        )
        if dispatch == "dispatch":
            infeasible_dispatch_penalty = (
                self.config.infeasible_dispatch_penalty_weight
                * clamp(infeasible_dispatch_attempt, 0.0, 1.0)
                * infeasible_dispatch_repeat_gate
                * infeasible_dispatch_repeat_exposure
                * (1.0 - infeasible_dispatch_scarcity_relief)
            )
            reward -= infeasible_dispatch_penalty
            feasible_dispatch_ratio = clamp(
                _safe_float(projection_info.get("feasible_dispatch_ratio"), 1.0),
                0.0,
                1.0,
            )
            already_assigned_exposure = clamp(
                max(
                    _safe_float(projection_info.get("already_assigned_rate"), 0.0),
                    _safe_float(projection_info.get("dispatch_already_assigned_count"), 0.0)
                    / max(
                        _safe_float(projection_info.get("dispatchable_order_count"), 0.0)
                        + _safe_float(projection_info.get("dispatch_already_assigned_count"), 0.0),
                        1.0,
                    ),
                ),
                0.0,
                1.0,
            )
            no_vehicle_exposure = clamp(
                max(
                    _safe_float(projection_info.get("no_vehicle_available_rate"), 0.0),
                    _safe_float(projection_info.get("dispatch_no_vehicle_available"), 0.0)
                    / max(
                        _safe_float(projection_info.get("dispatchable_order_count"), 0.0)
                        + _safe_float(projection_info.get("dispatch_no_vehicle_available"), 0.0),
                        1.0,
                    ),
                ),
                0.0,
                1.0,
            )
            low_feasible_dispatch_factor = _smoothstep_unit((0.40 - feasible_dispatch_ratio) / 0.40)
            projected_dispatch_exposure = 1.0 if bool(
                projection_info.get("projected", False) or projection_info.get("discrete_projected", False)
            ) else 0.0
            demand_dispatch_pressure = clamp(
                max(
                    pending_work_pressure,
                    raw_urgent_ratio,
                    urgent_ratio,
                    lateness_risk,
                    backlog_age_pressure,
                    self._demand_volatility_pressure(),
                ),
                0.0,
                1.0,
            )
            dispatch_quality_risk = clamp(
                max(
                    already_assigned_exposure,
                    no_vehicle_exposure,
                    low_feasible_dispatch_factor * projected_dispatch_exposure,
                    low_feasible_dispatch_factor * service_collapse_pressure,
                ),
                0.0,
                1.0,
            )
            dispatch_progress_quality_factor = 1.0 - (
                0.90 * demand_dispatch_pressure * dispatch_quality_risk
            )
            dispatch_progress_quality_factor = clamp(dispatch_progress_quality_factor, 0.10, 1.0)
            demand_degraded_low_feasibility = (
                demand_dispatch_pressure
                * demand_operational_degradation
                * low_feasible_dispatch_factor
            )
            dispatch_feasibility_quality_factor = clamp(
                1.0 - (0.90 * demand_degraded_low_feasibility),
                0.10,
                1.0,
            )
            demand_dispatch_quality_penalty = (
                demand_dispatch_pressure
                * (
                    (0.16 * already_assigned_exposure)
                    + (0.14 * no_vehicle_exposure)
                    + (0.14 * low_feasible_dispatch_factor * projected_dispatch_exposure)
                    + (0.12 * low_feasible_dispatch_factor * service_collapse_pressure)
                    + (
                        0.06
                        * low_feasible_dispatch_factor
                        * no_immediate_delivery_exposure
                        * service_collapse_pressure
                    )
                )
            )
            demand_service_collapse_penalty = (
                demand_degraded_low_feasibility
                * no_immediate_delivery_exposure
                * (0.18 + (0.12 * already_assigned_exposure))
            )
            degraded_action_family = (
                str(action_payload.get("route", "")) == "shortest"
                and str(action_payload.get("mode", "")) == "secondary_fleet"
                and str(action_payload.get("reorder", "none")) in {"none", "conservative"}
            )
            if degraded_action_family:
                action_family_concentration_proxy = clamp(
                    demand_dispatch_pressure
                    * max(
                        already_assigned_exposure,
                        no_vehicle_exposure,
                        low_feasible_dispatch_factor * projected_dispatch_exposure,
                        0.50 * low_feasible_dispatch_factor * no_immediate_delivery_exposure,
                    ),
                    0.0,
                    1.0,
                )
                demand_action_family_penalty = (
                    0.18
                    * demand_degraded_low_feasibility
                    * no_immediate_delivery_exposure
                    + 0.12
                    * action_family_concentration_proxy
                    * (0.50 + (0.50 * low_feasible_dispatch_factor))
                )
            no_current_dispatch_work = 1.0 if current_dispatch_work_signal <= 0.0 else 0.0
            no_current_or_unassigned_dispatch_penalty = max(
                no_current_dispatch_work,
                no_unassigned_dispatch_exposure,
            ) * (
                0.14
                + (0.10 * demand_dispatch_pressure)
                + (0.04 * no_immediate_delivery_exposure)
            )
            reward -= no_current_or_unassigned_dispatch_penalty
        if dispatch == "hold" and no_dispatched_orders > 0.0:
            hold_under_lateness_pressure = useful_dispatch_opportunity_pressure
            useful_dispatch_opportunity_missed = 1.0 if hold_under_lateness_pressure > 0.0 else 0.0
            hold_timing_degradation_pressure = clamp(
                max(
                    demand_operational_degradation,
                    self._holding_cost_pressure(),
                    self._macro_capacity_pressure(),
                    self._capacity_shock_pressure(),
                    vehicle_availability_pressure,
                ),
                0.0,
                1.0,
            )
            hold_under_lateness_pressure_penalty = min(
                0.16,
                0.04
                + (0.06 * hold_under_lateness_pressure)
                + (0.06 * hold_timing_degradation_pressure),
            ) if hold_under_lateness_pressure > 0.0 else 0.0
            reward -= hold_under_lateness_pressure_penalty
        if bool(projection_info.get("discrete_blocked", False)):
            reward -= 0.35
        elif dispatch == "dispatch":
            if dispatched_orders > 0.0:
                dispatch_feasibility_credit = (
                    0.05 + (0.10 * min(1.0, pending_ratio + raw_urgent_ratio))
                ) * dispatch_feasibility_quality_factor
                dispatch_progress_credit = (
                    self.config.dispatch_progress_weight
                    * successful_dispatch_count_capped
                    * pending_work_pressure
                    * dispatch_progress_quality_factor
                )
                reward += dispatch_feasibility_credit
                reward += dispatch_progress_credit
                useful_dispatch_lateness_urgency_credit = (
                    min(0.12, 0.03 + (0.09 * useful_dispatch_timing_pressure))
                    if current_dispatch_work_signal > 0.0
                    and useful_dispatch_timing_pressure > 0.0
                    and no_unassigned_dispatch_exposure <= 0.0
                    and infeasible_dispatch_attempt <= 0.0
                    and not bool(projection_info.get("blocked", False))
                    and not bool(projection_info.get("discrete_blocked", False))
                    else 0.0
                )
                reward += useful_dispatch_lateness_urgency_credit
                reward -= demand_dispatch_quality_penalty
                reward -= demand_service_collapse_penalty
                reward -= demand_action_family_penalty
            else:
                unnecessary_dispatch_penalty = 0.08
                reward -= unnecessary_dispatch_penalty
        elif pending_ratio > 0.50:
            unnecessary_dispatch_penalty = 0.08 * pending_ratio
            reward -= unnecessary_dispatch_penalty

        reorder = str(action_payload.get("reorder", "none"))
        emergency_procurement_penalty = 0.0
        emergency_procurement_justification_credit = 0.0
        emergency_useful_factor = 0.0
        emergency_cost_norm = 0.0
        delay_abuse_penalty = 0.0
        if reorder in {"emergency", "aggressive"}:
            override_units = _safe_float(projection_info.get("dqn_replenishment_override_units"), 0.0)
            override_useful_units = _safe_float(projection_info.get("dqn_replenishment_override_useful_units"), 0.0)
            override_cost = _safe_float(projection_info.get("dqn_replenishment_override_cost"), 0.0)
            emergency_useful_factor = clamp(
                override_useful_units / max(override_units, 1.0),
                0.0,
                1.0,
            )
            emergency_cost_norm = clamp(
                override_cost / self.config.transport_cost_penalty_scale,
                0.0,
                2.0,
            )
            emergency_procurement_justification_credit = (
                self.config.emergency_credit_weight
                * emergency_gate
                * emergency_useful_factor
            )
            emergency_procurement_penalty = (
                self.config.emergency_base_penalty * (1.0 - (0.60 * emergency_gate))
                + (self.config.emergency_cost_weight * emergency_cost_norm)
                + (self.config.emergency_abuse_penalty * (1.0 - emergency_gate))
            )
            delay_abuse_penalty = (
                self.config.delay_weight
                * emergency_gate
                * lateness_risk
                * clamp(self._previous_inaction_pressure, 0.0, 1.0)
            )
            reward += emergency_procurement_justification_credit - emergency_procurement_penalty - delay_abuse_penalty

        mode = str(action_payload.get("mode", ""))
        primary_fleet_cost_penalty = 0.0
        premium_sla_fleet_justification_credit = 0.0
        premium_primary_adaptation_credit = 0.0
        premium_sla_fleet_credit_blocked_no_current_work = 0.0
        premium_primary_adaptation_credit_blocked_no_current_work = 0.0
        premium_fleet_credit_allowed = 0.0
        premium_primary_underuse_penalty = 0.0
        primary_fleet_timing_justification_credit = 0.0
        premium_fleet_justification = clamp(
            max(
                premium_sla_pressure,
                urgent_ratio,
                raw_urgent_ratio * premium_sla_pressure,
                order_flow_metrics.due_soon_pressure,
                order_flow_metrics.actionable_lateness_risk,
                lateness_risk,
            ),
            0.0,
            1.0,
        )
        if mode == "primary_fleet":
            primary_fleet_cost_penalty = self.config.primary_fleet_penalty * (1.0 - premium_fleet_justification) ** 2
            potential_premium_primary_adaptation_credit = (
                self.config.primary_fleet_penalty
                * 0.80
                * premium_fleet_justification
            )
            premium_fleet_credit_allowed = (
                1.0
                if dispatch == "dispatch"
                and current_dispatch_work_signal > 0.0
                and (dispatch_success_count > 0.0 or dispatched_orders > 0.0)
                and potential_premium_primary_adaptation_credit > 0.0
                else 0.0
            )
            if premium_fleet_credit_allowed > 0.0:
                premium_primary_adaptation_credit = potential_premium_primary_adaptation_credit
                premium_sla_fleet_justification_credit = premium_primary_adaptation_credit
            elif potential_premium_primary_adaptation_credit > 0.0 and current_dispatch_work_signal <= 0.0:
                premium_sla_fleet_credit_blocked_no_current_work = 1.0
                premium_primary_adaptation_credit_blocked_no_current_work = 1.0
            primary_fleet_timing_pressure = clamp(
                max(useful_dispatch_timing_pressure, premium_fleet_justification),
                0.0,
                1.0,
            )
            primary_fleet_timing_justification_credit = (
                min(0.06, 0.02 + (0.04 * primary_fleet_timing_pressure))
                if dispatch == "dispatch"
                and current_dispatch_work_signal > 0.0
                and primary_fleet_timing_pressure > 0.0
                and no_unassigned_dispatch_exposure <= 0.0
                and infeasible_dispatch_attempt <= 0.0
                and not bool(projection_info.get("blocked", False))
                and not bool(projection_info.get("discrete_blocked", False))
                else 0.0
            )
            reward -= primary_fleet_cost_penalty
            reward += premium_sla_fleet_justification_credit
            reward += primary_fleet_timing_justification_credit
        elif mode == "secondary_fleet":
            secondary_fleet_lateness_exposure = (
                useful_dispatch_timing_pressure
                if dispatch == "dispatch" and current_dispatch_work_signal > 0.0
                else 0.0
            )
            secondary_underuse_penalty_allowed = (
                dispatch == "dispatch"
                and current_dispatch_work_signal > 0.0
                and no_unassigned_dispatch_exposure <= 0.0
                and infeasible_dispatch_attempt <= 0.0
                and not bool(projection_info.get("blocked", False))
                and not bool(projection_info.get("discrete_blocked", False))
            )
            if secondary_underuse_penalty_allowed:
                premium_primary_underuse_penalty = (
                    0.04
                    * premium_sla_pressure
                    * clamp(
                        max(
                            urgent_ratio,
                            order_flow_metrics.due_soon_pressure,
                            order_flow_metrics.actionable_lateness_risk,
                        ),
                        0.0,
                        1.0,
                    )
                )
                reward -= premium_primary_underuse_penalty

        route = str(action_payload.get("route", ""))
        hold_route_neutralized = 1.0 if dispatch == "hold" else 0.0
        route_label_ignored_for_hold = 1.0 if dispatch == "hold" else 0.0
        route_for_reward = "" if dispatch == "hold" else route
        selected_route_effective_for_reward = (
            -1.0
            if dispatch == "hold"
            else {
                "shortest": 0.0,
                "low_congestion": 1.0,
                "high_resilience": 2.0,
            }.get(route, -2.0)
        )
        route_credit_blocked_hold = hold_route_neutralized
        candidate_alignment_blocked_hold = hold_route_neutralized
        hold_inflight_completion_credit_blocked_for_route = (
            1.0 if dispatch == "hold" and delivered_delta > 0 else 0.0
        )
        route_resilience_credit = 0.0
        route_premium_penalty = 0.0
        shortest_route_disruption_risk_penalty = 0.0
        route_resilience_adaptation_credit = 0.0
        low_congestion_adaptation_credit = 0.0
        high_resilience_adaptation_credit = 0.0
        route_adaptation_reward_or_credit = 0.0
        low_congestion_balance_dampening = 0.0
        high_resilience_balance_support = 0.0
        low_congestion_no_useful_work_credit_blocked = 0.0
        high_resilience_no_useful_work_credit_blocked = 0.0
        alternative_route_balance_adjustment = 0.0
        route_narrowness_penalty = 0.0
        low_congestion_hold_credit_dampening = 0.0
        high_resilience_hold_credit_dampening = 0.0
        route_credit_useful_work_factor = 0.0 if dispatch == "hold" else 1.0
        passive_route_credit_dampening = 0.0
        shortest_moderate_pressure_relief = 0.0
        shortest_relief_pressure_factor = 0.0
        shortest_relief_observed_profile_support = 0.0
        shortest_relief_service_health_factor = 0.0
        shortest_relief_useful_work_factor = 0.0
        shortest_relief_eligible = 0.0
        shortest_relief_severe_pressure_blocked = 0.0
        shortest_relief_route_failure_blocked = 0.0
        shortest_relief_customer_revisit_blocked = 0.0
        shortest_relief_brittle_secondary_blocked = 0.0
        shortest_relief_no_useful_work_blocked = 0.0
        shortest_relief_impossible_dispatch_blocked = 0.0
        shortest_route_candidate_score = 0.0
        low_congestion_route_candidate_score = 0.0
        high_resilience_route_candidate_score = 0.0
        selected_route_candidate_score = 0.0
        best_route_candidate_score = 0.0
        selected_route_candidate_score_gap = 0.0
        selected_route_candidate_near_best = 0.0
        selected_route_candidate_action_gate = 0.0
        route_candidate_alignment_credit = 0.0
        route_candidate_mismatch_penalty = 0.0
        candidate_alignment_eligible = 0.0
        candidate_alignment_blocked_no_useful_work = 0.0
        candidate_alignment_blocked_brittle_secondary = 0.0
        candidate_alignment_blocked_route_failure = 0.0
        candidate_alignment_blocked_customer_revisit = 0.0
        candidate_alignment_blocked_severe_pressure = 0.0
        candidate_alignment_blocked_impossible_dispatch = 0.0
        candidate_alignment_blocked_service_lateness = 0.0
        shortest_secondary_brittle_risk = 0.0
        shortest_secondary_safe_useful_exception = 0.0
        shortest_secondary_guard_active = 0.0
        shortest_secondary_guard_relaxed_safe = 0.0
        action24_25_candidate_credit_allowed = 0.0
        action24_25_candidate_credit_blocked_reason = 0.0
        route_candidate_operational_health_factor = clamp(1.0 - demand_operational_degradation, 0.0, 1.0)
        route_candidate_service_factor = 0.0
        route_candidate_useful_work_factor = 0.0
        route_adaptation_pressure = clamp(
            max(disruption, congestion, scenario_route_disruption_pressure, route_cost_pressure),
            0.0,
            1.0,
        )
        scenario_congestion_pressure = clamp(
            (self._scenario_stress_value("congestion_multiplier", 1.0) - 1.0) / 3.0,
            0.0,
            1.0,
        )
        route_disruption_reliability_pressure = clamp(
            max(
                disruption,
                self._scenario_stress_value("route_disruption_probability", 0.0),
                (self._scenario_stress_value("shortest_route_risk_multiplier", 1.0) - 1.0) / 3.0,
                (self._scenario_stress_value("disruption_risk_multiplier", 1.0) - 1.0) / 3.0,
            ),
            0.0,
            1.0,
        )
        route_congestion_cost_pressure = clamp(
            max(congestion, scenario_congestion_pressure, route_cost_pressure),
            0.0,
            1.0,
        )
        route_pressure_total = route_disruption_reliability_pressure + route_congestion_cost_pressure
        if route_pressure_total > 1e-9:
            route_pressure_reliability_share = route_disruption_reliability_pressure / route_pressure_total
            route_pressure_congestion_share = route_congestion_cost_pressure / route_pressure_total
            route_pressure_balance = min(
                route_disruption_reliability_pressure,
                route_congestion_cost_pressure,
            ) / max(route_disruption_reliability_pressure, route_congestion_cost_pressure, 1e-9)
        else:
            route_pressure_reliability_share = 0.0
            route_pressure_congestion_share = 0.0
            route_pressure_balance = 0.0
        route_alt_pressure_imbalance = abs(route_pressure_congestion_share - route_pressure_reliability_share)
        route_balanced_pressure = min(route_disruption_reliability_pressure, route_congestion_cost_pressure)
        high_resilience_need = _smoothstep_unit(
            (
                max(
                    disruption,
                    route_disruption_reliability_pressure,
                    0.50 * route_cost_pressure,
                )
                - 0.15
            )
            / 0.45
        )
        low_congestion_need = _smoothstep_unit((route_congestion_cost_pressure - 0.05) / 0.55)
        route_failure_count = _safe_float(projection_info.get("dispatch_route_failure"), 0.0)
        route_success_count = dispatch_success_count
        route_failure_credit_suppression = 0.0
        if dispatch == "dispatch" and route_failure_count > 0.0:
            route_failure_credit_suppression = clamp(
                route_failure_count / max(route_failure_count + max(route_success_count, 0.0), 1.0),
                0.0,
                1.0,
            )
        route_credit_multiplier = 1.0 - route_failure_credit_suppression
        route_service_level = clamp(_safe_float(current.get("service_level"), 1.0), 0.0, 1.0)
        route_flow_service_level = (
            clamp(order_flow_metrics.delivered_orders / max(order_flow_metrics.eligible_orders, 1), 0.0, 1.0)
            if order_flow_metrics.eligible_orders > 0
            else route_service_level
        )
        route_effective_service_level = max(route_service_level, route_flow_service_level)
        route_service_health_factor = _smoothstep_unit(
            (route_effective_service_level - (self.config.service_level_target - 0.05)) / 0.08
        )
        route_candidate_operational_health_factor = clamp(
            route_effective_service_level * (1.0 - order_flow_metrics.true_lateness_pressure),
            0.0,
            1.0,
        )
        route_dispatch_success_signal = 0.0
        if dispatch == "dispatch":
            route_dispatch_success_signal = clamp(
                max(route_success_count, dispatched_orders) / max(macro_dispatch_budget, 1.0),
                0.0,
                1.0,
            )
        route_delivery_work_signal = (
            min(float(delivered_delta), 1.0) * current_dispatch_work_signal
            if dispatch == "dispatch"
            else 0.0
        )
        route_useful_work_signal = clamp(
            max(
                route_delivery_work_signal,
                route_dispatch_success_signal,
                1.0 if dispatch_feasibility_credit > 0.0 else 0.0,
                1.0 if dispatch_progress_credit > 0.0 else 0.0,
            ),
            0.0,
            1.0,
        )
        shortest_secondary_action_context = (
            dispatch == "dispatch"
            and route == "shortest"
            and mode == "secondary_fleet"
            and reorder in {"none", "conservative"}
        )
        customer_revisit_blocked_count = _safe_float(
            projection_info.get("customer_revisit_blocked_count"),
            0.0,
        )
        route_candidate_service_factor = min(
            route_service_health_factor,
            route_candidate_operational_health_factor,
        )
        route_candidate_useful_work_factor = route_useful_work_signal
        route_candidate_execution_factor = (
            route_candidate_service_factor
            * route_credit_multiplier
        )
        max_route_pressure = max(route_disruption_reliability_pressure, route_congestion_cost_pressure)
        shortest_candidate_moderate_pressure_support = _smoothstep_unit((0.45 - max_route_pressure) / 0.20)
        shortest_candidate_congestion_support = (
            _smoothstep_unit((route_congestion_cost_pressure - 0.34) / 0.06)
            * _smoothstep_unit((0.50 - route_congestion_cost_pressure) / 0.08)
        )
        shortest_candidate_reliability_support = (
            _smoothstep_unit((route_disruption_reliability_pressure - 0.16) / 0.06)
            * _smoothstep_unit((0.36 - route_disruption_reliability_pressure) / 0.10)
        )
        shortest_candidate_balance_support = _smoothstep_unit((route_pressure_balance - 0.45) / 0.20)
        shortest_candidate_observed_profile_support = min(
            shortest_candidate_congestion_support,
            shortest_candidate_reliability_support,
            shortest_candidate_balance_support,
        )
        shortest_candidate_pressure_support = max(
            shortest_candidate_moderate_pressure_support,
            shortest_candidate_observed_profile_support,
        )
        high_resilience_candidate_reliability_support = _smoothstep_unit(
            (route_disruption_reliability_pressure - 0.12) / 0.28
        )
        if route_adaptation_pressure > 0.0 and dispatch == "dispatch":
            shortest_route_candidate_score = clamp(
                shortest_candidate_pressure_support * route_candidate_execution_factor,
                0.0,
                1.0,
            )
            low_congestion_route_candidate_score = clamp(
                low_congestion_need
                * (0.70 + (0.30 * route_pressure_congestion_share))
                * (0.75 + (0.25 * route_pressure_balance))
                * route_candidate_execution_factor,
                0.0,
                1.0,
            )
            high_resilience_route_candidate_score = clamp(
                max(high_resilience_need, high_resilience_candidate_reliability_support)
                * (0.70 + (0.30 * route_pressure_reliability_share))
                * (0.70 + (0.30 * route_pressure_balance))
                * route_candidate_execution_factor,
                0.0,
                1.0,
            )
        if route_for_reward == "shortest":
            selected_route_candidate_score = shortest_route_candidate_score
        elif route_for_reward == "low_congestion":
            selected_route_candidate_score = low_congestion_route_candidate_score
        elif route_for_reward == "high_resilience":
            selected_route_candidate_score = high_resilience_route_candidate_score
        best_route_candidate_score = max(
            shortest_route_candidate_score,
            low_congestion_route_candidate_score,
            high_resilience_route_candidate_score,
        )
        selected_route_candidate_score_gap = max(best_route_candidate_score - selected_route_candidate_score, 0.0)
        selected_route_candidate_near_best = (
            1.0
            if selected_route_candidate_score > 0.0 and selected_route_candidate_score_gap <= 0.30
            else 0.0
        )
        selected_route_candidate_action_gate = (
            1.0
            if dispatch == "dispatch"
            and route_candidate_useful_work_factor >= 0.50
            and route_candidate_service_factor >= 0.50
            and route_failure_count <= 0.0
            and customer_revisit_blocked_count <= 0.0
            and infeasible_dispatch_attempt <= 0.0
            and not bool(projection_info.get("blocked", False))
            and not bool(projection_info.get("discrete_blocked", False))
            else 0.0
        )
        shortest_secondary_severe_pressure = (
            route_adaptation_pressure > 0.0
            and (
                max_route_pressure >= 0.55
                or route_disruption_reliability_pressure >= 0.45
                or scenario_route_disruption_pressure >= 0.55
                or shortest_candidate_pressure_support <= 0.0
            )
        )
        shortest_secondary_impossible_or_blocked = (
            infeasible_dispatch_attempt > 0.0
            or bool(projection_info.get("blocked", False))
            or bool(projection_info.get("discrete_blocked", False))
        )
        shortest_secondary_no_useful_work = route_candidate_useful_work_factor < 0.50
        shortest_secondary_service_degraded = route_candidate_service_factor < 0.80
        shortest_secondary_no_vehicle = no_vehicle_exposure > 0.0
        shortest_secondary_already_assigned_saturation = already_assigned_exposure >= 0.75
        shortest_secondary_concentration = action_family_concentration_proxy >= 0.35
        if shortest_secondary_action_context:
            if shortest_secondary_severe_pressure:
                action24_25_candidate_credit_blocked_reason = 1.0
            elif route_failure_count > 0.0:
                action24_25_candidate_credit_blocked_reason = 2.0
            elif customer_revisit_blocked_count > 0.0:
                action24_25_candidate_credit_blocked_reason = 3.0
            elif shortest_secondary_no_useful_work:
                action24_25_candidate_credit_blocked_reason = 4.0
            elif shortest_secondary_impossible_or_blocked:
                action24_25_candidate_credit_blocked_reason = 5.0
            elif shortest_secondary_no_vehicle:
                action24_25_candidate_credit_blocked_reason = 6.0
            elif shortest_secondary_already_assigned_saturation:
                action24_25_candidate_credit_blocked_reason = 7.0
            elif shortest_secondary_service_degraded:
                action24_25_candidate_credit_blocked_reason = 8.0
            elif shortest_secondary_concentration:
                action24_25_candidate_credit_blocked_reason = 9.0
            safe_moderate_shortest_secondary = (
                action24_25_candidate_credit_blocked_reason <= 0.0
                and route_adaptation_pressure > 0.0
                and max_route_pressure <= 0.50
                and route_disruption_reliability_pressure <= 0.35
                and selected_route_candidate_score > 0.0
                and selected_route_candidate_near_best > 0.0
                and selected_route_candidate_score_gap <= 0.12
                and route_candidate_useful_work_factor >= 0.50
                and route_candidate_service_factor >= 0.80
            )
            if safe_moderate_shortest_secondary:
                shortest_secondary_safe_useful_exception = 1.0
                shortest_secondary_guard_relaxed_safe = 1.0
                action24_25_candidate_credit_allowed = 1.0
            else:
                shortest_secondary_brittle_risk = 1.0
                shortest_secondary_guard_active = 1.0
                if action24_25_candidate_credit_blocked_reason <= 0.0:
                    action24_25_candidate_credit_blocked_reason = 10.0
        if shortest_secondary_guard_active > 0.0:
            selected_route_candidate_action_gate = 0.0
        if route_for_reward == "high_resilience":
            route_premium_penalty = self.config.high_resilience_route_penalty * (1.0 - route_adaptation_pressure) ** 2
            route_resilience_credit = min(
                self.config.high_resilience_route_penalty
                * (
                    (1.00 * route_disruption_reliability_pressure)
                    + (0.40 * route_balanced_pressure)
                )
                * (0.65 + (0.35 * high_resilience_need))
                * route_credit_multiplier,
                self.config.high_resilience_route_penalty * 1.40,
            )
            high_resilience_adaptation_credit = min(
                self.config.high_resilience_route_penalty
                * (
                    (1.05 * route_disruption_reliability_pressure)
                    + (0.50 * route_balanced_pressure)
                )
                * (0.70 + (0.30 * high_resilience_need))
                * route_credit_multiplier,
                self.config.high_resilience_route_penalty * 1.55,
            )
            if (
                route_adaptation_pressure > 0.0
                and route_disruption_reliability_pressure >= 0.20
                and route_pressure_balance > 0.45
                and route_pressure_reliability_share >= 0.30
            ):
                high_resilience_balance_support = (
                    self.config.high_resilience_route_penalty
                    * 0.08
                    * route_pressure_balance
                    * min(route_disruption_reliability_pressure / 0.25, 1.0)
                    * route_credit_multiplier
                )
                high_resilience_adaptation_credit = min(
                    high_resilience_adaptation_credit + high_resilience_balance_support,
                    self.config.high_resilience_route_penalty * 1.55,
                )
            if route_adaptation_pressure > 0.0 and high_resilience_adaptation_credit > 0.0:
                route_credit_useful_work_factor = 1.0 - (
                    0.80
                    * route_service_health_factor
                    * (1.0 - route_useful_work_signal)
                    * route_credit_multiplier
                )
                route_credit_useful_work_factor = clamp(route_credit_useful_work_factor, 0.20, 1.0)
                passive_route_credit_dampening = high_resilience_adaptation_credit * (
                    1.0 - route_credit_useful_work_factor
                )
                if dispatch == "hold" or dispatched_orders <= 0.0 or route_useful_work_signal < 0.50:
                    high_resilience_hold_credit_dampening = passive_route_credit_dampening
                    high_resilience_adaptation_credit = max(
                        high_resilience_adaptation_credit - high_resilience_hold_credit_dampening,
                        0.0,
                    )
            if dispatch == "dispatch" and route_candidate_useful_work_factor < 0.50:
                high_resilience_no_useful_work_credit_blocked = 1.0
                route_resilience_credit = 0.0
                high_resilience_adaptation_credit = 0.0
                high_resilience_balance_support = 0.0
            route_resilience_adaptation_credit = high_resilience_adaptation_credit
            route_adaptation_reward_or_credit = route_resilience_credit + route_resilience_adaptation_credit
            alternative_route_balance_adjustment = high_resilience_balance_support
            reward += route_resilience_credit + route_resilience_adaptation_credit - route_premium_penalty
        elif route_for_reward == "low_congestion":
            route_premium_penalty = self.config.low_congestion_route_penalty * (1.0 - route_adaptation_pressure) ** 2
            low_congestion_adaptation_credit = min(
                self.config.low_congestion_route_penalty
                * (
                    (2.80 * route_congestion_cost_pressure)
                    + (0.70 * route_balanced_pressure)
                )
                * (0.80 + (0.20 * low_congestion_need))
                * route_credit_multiplier,
                self.config.low_congestion_route_penalty * 3.50,
            )
            if (
                route_adaptation_pressure > 0.0
                and route_pressure_congestion_share > 0.60
                and route_pressure_balance > 0.45
                and route_disruption_reliability_pressure >= 0.20
            ):
                low_congestion_balance_dampening = min(
                    low_congestion_adaptation_credit * 0.22,
                    self.config.low_congestion_route_penalty * 0.28,
                )
                low_congestion_adaptation_credit = max(
                    low_congestion_adaptation_credit - low_congestion_balance_dampening,
                    0.0,
                )
            if route_adaptation_pressure > 0.0 and low_congestion_adaptation_credit > 0.0:
                route_credit_useful_work_factor = 1.0 - (
                    0.80
                    * route_service_health_factor
                    * (1.0 - route_useful_work_signal)
                    * route_credit_multiplier
                )
                route_credit_useful_work_factor = clamp(route_credit_useful_work_factor, 0.20, 1.0)
                passive_route_credit_dampening = low_congestion_adaptation_credit * (
                    1.0 - route_credit_useful_work_factor
                )
                if dispatch == "hold" or dispatched_orders <= 0.0 or route_useful_work_signal < 0.50:
                    low_congestion_hold_credit_dampening = passive_route_credit_dampening
                    low_congestion_adaptation_credit = max(
                        low_congestion_adaptation_credit - low_congestion_hold_credit_dampening,
                        0.0,
                    )
            if (
                dispatch == "dispatch"
                and route_adaptation_pressure > 0.0
                and route_candidate_useful_work_factor < 0.50
            ):
                low_congestion_no_useful_work_credit_blocked = 1.0
                low_congestion_adaptation_credit = 0.0
            route_resilience_adaptation_credit = low_congestion_adaptation_credit
            route_adaptation_reward_or_credit = low_congestion_adaptation_credit
            alternative_route_balance_adjustment = -low_congestion_balance_dampening
            reward += low_congestion_adaptation_credit - route_premium_penalty
        elif route_for_reward == "shortest":
            shortest_route_disruption_risk_penalty = (
                self.config.high_resilience_route_penalty
                * clamp(
                    (1.75 * scenario_route_disruption_pressure)
                    + (1.25 * route_cost_pressure)
                    + (0.50 * disruption),
                    0.0,
                    1.0,
                )
            )
            shortest_relief_observed_profile_support = shortest_candidate_observed_profile_support
            if route_adaptation_pressure > 0.0:
                shortest_relief_pressure_factor = max(
                    shortest_candidate_moderate_pressure_support,
                    shortest_relief_observed_profile_support,
                )
                shortest_relief_service_health_factor = route_service_health_factor
            if dispatch == "dispatch":
                shortest_relief_useful_work_factor = clamp(
                    max(route_delivery_work_signal, min(route_success_count, 1.0)),
                    0.0,
                    1.0,
                )
            shortest_relief_severe_pressure_blocked = (
                1.0 if route_adaptation_pressure > 0.0 and shortest_relief_pressure_factor <= 0.0 else 0.0
            )
            shortest_relief_route_failure_blocked = 1.0 if route_failure_count > 0.0 else 0.0
            shortest_relief_customer_revisit_blocked = 1.0 if customer_revisit_blocked_count > 0.0 else 0.0
            shortest_relief_brittle_secondary_blocked = shortest_secondary_guard_active
            shortest_relief_no_useful_work_blocked = (
                1.0 if route_adaptation_pressure > 0.0 and shortest_relief_useful_work_factor <= 0.0 else 0.0
            )
            shortest_relief_impossible_dispatch_blocked = (
                1.0
                if dispatch == "dispatch"
                and (
                    infeasible_dispatch_attempt > 0.0
                    or bool(projection_info.get("blocked", False))
                    or bool(projection_info.get("discrete_blocked", False))
                )
                else 0.0
            )
            shortest_relief_allowed = (
                route_adaptation_pressure > 0.0
                and shortest_relief_pressure_factor > 0.0
                and shortest_relief_service_health_factor > 0.0
                and shortest_relief_useful_work_factor > 0.0
                and shortest_relief_route_failure_blocked <= 0.0
                and shortest_relief_customer_revisit_blocked <= 0.0
                and shortest_relief_brittle_secondary_blocked <= 0.0
                and shortest_relief_impossible_dispatch_blocked <= 0.0
            )
            if shortest_relief_allowed:
                shortest_relief_eligible = 1.0
                shortest_moderate_pressure_relief = min(
                    shortest_route_disruption_risk_penalty
                    * 0.55
                    * shortest_relief_pressure_factor
                    * shortest_relief_service_health_factor
                    * shortest_relief_useful_work_factor
                    * route_credit_multiplier,
                    self.config.high_resilience_route_penalty * 0.32,
                )
            route_narrowness_penalty = max(
                shortest_route_disruption_risk_penalty - shortest_moderate_pressure_relief,
                0.0,
            )
            reward -= shortest_route_disruption_risk_penalty
            reward += shortest_moderate_pressure_relief
        candidate_alignment_blocked_no_useful_work = (
            1.0 if route_adaptation_pressure > 0.0 and route_candidate_useful_work_factor < 0.50 else 0.0
        )
        candidate_alignment_blocked_brittle_secondary = shortest_secondary_guard_active
        candidate_alignment_blocked_route_failure = 1.0 if route_failure_count > 0.0 else 0.0
        candidate_alignment_blocked_customer_revisit = 1.0 if customer_revisit_blocked_count > 0.0 else 0.0
        candidate_alignment_blocked_severe_pressure = (
            1.0
            if route_for_reward == "shortest"
            and route_adaptation_pressure > 0.0
            and shortest_candidate_pressure_support <= 0.0
            else 0.0
        )
        candidate_alignment_blocked_impossible_dispatch = (
            1.0
            if dispatch == "dispatch"
            and (
                infeasible_dispatch_attempt > 0.0
                or bool(projection_info.get("blocked", False))
                or bool(projection_info.get("discrete_blocked", False))
            )
            else 0.0
        )
        candidate_alignment_blocked_service_lateness = 1.0 if route_candidate_service_factor < 0.50 else 0.0
        candidate_alignment_eligible = (
            1.0
            if route_adaptation_pressure > 0.0
            and selected_route_candidate_score > 0.0
            and selected_route_candidate_near_best > 0.0
            and selected_route_candidate_action_gate > 0.0
            and candidate_alignment_blocked_no_useful_work <= 0.0
            and candidate_alignment_blocked_brittle_secondary <= 0.0
            and candidate_alignment_blocked_route_failure <= 0.0
            and candidate_alignment_blocked_customer_revisit <= 0.0
            and candidate_alignment_blocked_severe_pressure <= 0.0
            and candidate_alignment_blocked_impossible_dispatch <= 0.0
            and candidate_alignment_blocked_service_lateness <= 0.0
            else 0.0
        )
        if candidate_alignment_eligible > 0.0:
            candidate_gap_factor = 1.0 - clamp(selected_route_candidate_score_gap / 0.30, 0.0, 1.0)
            if route_for_reward == "shortest":
                route_label_drag = primary_fleet_cost_penalty + route_narrowness_penalty
                route_candidate_alignment_credit = min(
                    ((1.25 * route_label_drag) + 0.04)
                    * selected_route_candidate_score
                    * route_candidate_service_factor
                    * route_candidate_useful_work_factor
                    * route_credit_multiplier,
                    self.config.primary_fleet_penalty * 1.50,
                )
                if action24_25_candidate_credit_allowed > 0.0:
                    remaining_shortest_risk = max(
                        shortest_route_disruption_risk_penalty - shortest_moderate_pressure_relief,
                        0.0,
                    )
                    route_candidate_alignment_credit = min(
                        route_candidate_alignment_credit,
                        remaining_shortest_risk * 0.45,
                        self.config.high_resilience_route_penalty * 0.24,
                    )
            else:
                route_candidate_alignment_credit = min(
                    0.004
                    * selected_route_candidate_score
                    * candidate_gap_factor
                    * candidate_gap_factor
                    * route_candidate_service_factor
                    * route_candidate_useful_work_factor
                    * route_credit_multiplier,
                    0.006,
                )
            reward += route_candidate_alignment_credit
        mismatch_gap_factor = _smoothstep_unit((selected_route_candidate_score_gap - 0.02) / 0.08)
        passive_or_no_work_route = dispatch == "hold" or dispatched_orders <= 0.0 or route_candidate_useful_work_factor < 0.50
        low_throughput_mismatch = (
            selected_route_candidate_near_best <= 0.0
            and route_candidate_useful_work_factor < 0.50
        )
        if (
            route_adaptation_pressure > 0.0
            and dispatch != "hold"
            and selected_route_candidate_score_gap > 0.0
            and (passive_or_no_work_route or low_throughput_mismatch)
            and candidate_alignment_blocked_route_failure <= 0.0
            and candidate_alignment_blocked_customer_revisit <= 0.0
            and candidate_alignment_blocked_impossible_dispatch <= 0.0
        ):
            mismatch_pressure = max(
                selected_route_candidate_score_gap,
                0.20 if route_candidate_useful_work_factor < 0.50 else 0.0,
            )
            route_candidate_mismatch_penalty = min(
                0.14
                * mismatch_pressure
                * route_candidate_service_factor
                * (0.50 + (0.50 * route_adaptation_pressure))
                * (1.0 - (0.50 * route_candidate_useful_work_factor)),
                0.06,
            )
            reward -= route_candidate_mismatch_penalty
        dqn_delivery_credit_blocked_no_current_dispatch_work = (
            1.0
            if dispatch == "dispatch" and delivered_delta > 0 and current_dispatch_work_signal <= 0.0
            else 0.0
        )
        dqn_delivery_credit = (
            0.10 * min(delivered_delta, 3)
            if dispatch == "dispatch" and current_dispatch_work_signal > 0.0
            else 0.0
        )
        reward += dqn_delivery_credit
        return {
            "dqn_local": float(reward),
            "current_dispatch_work_count": float(current_dispatch_work_count),
            "current_dispatch_work_signal": float(current_dispatch_work_signal),
            "dispatch_feasibility_credit": float(dispatch_feasibility_credit),
            "dispatch_progress_credit": float(dispatch_progress_credit),
            "dispatch_progress_quality_factor": float(dispatch_progress_quality_factor),
            "dispatch_feasibility_quality_factor": float(dispatch_feasibility_quality_factor),
            "demand_dispatch_quality_penalty": float(demand_dispatch_quality_penalty),
            "demand_service_collapse_penalty": float(demand_service_collapse_penalty),
            "demand_action_family_penalty": float(demand_action_family_penalty),
            "demand_operational_degradation": float(demand_operational_degradation),
            "demand_dispatch_pressure": float(demand_dispatch_pressure),
            "useful_dispatch_timing_pressure": float(useful_dispatch_timing_pressure),
            "useful_dispatch_lateness_urgency_credit": float(useful_dispatch_lateness_urgency_credit),
            "useful_dispatch_opportunity_pressure": float(useful_dispatch_opportunity_pressure),
            "useful_dispatch_opportunity_missed": float(useful_dispatch_opportunity_missed),
            "hold_under_lateness_pressure": float(hold_under_lateness_pressure),
            "hold_under_lateness_pressure_penalty": float(hold_under_lateness_pressure_penalty),
            "hold_timing_degradation_pressure": float(hold_timing_degradation_pressure),
            "secondary_fleet_lateness_exposure": float(secondary_fleet_lateness_exposure),
            "low_feasible_dispatch_factor": float(low_feasible_dispatch_factor),
            "projected_dispatch_exposure": float(projected_dispatch_exposure),
            "already_assigned_exposure": float(already_assigned_exposure),
            "no_vehicle_exposure": float(no_vehicle_exposure),
            "action_family_concentration_proxy": float(action_family_concentration_proxy),
            "successful_dispatch_count_capped": float(successful_dispatch_count_capped),
            "unnecessary_dispatch_penalty": float(unnecessary_dispatch_penalty),
            "no_unassigned_dispatch_exposure": float(no_unassigned_dispatch_exposure),
            "no_current_or_unassigned_dispatch_penalty": float(
                no_current_or_unassigned_dispatch_penalty
            ),
            "infeasible_dispatch_penalty": float(infeasible_dispatch_penalty),
            "infeasible_dispatch_repeat_gate": float(infeasible_dispatch_repeat_gate),
            "infeasible_dispatch_repeat_exposure": float(infeasible_dispatch_repeat_exposure),
            "infeasible_dispatch_scarcity_relief": float(infeasible_dispatch_scarcity_relief),
            "hold_route_neutralized": float(hold_route_neutralized),
            "route_label_ignored_for_hold": float(route_label_ignored_for_hold),
            "selected_route_effective_for_reward": float(selected_route_effective_for_reward),
            "route_credit_blocked_hold": float(route_credit_blocked_hold),
            "candidate_alignment_blocked_hold": float(candidate_alignment_blocked_hold),
            "hold_inflight_completion_credit_blocked_for_route": float(
                hold_inflight_completion_credit_blocked_for_route
            ),
            "dqn_delivery_credit": float(dqn_delivery_credit),
            "dqn_delivery_credit_blocked_no_current_dispatch_work": float(
                dqn_delivery_credit_blocked_no_current_dispatch_work
            ),
            "route_resilience_credit": float(route_resilience_credit),
            "route_premium_penalty": float(route_premium_penalty),
            "shortest_route_disruption_risk_penalty": float(shortest_route_disruption_risk_penalty),
            "route_resilience_adaptation_credit": float(route_resilience_adaptation_credit),
            "low_congestion_adaptation_credit": float(low_congestion_adaptation_credit),
            "high_resilience_adaptation_credit": float(high_resilience_adaptation_credit),
            "route_failure_credit_suppression": float(route_failure_credit_suppression),
            "route_adaptation_reward_or_credit": float(route_adaptation_reward_or_credit),
            "low_congestion_balance_dampening": float(low_congestion_balance_dampening),
            "high_resilience_balance_support": float(high_resilience_balance_support),
            "low_congestion_no_useful_work_credit_blocked": float(
                low_congestion_no_useful_work_credit_blocked
            ),
            "high_resilience_no_useful_work_credit_blocked": float(
                high_resilience_no_useful_work_credit_blocked
            ),
            "alternative_route_balance_adjustment": float(alternative_route_balance_adjustment),
            "route_narrowness_penalty": float(route_narrowness_penalty),
            "low_congestion_hold_credit_dampening": float(low_congestion_hold_credit_dampening),
            "high_resilience_hold_credit_dampening": float(high_resilience_hold_credit_dampening),
            "route_credit_useful_work_factor": float(route_credit_useful_work_factor),
            "passive_route_credit_dampening": float(passive_route_credit_dampening),
            "shortest_moderate_pressure_relief": float(shortest_moderate_pressure_relief),
            "shortest_relief_pressure_factor": float(shortest_relief_pressure_factor),
            "shortest_relief_observed_profile_support": float(shortest_relief_observed_profile_support),
            "shortest_relief_service_health_factor": float(shortest_relief_service_health_factor),
            "shortest_relief_useful_work_factor": float(shortest_relief_useful_work_factor),
            "shortest_relief_eligible": float(shortest_relief_eligible),
            "shortest_relief_severe_pressure_blocked": float(shortest_relief_severe_pressure_blocked),
            "shortest_relief_route_failure_blocked": float(shortest_relief_route_failure_blocked),
            "shortest_relief_customer_revisit_blocked": float(shortest_relief_customer_revisit_blocked),
            "shortest_relief_brittle_secondary_blocked": float(shortest_relief_brittle_secondary_blocked),
            "shortest_relief_no_useful_work_blocked": float(shortest_relief_no_useful_work_blocked),
            "shortest_relief_impossible_dispatch_blocked": float(shortest_relief_impossible_dispatch_blocked),
            "route_adaptation_pressure": float(route_adaptation_pressure),
            "route_disruption_reliability_pressure": float(route_disruption_reliability_pressure),
            "route_congestion_cost_pressure": float(route_congestion_cost_pressure),
            "route_pressure_reliability_share": float(route_pressure_reliability_share),
            "route_pressure_congestion_share": float(route_pressure_congestion_share),
            "route_pressure_balance": float(route_pressure_balance),
            "route_alt_pressure_imbalance": float(route_alt_pressure_imbalance),
            "shortest_route_candidate_score": float(shortest_route_candidate_score),
            "low_congestion_route_candidate_score": float(low_congestion_route_candidate_score),
            "high_resilience_route_candidate_score": float(high_resilience_route_candidate_score),
            "selected_route_candidate_score": float(selected_route_candidate_score),
            "best_route_candidate_score": float(best_route_candidate_score),
            "selected_route_candidate_score_gap": float(selected_route_candidate_score_gap),
            "selected_route_score_gap": float(selected_route_candidate_score_gap),
            "selected_route_candidate_near_best": float(selected_route_candidate_near_best),
            "selected_route_candidate_action_gate": float(selected_route_candidate_action_gate),
            "route_candidate_alignment_credit": float(route_candidate_alignment_credit),
            "route_candidate_mismatch_penalty": float(route_candidate_mismatch_penalty),
            "candidate_alignment_eligible": float(candidate_alignment_eligible),
            "candidate_alignment_blocked_no_useful_work": float(candidate_alignment_blocked_no_useful_work),
            "candidate_alignment_blocked_brittle_secondary": float(candidate_alignment_blocked_brittle_secondary),
            "candidate_alignment_blocked_route_failure": float(candidate_alignment_blocked_route_failure),
            "candidate_alignment_blocked_customer_revisit": float(candidate_alignment_blocked_customer_revisit),
            "candidate_alignment_blocked_severe_pressure": float(candidate_alignment_blocked_severe_pressure),
            "candidate_alignment_blocked_impossible_dispatch": float(candidate_alignment_blocked_impossible_dispatch),
            "candidate_alignment_blocked_service_lateness": float(candidate_alignment_blocked_service_lateness),
            "shortest_secondary_brittle_risk": float(shortest_secondary_brittle_risk),
            "shortest_secondary_safe_useful_exception": float(shortest_secondary_safe_useful_exception),
            "shortest_secondary_guard_active": float(shortest_secondary_guard_active),
            "shortest_secondary_guard_relaxed_safe": float(shortest_secondary_guard_relaxed_safe),
            "action24_25_candidate_credit_allowed": float(action24_25_candidate_credit_allowed),
            "action24_25_candidate_credit_blocked_reason": float(action24_25_candidate_credit_blocked_reason),
            "route_candidate_operational_health_factor": float(route_candidate_operational_health_factor),
            "route_candidate_service_factor": float(route_candidate_service_factor),
            "route_candidate_useful_work_factor": float(route_candidate_useful_work_factor),
            "route_delivery_work_signal": float(route_delivery_work_signal),
            "primary_fleet_cost_penalty": float(primary_fleet_cost_penalty),
            "premium_sla_fleet_justification_credit": float(premium_sla_fleet_justification_credit),
            "premium_primary_adaptation_credit": float(premium_primary_adaptation_credit),
            "primary_fleet_timing_justification_credit": float(primary_fleet_timing_justification_credit),
            "premium_sla_fleet_credit_blocked_no_current_work": float(
                premium_sla_fleet_credit_blocked_no_current_work
            ),
            "premium_primary_adaptation_credit_blocked_no_current_work": float(
                premium_primary_adaptation_credit_blocked_no_current_work
            ),
            "premium_fleet_credit_allowed": float(premium_fleet_credit_allowed),
            "premium_primary_underuse_penalty": float(premium_primary_underuse_penalty),
            "premium_fleet_justification": float(premium_fleet_justification),
            "scenario_route_disruption_pressure": float(scenario_route_disruption_pressure),
            "route_cost_pressure": float(route_cost_pressure),
            "premium_sla_pressure": float(premium_sla_pressure),
            "route_stress": float(route_stress),
            "emergency_procurement_penalty": float(emergency_procurement_penalty),
            "emergency_procurement_justification_credit": float(emergency_procurement_justification_credit),
            "emergency_inventory_shortfall": float(inventory_shortfall),
            "emergency_crisis": float(crisis),
            "emergency_gate": float(emergency_gate),
            "emergency_useful_factor": float(emergency_useful_factor),
            "emergency_cost_norm": float(emergency_cost_norm),
            "pre_crisis_pressure": float(pre_crisis_pressure),
            "pending_work_pressure": float(pending_work_pressure),
            "lateness_risk": float(lateness_risk),
            "raw_lateness_risk": float(raw_lateness_risk),
            "urgent_ratio": float(urgent_ratio),
            "raw_urgent_ratio": float(raw_urgent_ratio),
            "backlog_age_pressure": float(backlog_age_pressure),
            "hold_gate": float(hold_gate),
            "hold_penalty": float(hold_penalty),
            "current_inaction_pressure": float(current_inaction_pressure),
            "previous_inaction_pressure": float(clamp(self._previous_inaction_pressure, 0.0, 1.0)),
            "delay_abuse_penalty": float(delay_abuse_penalty),
            **order_flow_metrics.as_reward_components(),
        }

    def _economic_penalty(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        projection_info: Mapping[str, Any],
    ) -> float:
        action = projection_info.get("action", {})
        action_payload = action if isinstance(action, Mapping) else {}
        delivered_delta = max(0, _delivered_count(current) - _delivered_count(previous))
        cost_delta = max(
            0.0,
            float(current.get("total_transport_cost", 0.0)) - float(previous.get("total_transport_cost", 0.0)),
        )
        cost_per_delivery = cost_delta / max(delivered_delta, 1)
        penalty = clamp(cost_per_delivery / self.config.transport_cost_penalty_scale, 0.0, 2.0) * 0.45

        pending_ratio = self._pending_order_ratio(current)
        penalty += pending_ratio * 0.20

        reorder = str(action_payload.get("reorder", "none"))
        inventory_pressure = self._average_inventory_pressure(current)
        if reorder == "emergency":
            penalty += self.config.emergency_reorder_penalty
            if inventory_pressure < 0.50:
                penalty += self.config.emergency_reorder_penalty
        elif reorder == "aggressive" and inventory_pressure < 0.35:
            penalty += self.config.emergency_reorder_penalty * 0.50

        disruption = clamp(float(current.get("disruption_score", 0.0)), 0.0, 1.0)
        safety_potential_value = clamp(float(current.get("network_safety_potential", 0.0)), 0.0, 1.0)
        stress_discount = clamp(max(disruption, safety_potential_value), 0.0, 1.0)

        mode = str(action_payload.get("mode", ""))
        if mode == "primary_fleet":
            penalty += self.config.primary_fleet_penalty * (1.0 - pending_ratio)

        route = str(action_payload.get("route", ""))
        if route == "high_resilience":
            penalty += self.config.high_resilience_route_penalty * (1.0 - stress_discount)
        elif route == "low_congestion":
            penalty += self.config.low_congestion_route_penalty * (1.0 - disruption)

        if int(projection_info.get("rolling_demand_orders", 0)) == 0 and delivered_delta == 0:
            penalty += 0.02
        return penalty

    def _generate_rolling_demand(self) -> tuple[UUID, ...]:
        if not self.config.rolling_demand_enabled:
            return ()
        if not self.simulation.hubs or not self.simulation.customers or not self.simulation.products:
            return ()
        if self.simulation.rng.random() > self.config.rolling_demand_probability:
            return ()

        max_orders = self.config.rolling_demand_max_orders_per_step
        arrivals = self.simulation.rng.randint(1, max_orders)
        created: list[UUID] = []
        product_ids = tuple(self.simulation.products)
        hub_ids = tuple(self.simulation.hubs)
        customers = tuple(self.simulation.customers.values())

        for _ in range(arrivals):
            product_id = self.simulation.rng.choice(product_ids)
            hub_id = self.simulation.rng.choice(hub_ids)
            customer = self.simulation.rng.choice(customers)
            quantity = self.simulation.rng.uniform(
                self.config.rolling_demand_min_units,
                self.config.rolling_demand_max_units,
            )
            urgent = self.simulation.rng.random() < self.config.urgent_order_probability
            due_window = (
                self.simulation.rng.uniform(1_800.0, 7_200.0) * self.config.urgent_due_window_multiplier
                if urgent
                else self.simulation.rng.uniform(7_200.0, 28_800.0)
            )
            order = Order(
                order_id=uuid4(),
                customer_id=customer.customer_id,
                origin_node_id=hub_id,
                destination_node_id=customer.customer_id,
                lines=[OrderLine(product_id=product_id, quantity_units=quantity)],
                release_time=float(self.simulation.env.now),
                due_time=float(self.simulation.env.now + due_window),
                metadata={
                    "rolling_demand": True,
                    "urgency": "urgent" if urgent else "standard",
                    "created_step": self._step_count,
                },
            )
            self.simulation.add_order(order)
            self._update_rolling_demand_forecast(customer.customer_id, product_id, quantity)
            self.simulation.state.event_log.append(
                {
                    "time": self.simulation.env.now,
                    "event": "rolling_order_created",
                    "order_id": str(order.order_id),
                    "customer_id": str(customer.customer_id),
                    "origin_node_id": str(hub_id),
                    "destination_node_id": str(customer.customer_id),
                    "product_id": str(product_id),
                    "quantity_units": quantity,
                    "urgency": order.metadata["urgency"],
                    "due_time": order.due_time,
                }
            )
            created.append(order.order_id)
        return tuple(created)

    def _update_rolling_demand_forecast(self, node_id: UUID, product_id: UUID, quantity: float) -> None:
        try:
            position = self.simulation.inventory_network.get_position(node_id, product_id)
        except KeyError:
            return
        position.update_forecast(quantity, alpha=0.35)

    @staticmethod
    def _pending_order_ratio(
        snapshot: Mapping[str, Any],
        masked_order_ids: set[str] | None = None,
    ) -> float:
        orders = snapshot.get("orders", {})
        if not isinstance(orders, Mapping) or not orders:
            return 0.0
        excluded = masked_order_ids or set()
        eligible_orders = {
            str(order_id): order
            for order_id, order in orders.items()
            if str(order_id) not in excluded
        }
        if not eligible_orders:
            return 0.0
        pending = sum(
            1
            for order in eligible_orders.values()
            if isinstance(order, Mapping) and order.get("status") != "delivered"
        )
        return clamp(pending / len(eligible_orders), 0.0, 1.0)

    @staticmethod
    def _average_inventory_pressure(snapshot: Mapping[str, Any]) -> float:
        inventory = snapshot.get("inventory", {})
        if not isinstance(inventory, Mapping) or not inventory:
            return 0.0
        values = [
            float(row.get("inventory_pressure", 0.0))
            for row in inventory.values()
            if isinstance(row, Mapping)
        ]
        return clamp(sum(values) / len(values), 0.0, 1.0) if values else 0.0

    @staticmethod
    def _average_backlog_pressure(snapshot: Mapping[str, Any]) -> float:
        inventory = snapshot.get("inventory", {})
        if not isinstance(inventory, Mapping) or not inventory:
            return 0.0
        values = [
            float(row.get("backlog_pressure", 0.0))
            for row in inventory.values()
            if isinstance(row, Mapping)
        ]
        return clamp(sum(values) / len(values), 0.0, 1.0) if values else 0.0

    @staticmethod
    def _backlog_age_pressure(
        snapshot: Mapping[str, Any],
        masked_order_ids: set[str] | None = None,
    ) -> float:
        orders = snapshot.get("orders", {})
        current_time = _safe_float(snapshot.get("time"), 0.0)
        if not isinstance(orders, Mapping) or not orders:
            return 0.0
        excluded = masked_order_ids or set()
        ages: list[float] = []
        for order_id, order in orders.items():
            if str(order_id) in excluded:
                continue
            if not isinstance(order, Mapping) or order.get("status") == "delivered":
                continue
            release_time = _safe_float(order.get("release_time"), current_time)
            ages.append(clamp((current_time - release_time) / 28_800.0, 0.0, 1.0))
        return clamp(sum(ages) / len(ages), 0.0, 1.0) if ages else 0.0

    @staticmethod
    def _stockout_risk(snapshot: Mapping[str, Any]) -> float:
        inventory = snapshot.get("inventory", {})
        if not isinstance(inventory, Mapping) or not inventory:
            return 0.0
        risks: list[float] = []
        for row in inventory.values():
            if not isinstance(row, Mapping):
                continue
            on_hand = max(0.0, _safe_float(row.get("on_hand_units"), 0.0))
            in_transit = max(0.0, _safe_float(row.get("in_transit_units"), 0.0))
            reserved = max(0.0, _safe_float(row.get("reserved_units"), 0.0))
            backlog = max(0.0, _safe_float(row.get("backlog_units"), 0.0))
            safety_stock = max(0.0, _safe_float(row.get("safety_stock_units"), 0.0))
            denominator = max(on_hand + in_transit + safety_stock, 1.0)
            risks.append(clamp((reserved + backlog) / denominator, 0.0, 1.0))
        return clamp(sum(risks) / len(risks), 0.0, 1.0) if risks else 0.0

    @staticmethod
    def _inventory_coverage(snapshot: Mapping[str, Any]) -> float:
        inventory = snapshot.get("inventory", {})
        if not isinstance(inventory, Mapping) or not inventory:
            return 1.0
        values: list[float] = []
        for row in inventory.values():
            if not isinstance(row, Mapping):
                continue
            on_hand = max(0.0, _safe_float(row.get("on_hand_units"), 0.0))
            in_transit = max(0.0, _safe_float(row.get("in_transit_units"), 0.0))
            backlog = max(0.0, _safe_float(row.get("backlog_units"), 0.0))
            demand_mean = max(0.0, _safe_float(row.get("demand_mean"), 0.0))
            required = max(demand_mean + backlog, 1.0)
            values.append(clamp((on_hand + in_transit) / required, 0.0, 1.0))
        return clamp(sum(values) / len(values), 0.0, 1.0) if values else 1.0

    def _safety_stock_target_gap(self) -> float:
        positions = list(self.simulation.inventory_network.positions.values())
        if not positions:
            return 0.0
        gaps = []
        for position in positions:
            target = max(position.target_safety_stock(), 1.0)
            gaps.append(clamp((target - position.safety_stock_units) / target, 0.0, 1.0))
        return clamp(sum(gaps) / len(gaps), 0.0, 1.0)

    def _demand_volatility_pressure(self) -> float:
        positions = list(self.simulation.inventory_network.positions.values())
        if not positions:
            return 0.0
        values = [
            clamp(
                position.demand_variance / max(position.demand_mean + position.demand_variance + 1.0, 1.0),
                0.0,
                1.0,
            )
            for position in positions
        ]
        return clamp(sum(values) / len(values), 0.0, 1.0)

    def _lead_time_volatility_pressure(self) -> float:
        positions = list(self.simulation.inventory_network.positions.values())
        if not positions:
            return 0.0
        values = [
            clamp(
                position.lead_time_variance / max(position.lead_time_mean + position.lead_time_variance + 1.0, 1.0),
                0.0,
                1.0,
            )
            for position in positions
        ]
        return clamp(sum(values) / len(values), 0.0, 1.0)

    def _scenario_stress_value(self, name: str, default: float) -> float:
        try:
            return float(self.scenario_stress_state.get(name, default))
        except (TypeError, ValueError):
            return default

    def _holding_cost_pressure(self) -> float:
        return clamp(
            (
                max(
                    self._scenario_stress_value("holding_cost_multiplier", 1.0),
                    self._scenario_stress_value("excess_inventory_penalty_multiplier", 1.0),
                    self._scenario_stress_value("planned_replenishment_cost_multiplier", 1.0),
                )
                - 1.0
            )
            / 3.0,
            0.0,
            1.0,
        )

    def _stockout_penalty_pressure(self) -> float:
        return clamp(
            (
                max(
                    self._scenario_stress_value("stockout_penalty_multiplier", 1.0),
                    self._scenario_stress_value("inventory_shortfall_penalty_multiplier", 1.0),
                )
                - 1.0
            )
            / 3.0,
            0.0,
            1.0,
        )

    def _scenario_supplier_delay_pressure(self) -> float:
        return clamp(self._scenario_stress_value("supplier_delay_probability", 0.0), 0.0, 1.0)

    def _scenario_lead_time_volatility_pressure(self) -> float:
        supplier_delay_pressure = self._scenario_supplier_delay_pressure()
        return clamp(
            max(
                (self._scenario_stress_value("lead_time_mean_multiplier", 1.0) - 1.0) / 3.0,
                (self._scenario_stress_value("lead_time_variance_multiplier", 1.0) - 1.0) / 4.0,
                supplier_delay_pressure,
            ),
            0.0,
            1.0,
        )

    def _capacity_shock_pressure(self) -> float:
        return clamp(
            max(
                1.0 - self._scenario_stress_value("vehicle_availability_multiplier", 1.0),
                1.0 - self._scenario_stress_value("fleet_capacity_multiplier", 1.0),
                self._scenario_stress_value("capacity_shock_severity", 0.0),
            ),
            0.0,
            1.0,
        )

    def _scenario_route_disruption_pressure(self) -> float:
        return clamp(
            max(
                self._scenario_stress_value("route_disruption_probability", 0.0),
                (self._scenario_stress_value("congestion_multiplier", 1.0) - 1.0) / 3.0,
                (self._scenario_stress_value("shortest_route_risk_multiplier", 1.0) - 1.0) / 3.0,
                (self._scenario_stress_value("disruption_risk_multiplier", 1.0) - 1.0) / 3.0,
            ),
            0.0,
            1.0,
        )

    def _route_cost_pressure(self) -> float:
        return clamp(
            (
                max(
                    self._scenario_stress_value("traversal_cost_multiplier", 1.0),
                    self._scenario_stress_value("route_fleet_cost_multiplier", 1.0),
                )
                - 1.0
            )
            / 3.0,
            0.0,
            1.0,
        )

    def _premium_sla_pressure(self) -> float:
        return clamp(
            max(
                self._scenario_stress_value("premium_sla_ratio", 0.0),
                max(0.0, self._scenario_stress_value("service_target", 0.95) - 0.95) / 0.05,
                max(0.0, 1.0 - self._scenario_stress_value("urgent_due_window_multiplier", 1.0)),
                (self._scenario_stress_value("premium_lateness_penalty_multiplier", 1.0) - 1.0) / 3.0,
            ),
            0.0,
            1.0,
        )

    @staticmethod
    def _urgent_pending_ratio(snapshot: Mapping[str, Any]) -> float:
        orders = snapshot.get("orders", {})
        current_time = _safe_float(snapshot.get("time"), 0.0)
        if not isinstance(orders, Mapping) or not orders:
            return 0.0
        pending = 0
        urgent = 0
        for order in orders.values():
            if not isinstance(order, Mapping) or order.get("status") == "delivered":
                continue
            pending += 1
            due_time = _safe_float(order.get("due_time"), current_time)
            if due_time - current_time <= 7_200.0:
                urgent += 1
        return clamp(urgent / max(pending, 1), 0.0, 1.0)

    @staticmethod
    def _lateness_risk(snapshot: Mapping[str, Any]) -> float:
        orders = snapshot.get("orders", {})
        current_time = _safe_float(snapshot.get("time"), 0.0)
        if not isinstance(orders, Mapping) or not orders:
            return 0.0
        risks: list[float] = []
        for order in orders.values():
            if not isinstance(order, Mapping) or order.get("status") == "delivered":
                continue
            due_time = _safe_float(order.get("due_time"), current_time)
            remaining = due_time - current_time
            risks.append(clamp(1.0 - (remaining / 28_800.0), 0.0, 1.0))
        return clamp(sum(risks) / len(risks), 0.0, 1.0) if risks else 0.0

    def _network_congestion_score(self) -> float:
        values = list(self.simulation.routing_physics.congestion_by_arc.values())
        if not values:
            return 0.0
        return clamp(sum(clamp(value, 0.0, 1.0) for value in values) / len(values), 0.0, 1.0)

    def _macro_capacity_pressure(self) -> float:
        states = list(self.simulation.capacity_states.values())
        if not states:
            return 0.0
        return clamp(max(state.capacity_pressure for state in states), 0.0, 1.0)

    @staticmethod
    def _continuous_action_payload(projection_info: Mapping[str, Any]) -> Mapping[str, Any]:
        action = projection_info.get("action", {})
        if not isinstance(action, Mapping):
            return {}
        continuous = action.get("continuous")
        if isinstance(continuous, Mapping):
            return continuous
        if "speed_multiplier" in action or "capacity_buffer_fraction" in action:
            return action
        return {}

    @staticmethod
    def _discrete_action_payload(projection_info: Mapping[str, Any]) -> Mapping[str, Any]:
        action = projection_info.get("action", {})
        if not isinstance(action, Mapping):
            return {}
        discrete = action.get("discrete")
        if isinstance(discrete, Mapping):
            return discrete
        if "dispatch" in action or "route" in action or "mode" in action:
            return action
        return {}

    def _safety_context(self) -> dict[str, float]:
        disruption_score = self.simulation.disruptions.disruption_score()
        safety_potential = self.simulation.inventory_network.network_safety_potential(
            disruption_score=disruption_score
        )
        vehicle_util = (
            sum(vehicle.utilization for vehicle in self.simulation.vehicles.values())
            / max(len(self.simulation.vehicles), 1)
        )
        macro_capacity_pressure = self._macro_capacity_pressure()
        positions = list(self.simulation.inventory_network.positions.values())
        backlog_pressure = (
            sum(position.backlog_pressure for position in positions) / len(positions)
            if positions
            else 0.0
        )
        return {
            "safety_potential": clamp(float(safety_potential), 0.0, 1.0),
            "capacity_utilization": clamp(max(macro_capacity_pressure, vehicle_util * 0.25), 0.0, 1.0),
            "backlog_pressure": clamp(backlog_pressure, 0.0, 1.0),
        }

    def _info(
        self,
        *,
        projected: bool,
        blocked: bool,
        reasons: tuple[str, ...],
        projected_action: Mapping[str, Any] | None = None,
        raw_action: Any | None = None,
        rolling_demand_orders: int = 0,
        termination_reason: str | None = None,
        reward_components: Mapping[str, float] | None = None,
        snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        info = {
            "step": self._step_count,
            "simulation_time": self.simulation.env.now,
            "raw_action": raw_action,
            "projected": projected,
            "blocked": blocked,
            "projection_reasons": reasons,
            "projected_action": dict(projected_action or {}),
            "rolling_demand_orders": rolling_demand_orders,
            "termination_reason": termination_reason,
            "reward_components": dict(reward_components or {}),
        }
        if self.config.info_mode == "full":
            info["snapshot"] = copy.deepcopy(snapshot) if snapshot is not None else self.simulation.snapshot()
        return info

    def _default_simulation_factory(self, seed: int | None) -> DigitalTwinSimulation:
        simulation = DigitalTwinSimulation(
            SimulationConfig(decision_interval=self.config.decision_interval, random_seed=seed)
        )
        product_id = uuid4()
        hub_id = uuid4()
        customer_ids = [uuid4() for _ in range(3)]
        vehicle_specs = (
            (uuid4(), VehicleTier.PRIMARY, 240.0, 16.0, 70.0, 0.003),
            (uuid4(), VehicleTier.SECONDARY, 120.0, 14.0, 40.0, 0.002),
        )

        simulation.add_product(Product(product_id=product_id, sku="default-sku", unit_weight_kg=2.0, unit_volume_m3=0.02))
        hub = Hub(
            hub_id=hub_id,
            name="default-hub",
            location=GeoPoint(latitude=41.0082, longitude=28.9784),
            echelon_level=1,
            storage_capacity_units=10_000.0,
            throughput_units_per_time=500.0,
        )
        simulation.add_hub(hub, initial_inventory_units=1_000.0, dock_count=2)
        simulation.add_inventory_position(
            InventoryPosition(
                node_id=hub_id,
                product_id=product_id,
                echelon_level=1,
                on_hand_units=1_000.0,
                safety_stock_units=250.0,
                demand_mean=120.0,
                demand_variance=25.0,
                lead_time_mean=2.0,
                lead_time_variance=0.5,
            )
        )

        for index, customer_id in enumerate(customer_ids):
            customer = Customer(
                customer_id=customer_id,
                name=f"default-customer-{index}",
                location=GeoPoint(latitude=41.02 + index * 0.01, longitude=29.00 + index * 0.015),
                demand_units={product_id: 20.0 + index * 5.0},
                time_window_start=0.0,
                time_window_end=86_400.0,
                service_time=120.0,
            )
            simulation.add_customer(customer)
            simulation.add_inventory_position(
                InventoryPosition(
                    node_id=customer_id,
                    product_id=product_id,
                    echelon_level=2,
                    demand_mean=customer.total_demand_units,
                    demand_variance=8.0,
                    lead_time_mean=1.0,
                    lead_time_variance=0.25,
                )
            )
            simulation.add_route_arc(
                make_arc_from_points(
                    origin_node_id=hub_id,
                    destination_node_id=customer_id,
                    origin=hub.location,
                    destination=customer.location,
                    nominal_speed_mps=14.0,
                    capacity_units=200.0,
                )
            )
            simulation.add_order(
                Order(
                    order_id=uuid4(),
                    customer_id=customer_id,
                    origin_node_id=hub_id,
                    destination_node_id=customer_id,
                    lines=[OrderLine(product_id=product_id, quantity_units=customer.total_demand_units)],
                    release_time=0.0,
                    due_time=86_400.0,
                )
            )

        for vehicle_id, tier, capacity_units, nominal_speed_mps, fixed_cost, variable_cost in vehicle_specs:
            simulation.add_vehicle(
                Vehicle(
                    vehicle_id=vehicle_id,
                    asset_kind=AssetKind.VEHICLE,
                    tier=tier,
                    capacity_units=capacity_units,
                    capacity_weight_kg=10_000.0,
                    capacity_volume_m3=80.0,
                    nominal_speed_mps=nominal_speed_mps,
                    fixed_usage_cost=fixed_cost,
                    transport_cost_per_meter=variable_cost,
                    current_node_id=hub_id,
                    current_location=hub.location,
                )
            )
        return simulation


def _delivered_count(snapshot: Mapping[str, Any]) -> int:
    orders = snapshot.get("orders", {})
    if not isinstance(orders, Mapping):
        return 0
    return sum(1 for order in orders.values() if isinstance(order, Mapping) and order.get("status") == "delivered")


def _coerce_discrete_action(action: Any) -> int:
    if isinstance(action, Mapping):
        value = action.get("discrete", action.get("dqn"))
        if value is None:
            raise ValueError("discrete action mapping must contain discrete or dqn.")
        return _coerce_discrete_action(value)
    array = np.asarray(action)
    if array.size != 1:
        raise ValueError(f"discrete action must be scalar-like, got shape {array.shape}.")
    return int(array.reshape(-1)[0])


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return {str(item) for item in value}
    return {str(value)}


def _event_float_sum_since(
    snapshot: Mapping[str, Any],
    *,
    after_time: float,
    event_name: str,
    key: str,
) -> float:
    events = snapshot.get("event_log", ())
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        return 0.0
    return float(_event_float_sums_since(events, after_time=after_time, event_name=event_name).get(str(key), 0.0))


def _event_float_sums_since(
    events: Sequence[Any],
    *,
    after_time: float,
    event_name: str,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("event") != event_name:
            continue
        if _safe_float(event.get("time"), 0.0) <= after_time:
            continue
        for field, value in event.items():
            if field in {"event", "time"}:
                continue
            totals[str(field)] = totals.get(str(field), 0.0) + max(0.0, _safe_float(value, 0.0))
    return totals


def _empty_order_flow_metrics() -> OrderFlowMetrics:
    return OrderFlowMetrics(
        eligible_orders=0,
        delivered_orders=0,
        pending_orders=0,
        unassigned_orders=0,
        in_transit_orders=0,
        healthy_in_transit_orders=0,
        urgent_unassigned_orders=0,
        unassigned_backlog_pressure=0.0,
        in_transit_pressure=0.0,
        due_soon_pressure=0.0,
        true_lateness_pressure=0.0,
        healthy_in_transit_ratio=0.0,
        urgent_unassigned_ratio=0.0,
        actionable_lateness_risk=0.0,
        actionable_flow_pressure=0.0,
    )


def _empty_dispatch_diagnostics() -> dict[str, float]:
        return {
            "dispatch_attempted": 0.0,
            "dispatch_success_count": 0.0,
            "dispatch_no_unassigned_orders": 0.0,
            "dispatch_no_vehicle_available": 0.0,
            "dispatch_route_failure": 0.0,
            "dispatch_capacity_rejected": 0.0,
            "dispatch_already_assigned_count": 0.0,
            "route_feasibility_failed_count": 0.0,
            "failed_route_vehicle_rollback_count": 0.0,
            "customer_revisit_blocked_count": 0.0,
            "available_vehicle_count": 0.0,
            "dispatchable_order_count": 0.0,
            "feasible_dispatch_opportunity": 0.0,
            "feasible_dispatch_ratio": 0.0,
            "vehicle_availability_pressure": 0.0,
            "infeasible_dispatch_attempt": 0.0,
            "infeasible_dispatch_attempt_window_count": 0.0,
            "repeated_infeasible_dispatch_attempt_rate": 0.0,
            "dispatch_success_rate": 0.0,
            "no_vehicle_available_rate": 0.0,
            "already_assigned_rate": 0.0,
            "macro_dispatch_release": 0.0,
            "raw_macro_dispatch_budget": 1.0,
            "feasible_macro_dispatch_cap": 1.0,
            "macro_dispatch_budget": 1.0,
            "dqn_dispatch_requested": 0.0,
    }


def _mean_or_zero(values: Sequence[float]) -> float:
    return clamp(sum(values) / len(values), 0.0, 1.0) if values else 0.0


def _vehicle_baseline_speed_mps(vehicle: Vehicle) -> float:
    baseline = vehicle.baseline_speed_mps
    if baseline is None:
        metadata_baseline = vehicle.metadata.get("baseline_speed_mps")
        baseline = _safe_float(metadata_baseline, vehicle.nominal_speed_mps)
        vehicle.baseline_speed_mps = baseline
    baseline = max(0.1, float(baseline))
    vehicle.metadata["baseline_speed_mps"] = baseline
    return baseline


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _smoothstep_unit(value: float) -> float:
    x = clamp(value, 0.0, 1.0)
    return float((x * x) * (3.0 - (2.0 * x)))


def _is_dispatchable_order(order: Any) -> bool:
    return (
        getattr(order, "assigned_vehicle_id", None) is None
        and getattr(order, "delivery_time", None) is None
        and getattr(order, "status", None) != ShipmentStatus.DELAYED
    )
