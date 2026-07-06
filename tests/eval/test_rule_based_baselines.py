from __future__ import annotations

import unittest

from src.eval.rule_based_baselines import (
    ACTION_DISPATCH_LOW_CONGESTION_PRIMARY_NONE,
    ACTION_DISPATCH_LOW_CONGESTION_SECONDARY_NONE,
    ACTION_DISPATCH_SHORTEST_PRIMARY_CONSERVATIVE,
    ACTION_DISPATCH_SHORTEST_PRIMARY_EMERGENCY,
    ACTION_DISPATCH_SHORTEST_PRIMARY_NONE,
    ACTION_DISPATCH_SHORTEST_SECONDARY_NONE,
    ACTION_HOLD,
    OrderView,
    RuleBasedDecisionContext,
    build_rule_based_baselines,
    decode_action_id,
    encode_action_id,
)


def _order(
    order_id: str,
    *,
    created_at: float,
    due_time: float,
    premium: bool = False,
    urgent: bool = False,
) -> OrderView:
    return OrderView(
        order_id=order_id,
        created_at=created_at,
        due_time=due_time,
        total_units=10.0,
        premium=premium,
        urgent=urgent,
    )


def _context(
    *orders: OrderView,
    primary_vehicle_available: bool = True,
    secondary_vehicle_available: bool = True,
    route_disruption_pressure: float = 0.0,
    congestion_pressure: float = 0.0,
    stockout_risk: float = 0.0,
    holding_cost_pressure: float = 0.0,
    safety_stock_gap: float = 0.0,
    inventory_coverage: float = 1.0,
) -> RuleBasedDecisionContext:
    return RuleBasedDecisionContext(
        dispatchable_orders=orders,
        primary_vehicle_available=primary_vehicle_available,
        secondary_vehicle_available=secondary_vehicle_available,
        route_disruption_pressure=route_disruption_pressure,
        congestion_pressure=congestion_pressure,
        stockout_risk=stockout_risk,
        holding_cost_pressure=holding_cost_pressure,
        safety_stock_gap=safety_stock_gap,
        inventory_coverage=inventory_coverage,
    )


class RuleBasedBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baselines = build_rule_based_baselines()

    def test_action_encode_decode_table_covers_actions_24_and_32(self) -> None:
        action24 = decode_action_id(ACTION_DISPATCH_SHORTEST_SECONDARY_NONE).as_dict()
        action32 = decode_action_id(ACTION_DISPATCH_LOW_CONGESTION_SECONDARY_NONE).as_dict()

        self.assertEqual(ACTION_DISPATCH_SHORTEST_SECONDARY_NONE, 24)
        self.assertEqual(ACTION_DISPATCH_LOW_CONGESTION_SECONDARY_NONE, 32)
        self.assertEqual(
            action24,
            {
                "dispatch": "dispatch",
                "route": "shortest",
                "mode": "secondary_fleet",
                "reorder": "none",
            },
        )
        self.assertEqual(
            action32,
            {
                "dispatch": "dispatch",
                "route": "low_congestion",
                "mode": "secondary_fleet",
                "reorder": "none",
            },
        )
        self.assertEqual(encode_action_id(decode_action_id(24)), 24)
        self.assertEqual(encode_action_id(decode_action_id(32)), 32)

    def test_fifo_picks_oldest_eligible_order(self) -> None:
        decision = self.baselines["fifo_shortest_primary_none"].select(
            _context(
                _order("newer", created_at=12.0, due_time=20.0),
                _order("oldest", created_at=3.0, due_time=50.0),
                _order("middle", created_at=7.0, due_time=10.0),
            )
        )

        self.assertEqual(decision.action_id, ACTION_DISPATCH_SHORTEST_PRIMARY_NONE)
        self.assertEqual(decision.selected_order_id, "oldest")

    def test_earliest_due_date_picks_earliest_due_order(self) -> None:
        decision = self.baselines["earliest_due_shortest_primary_none"].select(
            _context(
                _order("oldest", created_at=1.0, due_time=30.0),
                _order("earliest_due", created_at=5.0, due_time=8.0),
                _order("later_due", created_at=2.0, due_time=12.0),
            )
        )

        self.assertEqual(decision.action_id, ACTION_DISPATCH_SHORTEST_PRIMARY_NONE)
        self.assertEqual(decision.selected_order_id, "earliest_due")

    def test_premium_first_picks_premium_or_urgent_work_before_normal_work(self) -> None:
        decision = self.baselines["premium_first_shortest_primary_secondary_none"].select(
            _context(
                _order("normal_early", created_at=1.0, due_time=5.0),
                _order("premium_later", created_at=2.0, due_time=12.0, premium=True),
                _order("normal_later", created_at=3.0, due_time=7.0),
            )
        )

        self.assertEqual(decision.action_id, ACTION_DISPATCH_SHORTEST_PRIMARY_NONE)
        self.assertEqual(decision.selected_order_id, "premium_later")

    def test_low_congestion_rule_emits_action_32_or_36_under_route_pressure(self) -> None:
        secondary_decision = self.baselines["low_congestion_under_disruption"].select(
            _context(
                _order("route_job", created_at=1.0, due_time=10.0),
                primary_vehicle_available=False,
                secondary_vehicle_available=True,
                route_disruption_pressure=0.8,
                congestion_pressure=0.7,
            )
        )
        primary_decision = self.baselines["low_congestion_under_disruption"].select(
            _context(
                _order("route_job", created_at=1.0, due_time=10.0),
                primary_vehicle_available=True,
                secondary_vehicle_available=True,
                route_disruption_pressure=0.8,
                congestion_pressure=0.7,
            )
        )

        self.assertEqual(secondary_decision.action_id, ACTION_DISPATCH_LOW_CONGESTION_SECONDARY_NONE)
        self.assertEqual(primary_decision.action_id, ACTION_DISPATCH_LOW_CONGESTION_PRIMARY_NONE)
        self.assertIn(
            secondary_decision.action_id,
            {ACTION_DISPATCH_LOW_CONGESTION_SECONDARY_NONE, ACTION_DISPATCH_LOW_CONGESTION_PRIMARY_NONE},
        )

    def test_conservative_stock_threshold_reorder_emits_action_29(self) -> None:
        decision = self.baselines["conservative_stock_threshold_reorder"].select(
            _context(
                _order("stock_job", created_at=1.0, due_time=10.0),
                stockout_risk=0.35,
                safety_stock_gap=0.6,
                inventory_coverage=0.45,
            )
        )

        self.assertEqual(decision.action_id, ACTION_DISPATCH_SHORTEST_PRIMARY_CONSERVATIVE)
        self.assertEqual(decision.selected_order_id, "stock_job")

    def test_emergency_stockout_prevention_emits_action_31(self) -> None:
        decision = self.baselines["emergency_stockout_prevention_reorder"].select(
            _context(
                _order("emergency_job", created_at=1.0, due_time=10.0),
                stockout_risk=0.92,
                safety_stock_gap=0.9,
                inventory_coverage=0.1,
            )
        )

        self.assertEqual(decision.action_id, ACTION_DISPATCH_SHORTEST_PRIMARY_EMERGENCY)
        self.assertEqual(decision.selected_order_id, "emergency_job")

    def test_vehicle_scarcity_rule_uses_primary_then_secondary_fallback(self) -> None:
        primary_decision = self.baselines["vehicle_scarcity_primary_first_secondary_fallback"].select(
            _context(_order("primary_job", created_at=1.0, due_time=10.0), primary_vehicle_available=True)
        )
        secondary_decision = self.baselines["vehicle_scarcity_primary_first_secondary_fallback"].select(
            _context(
                _order("secondary_job", created_at=1.0, due_time=10.0),
                primary_vehicle_available=False,
                secondary_vehicle_available=True,
            )
        )

        self.assertEqual(primary_decision.action_id, ACTION_DISPATCH_SHORTEST_PRIMARY_NONE)
        self.assertEqual(secondary_decision.action_id, ACTION_DISPATCH_SHORTEST_SECONDARY_NONE)

    def test_high_holding_no_overstock_keeps_reorder_none(self) -> None:
        decision = self.baselines["high_holding_no_overstock"].select(
            _context(
                _order("holding_job", created_at=1.0, due_time=10.0),
                holding_cost_pressure=0.95,
                stockout_risk=0.1,
                safety_stock_gap=0.1,
                inventory_coverage=0.9,
            )
        )

        self.assertEqual(decision.action_id, ACTION_DISPATCH_SHORTEST_PRIMARY_NONE)
        self.assertEqual(decode_action_id(decision.action_id).reorder.name.lower(), "none")

    def test_invalid_or_no_work_state_returns_canonical_hold(self) -> None:
        for name, baseline in self.baselines.items():
            with self.subTest(name=name):
                decision = baseline.select(_context())
                self.assertEqual(decision.action_id, ACTION_HOLD)
                self.assertIsNone(decision.selected_order_id)

    def test_same_synthetic_context_returns_deterministic_action(self) -> None:
        baseline = self.baselines["vehicle_scarcity_primary_first_secondary_fallback"]
        context = _context(
            _order("first", created_at=5.0, due_time=20.0),
            _order("second", created_at=1.0, due_time=8.0),
            primary_vehicle_available=False,
            secondary_vehicle_available=True,
        )

        first = baseline.select(context)
        second = baseline.select(context)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
