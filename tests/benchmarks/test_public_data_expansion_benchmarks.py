from __future__ import annotations

import unittest

import numpy as np

from scripts import public_data_expansion_benchmarks as bench


class PublicDataExpansionBenchmarkTests(unittest.TestCase):
    def test_lade_summary_counts_couriers_cities_and_latency(self) -> None:
        rows = [
            {
                "package_id": "p1",
                "courier_id": "c1",
                "city": "hz",
                "accept_time": "2026-01-01 08:00:00",
                "finish_time": "2026-01-01 09:30:00",
            },
            {
                "package_id": "p2",
                "courier_id": "c2",
                "city": "hz",
                "accept_time": "2026-01-01 10:00:00",
                "finish_time": "2026-01-01 10:30:00",
            },
        ]

        summary = bench.summarize_lade_rows(rows)

        self.assertEqual(summary["package_count"], 2)
        self.assertEqual(summary["courier_count"], 2)
        self.assertEqual(summary["city_count"], 1)
        self.assertEqual(summary["accept_to_finish_latency_seconds"]["mean"], 3600.0)

    def test_planned_actual_summary_reports_sequence_deviation(self) -> None:
        rows = [
            {"route_id": "r1", "planned_sequence": "A>B>C", "actual_sequence": "A>C>B"},
            {"route_id": "r2", "planned_sequence": "A,B", "actual_sequence": "A,B"},
        ]

        summary = bench.summarize_planned_actual_rows(rows)

        self.assertEqual(summary["route_pair_count"], 2)
        self.assertEqual(summary["sequence_deviation"]["mean_position_mismatch_rate"], 1 / 3 / 2)
        self.assertEqual(summary["sequence_deviation"]["routes_with_deviation"], 1)

    def test_olist_summary_computes_late_rate_and_freight(self) -> None:
        orders = [
            {
                "order_id": "o1",
                "order_status": "delivered",
                "order_delivered_customer_date": "2026-01-03 00:00:00",
                "order_estimated_delivery_date": "2026-01-02 00:00:00",
            },
            {
                "order_id": "o2",
                "order_status": "delivered",
                "order_delivered_customer_date": "2026-01-01 00:00:00",
                "order_estimated_delivery_date": "2026-01-02 00:00:00",
            },
        ]
        items = [
            {"order_id": "o1", "freight_value": "10.5"},
            {"order_id": "o2", "freight_value": "20.5"},
        ]

        summary = bench.summarize_olist_rows(orders, items)

        self.assertEqual(summary["order_count"], 2)
        self.assertEqual(summary["late_delivery_rate"], 0.5)
        self.assertEqual(summary["freight_value"]["mean"], 15.5)

    def test_nyc_tlc_dispatch_proxy_serves_when_idle_vehicle_available(self) -> None:
        trips = [
            {"pickup_datetime": "2026-01-01 08:00:00", "pickup_zone": "1", "dropoff_zone": "2"},
            {"pickup_datetime": "2026-01-01 08:01:00", "pickup_zone": "1", "dropoff_zone": "3"},
            {"pickup_datetime": "2026-01-01 09:00:00", "pickup_zone": "2", "dropoff_zone": "1"},
        ]

        report = bench.simulate_zone_dispatch_proxy(trips, vehicles_per_zone=1, service_minutes=20)

        self.assertEqual(report["trip_count"], 3)
        self.assertGreater(report["served_demand_proxy_rate"], 0.0)
        self.assertLess(report["served_demand_proxy_rate"], 1.0)

    def test_svrpbench_summary_detects_stochastic_fields(self) -> None:
        records = [
            {"instance_id": "i1", "customers": [1, 2], "accident_prob": 0.1, "time_windows": [[0, 1]]},
            {"instance_id": "i2", "num_customers": 3, "delay": [0.2, 0.3]},
        ]

        summary = bench.summarize_svrpbench_records(records)

        self.assertEqual(summary["instance_count"], 2)
        self.assertEqual(summary["customer_count"]["max"], 3.0)
        self.assertIn("accident_prob", summary["stochastic_fields_present"])
        self.assertIn("time_windows", summary["time_window_fields_present"])

    def test_svrpbench_summary_accepts_array_valued_locations(self) -> None:
        records = [{"instance_id": "i1", "locations": np.array([[0.0, 0.0], [1.0, 1.0]])}]

        summary = bench.summarize_svrpbench_records(records)

        self.assertEqual(summary["instance_count"], 1)
        self.assertEqual(summary["customer_count"]["max"], 2.0)


if __name__ == "__main__":
    unittest.main()
