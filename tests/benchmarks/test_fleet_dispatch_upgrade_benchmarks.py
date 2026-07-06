from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import fleet_dispatch_upgrade_benchmarks as bench


class FleetDispatchUpgradeBenchmarkTests(unittest.TestCase):
    def test_hvfhs_summary_computes_waits_bases_and_demand_pressure(self) -> None:
        rows = [
            {
                "request_datetime": "2026-01-01 08:00:00",
                "on_scene_datetime": "2026-01-01 08:05:00",
                "pickup_datetime": "2026-01-01 08:10:00",
                "dropoff_datetime": "2026-01-01 08:30:00",
                "dispatching_base_num": "B1",
                "PULocationID": "10",
                "DOLocationID": "20",
                "trip_miles": "4.5",
            },
            {
                "request_datetime": "2026-01-01 18:00:00",
                "on_scene_datetime": "2026-01-01 18:03:00",
                "pickup_datetime": "2026-01-01 18:12:00",
                "dropoff_datetime": "2026-01-01 18:32:00",
                "dispatching_base_num": "B2",
                "PULocationID": "10",
                "DOLocationID": "30",
                "trip_miles": "6.0",
            },
        ]

        report = bench.summarize_hvfhs_dispatch(rows, capacity_variants={"low": (1, 30.0)})

        self.assertEqual(report["trip_rows"], 2)
        self.assertEqual(report["base_count"], 2)
        self.assertEqual(report["request_to_pickup_wait_seconds"]["mean"], 660.0)
        self.assertEqual(report["request_to_on_scene_wait_seconds"]["mean"], 240.0)
        self.assertEqual(report["peak_offpeak_counts"]["peak"], 2)
        self.assertEqual(report["zone_hour_demand_pressure"]["max"], 1.0)
        self.assertIn("low", report["dispatch_capacity_proxy"])

    def test_city_trip_dispatch_summary_supports_chicago_tnp_fields(self) -> None:
        rows = [
            {
                "trip_start_timestamp": "2026-01-01T08:00:00",
                "trip_end_timestamp": "2026-01-01T08:20:00",
                "trip_seconds": "1200",
                "trip_miles": "5.0",
                "pickup_community_area": "1",
                "dropoff_community_area": "2",
                "trip_total": "22.50",
            },
            {
                "trip_start_timestamp": "2026-01-01T08:15:00",
                "trip_end_timestamp": "2026-01-01T08:45:00",
                "trip_seconds": "1800",
                "trip_miles": "7.0",
                "pickup_community_area": "1",
                "dropoff_community_area": "3",
                "trip_total": "30.00",
            },
        ]

        report = bench.summarize_city_trip_dispatch(rows, capacity_variants={"balanced": (1, 25.0)})

        self.assertEqual(report["trip_rows"], 2)
        self.assertEqual(report["trip_seconds"]["mean"], 1500.0)
        self.assertEqual(report["trip_miles"]["mean"], 6.0)
        self.assertEqual(report["fare_proxy"]["mean"], 26.25)
        self.assertEqual(report["zone_hour_demand_pressure"]["max"], 2.0)

    def test_vehicle_fleet_summary_computes_utilization_and_reposition_proxy(self) -> None:
        rows = [
            {
                "taxi_id": "taxi-a",
                "trip_start_timestamp": "2026-01-01 08:00:00",
                "trip_end_timestamp": "2026-01-01 08:10:00",
                "trip_seconds": "600",
                "trip_miles": "2.0",
                "pickup_community_area": "1",
                "dropoff_community_area": "2",
            },
            {
                "taxi_id": "taxi-a",
                "trip_start_timestamp": "2026-01-01 08:20:00",
                "trip_end_timestamp": "2026-01-01 08:30:00",
                "trip_seconds": "600",
                "trip_miles": "2.0",
                "pickup_community_area": "3",
                "dropoff_community_area": "4",
            },
            {
                "taxi_id": "taxi-b",
                "trip_start_timestamp": "2026-01-01 09:00:00",
                "trip_end_timestamp": "2026-01-01 09:15:00",
                "trip_seconds": "900",
                "trip_miles": "3.0",
                "pickup_community_area": "2",
                "dropoff_community_area": "2",
            },
        ]

        report = bench.summarize_vehicle_fleet_proxy(rows)

        self.assertEqual(report["vehicle_count"], 2)
        self.assertEqual(report["trips_per_vehicle"]["max"], 2.0)
        self.assertEqual(report["reposition_zone_change_proxy_count"], 1)
        self.assertEqual(report["trip_seconds"]["mean"], 700.0)

    def test_trajectory_summary_counts_taxis_and_distance_proxy(self) -> None:
        rows = [
            {"taxi_id": "1", "timestamp": "2026-01-01 08:00:00", "source_point": "POINT(-122.0 37.0)", "target_point": "POINT(-122.1 37.1)"},
            {"taxi_id": "1", "timestamp": "2026-01-01 09:00:00", "source_point": "POINT(-122.1 37.1)", "target_point": "POINT(-122.2 37.2)"},
            {"taxi_id": "2", "timestamp": "2026-01-01 09:00:00", "source_point": "POINT(-122.0 37.0)", "target_point": "POINT(-122.0 37.2)"},
        ]

        report = bench.summarize_taxi_trajectories(rows)

        self.assertEqual(report["trajectory_count"], 3)
        self.assertEqual(report["taxi_count"], 2)
        self.assertGreater(report["distance_km_proxy"]["mean"], 0.0)

    def test_trajectory_summary_extracts_figshare_taxi_id_prefix(self) -> None:
        rows = [
            {"trajectory": " aggjuo_55940", "timestamp": "2008-05-31 16:59:51", "start_point": "POINT(-122.42218 37.79769)", "end_point": "POINT(-122.42549 37.79722)"},
            {"trajectory": " aggjuo_55941", "timestamp": "2008-05-31 17:59:51", "start_point": "POINT(-122.42218 37.79769)", "end_point": "POINT(-122.42549 37.79722)"},
            {"trajectory": " agivle_58203", "timestamp": "2008-05-27 17:59:01", "start_point": "POINT(-122.41919 37.79635)", "end_point": "POINT(-122.40028 37.7943)"},
        ]

        report = bench.summarize_taxi_trajectories(rows)

        self.assertEqual(report["taxi_count"], 2)
        self.assertEqual(report["trajectory_count"], 3)

    def test_dynamic_routing_summary_detects_dynamic_fields(self) -> None:
        rows = [
            {"instance_id": "i1", "locations": [[0, 0], [1, 1]], "appear_times": [0, 10], "demands": [0, 1]},
            {"instance_id": "i2", "locations": [[0, 0], [2, 2], [3, 3]], "request_times": [0, 5, 15]},
        ]

        report = bench.summarize_dynamic_routing_records(rows)

        self.assertEqual(report["instance_count"], 2)
        self.assertEqual(report["customer_count"]["max"], 3.0)
        self.assertIn("appear_times", report["dynamic_fields_present"])
        self.assertIn("request_times", report["dynamic_fields_present"])

    def test_write_report_rejects_protected_paths_before_write(self) -> None:
        protected_output = Path("models/registry/fleet_dispatch_report_should_not_write.json")

        with patch.object(Path, "write_text", side_effect=AssertionError("write_text should not be reached")):
            with self.assertRaises(ValueError):
                bench.write_report_fresh(protected_output, {"decision": "SHOULD_NOT_WRITE"})

        self.assertFalse(protected_output.exists())

    def test_write_report_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            output.write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                bench.write_report_fresh(output, {"decision": "SHOULD_NOT_WRITE"})


if __name__ == "__main__":
    unittest.main()
