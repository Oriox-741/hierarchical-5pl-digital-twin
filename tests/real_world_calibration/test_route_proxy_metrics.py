from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.real_world_calibration.route_proxy_metrics import (
    bootstrap_mean_interval,
    greedy_nearest_neighbor_sequence,
    route_concentration,
    route_efficiency_ratio,
    route_sequence_from_rank_map,
    sequence_travel_time,
    summarize_amazon_route_root,
    tight_time_window_stress,
    travel_time_asymmetry,
)


class RouteProxyMetricsTests(unittest.TestCase):
    def test_sequence_travel_time_sums_directed_edges_in_rank_order(self) -> None:
        sequence = ["station", "a", "b", "station"]
        travel_times = {
            "station": {"a": 10.0, "b": 50.0},
            "a": {"b": 20.0, "station": 40.0},
            "b": {"station": 30.0, "a": 25.0},
        }

        self.assertEqual(sequence_travel_time(sequence, travel_times), 60.0)

    def test_route_sequence_from_rank_map_orders_lowest_rank_first(self) -> None:
        actual = {"stop_b": 2, "station": 0, "stop_a": 1}

        self.assertEqual(route_sequence_from_rank_map(actual), ["station", "stop_a", "stop_b"])

    def test_travel_time_asymmetry_reports_normalized_directed_gap(self) -> None:
        travel_times = {
            "a": {"a": 0.0, "b": 100.0, "c": 90.0},
            "b": {"a": 140.0, "b": 0.0, "c": 60.0},
            "c": {"a": 90.0, "b": 60.0, "c": 0.0},
        }

        metric = travel_time_asymmetry(travel_times)

        self.assertAlmostEqual(metric.mean_pair_asymmetry, 40.0 / 120.0 / 3.0)
        self.assertAlmostEqual(metric.max_pair_asymmetry, 40.0 / 120.0)
        self.assertEqual(metric.compared_pairs, 3)

    def test_route_concentration_returns_top_share_and_label(self) -> None:
        concentration = route_concentration({"shortest": 4, "low_congestion": 6, "high_resilience": 0})

        self.assertEqual(concentration.top_label, "low_congestion")
        self.assertEqual(concentration.top_count, 6)
        self.assertAlmostEqual(concentration.top_share, 0.6)
        self.assertEqual(concentration.total, 10)

    def test_greedy_nearest_neighbor_sequence_prefers_lowest_directed_travel_time(self) -> None:
        travel_times = {
            "station": {"station": 0.0, "a": 5.0, "b": 1.0, "c": 9.0},
            "a": {"station": 5.0, "a": 0.0, "b": 2.0, "c": 1.0},
            "b": {"station": 1.0, "a": 2.0, "b": 0.0, "c": 8.0},
            "c": {"station": 9.0, "a": 1.0, "b": 8.0, "c": 0.0},
        }

        sequence = greedy_nearest_neighbor_sequence(travel_times, start_stop_id="station")

        self.assertEqual(sequence, ["station", "b", "a", "c"])

    def test_route_efficiency_ratio_compares_actual_to_reference_travel_time(self) -> None:
        travel_times = {
            "station": {"station": 0.0, "a": 10.0, "b": 1.0},
            "a": {"station": 10.0, "a": 0.0, "b": 1.0},
            "b": {"station": 1.0, "a": 1.0, "b": 0.0},
        }

        ratio = route_efficiency_ratio(
            actual_sequence=["station", "a", "b"],
            reference_sequence=["station", "b", "a"],
            travel_times=travel_times,
        )

        self.assertAlmostEqual(ratio, 11.0 / 2.0)

    def test_tight_time_window_stress_counts_package_windows_below_threshold(self) -> None:
        package_data = {
            "route_1": {
                "stop_a": {
                    "pkg_1": {
                        "time_window": {
                            "start_time_utc": "2018-07-23 11:00:00",
                            "end_time_utc": "2018-07-23 12:00:00",
                        }
                    }
                },
                "stop_b": {
                    "pkg_2": {
                        "time_window": {
                            "start_time_utc": "2018-07-23 14:00:00",
                            "end_time_utc": "2018-07-23 21:00:00",
                        }
                    },
                    "pkg_3": {"time_window": {"start_time_utc": "NaN", "end_time_utc": "NaN"}},
                },
            }
        }

        stress = tight_time_window_stress(package_data["route_1"], tight_window_seconds=7200)

        self.assertEqual(stress.package_count, 3)
        self.assertEqual(stress.windowed_package_count, 2)
        self.assertEqual(stress.tight_package_count, 1)
        self.assertAlmostEqual(stress.tight_window_share, 0.5)
        self.assertEqual(stress.min_window_seconds, 3600.0)

    def test_bootstrap_mean_interval_is_deterministic_and_bounded(self) -> None:
        interval = bootstrap_mean_interval([1.0, 2.0, 3.0, 4.0], iterations=200, seed=7)

        self.assertLessEqual(interval.lower, interval.mean)
        self.assertGreaterEqual(interval.upper, interval.mean)
        self.assertAlmostEqual(interval.mean, 2.5)
        self.assertEqual(interval.sample_size, 4)

    def test_amazon_summary_includes_invalid_sequence_scores_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            route_id = "RouteID_sample"
            (root / "new_route_data.json").write_text(
                json.dumps(
                    {
                        route_id: {
                            "station_code": "DXX1",
                            "stops": {
                                "station": {"type": "Station"},
                                "a": {"type": "Dropoff"},
                                "b": {"type": "Dropoff"},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "new_actual_sequences.json").write_text(
                json.dumps({route_id: {"actual": {"station": 0, "a": 1, "b": 2}}}),
                encoding="utf-8",
            )
            (root / "new_travel_times.json").write_text(
                json.dumps(
                    {
                        route_id: {
                            "station": {"station": 0.0, "a": 2.0, "b": 5.0},
                            "a": {"station": 2.0, "a": 0.0, "b": 3.0},
                            "b": {"station": 5.0, "a": 3.0, "b": 0.0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "new_package_data.json").write_text(
                json.dumps({route_id: {"a": {"pkg": {"time_window": {}}}}}),
                encoding="utf-8",
            )
            (root / "new_invalid_sequence_scores.json").write_text(
                json.dumps({route_id: 0.42}),
                encoding="utf-8",
            )

            summary = summarize_amazon_route_root(root, route_limit=1)

            self.assertIn("invalid_sequence_scores", summary["present_files"])
            self.assertEqual(
                summary["present_files"]["invalid_sequence_scores"],
                str(root / "new_invalid_sequence_scores.json"),
            )
            self.assertEqual(summary["route_summaries"][0]["invalid_sequence_score"], 0.42)


if __name__ == "__main__":
    unittest.main()
