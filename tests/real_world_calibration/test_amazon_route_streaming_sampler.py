from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.real_world_calibration.amazon_route_streaming_sampler import (
    analyze_amazon_full_route_proxy,
    stream_top_level_json_items,
)


class AmazonRouteStreamingSamplerTests(unittest.TestCase):
    def test_stream_top_level_json_items_yields_route_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.json"
            path.write_text(json.dumps({"r2": {"x": 2}, "r1": {"x": 1}}), encoding="utf-8")

            items = list(stream_top_level_json_items(path))

            self.assertEqual(items, [("r2", {"x": 2}), ("r1", {"x": 1})])

    def test_stream_top_level_json_items_tolerates_amazon_nan_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.json"
            path.write_text('{"r1": {"zone_id": NaN}, "r2": {"zone_id": "ok"}}', encoding="utf-8")

            items = list(stream_top_level_json_items(path))

            self.assertEqual(items[0], ("r1", {"zone_id": None}))
            self.assertEqual(items[1], ("r2", {"zone_id": "ok"}))

    def test_analyze_amazon_full_route_proxy_writes_summary_for_tiny_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "out"
            build = root / "almrrc2021-data-training" / "model_build_inputs"
            build.mkdir(parents=True)
            _write_fixture(build)

            report = analyze_amazon_full_route_proxy(data_root=root, output_dir=output)

            self.assertEqual(report["decision"], "AMAZON_FULL_ROUTE_PROXY_READY")
            self.assertEqual(report["coverage"]["routes_processed"], 2)
            self.assertEqual(report["coverage"]["package_count_processed"], 3)
            self.assertEqual(report["actual_to_greedy_travel_time_ratio"]["count"], 2)
            self.assertEqual(report["route_score_distribution"], {"High": 1, "Medium": 1})
            self.assertGreater(report["travel_time_asymmetry_mean_pair"]["mean"], 0.0)
            self.assertTrue((output / "amazon_full_route_proxy_report.json").exists())
            self.assertTrue((output / "amazon_full_route_proxy_summary.csv").exists())
            self.assertTrue((output / "per_route_metrics.jsonl").exists())


def _write_fixture(build: Path) -> None:
    (build / "actual_sequences.json").write_text(
        json.dumps(
            {
                "r1": {"actual": {"A": 0, "B": 1, "C": 2}},
                "r2": {"actual": {"S": 0, "Y": 1, "X": 2}},
            }
        ),
        encoding="utf-8",
    )
    (build / "invalid_sequence_scores.json").write_text(
        json.dumps({"r1": 0.2, "r2": 0.4}),
        encoding="utf-8",
    )
    (build / "route_data.json").write_text(
        json.dumps(
            {
                "r1": {
                    "station_code": "D1",
                    "route_score": "High",
                    "stops": {
                        "A": {"type": "Station"},
                        "B": {"type": "Dropoff"},
                        "C": {"type": "Dropoff"},
                    },
                },
                "r2": {
                    "station_code": "D2",
                    "route_score": "Medium",
                    "stops": {
                        "S": {"type": "Station"},
                        "X": {"type": "Dropoff"},
                        "Y": {"type": "Dropoff"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (build / "package_data.json").write_text(
        json.dumps(
            {
                "r1": {
                    "B": {
                        "p1": {
                            "time_window": {
                                "start_time_utc": "2021-01-01T08:00:00Z",
                                "end_time_utc": "2021-01-01T09:30:00Z",
                            }
                        }
                    },
                    "C": {
                        "p2": {
                            "time_window": {
                                "start_time_utc": "2021-01-01T08:00:00Z",
                                "end_time_utc": "2021-01-01T14:00:00Z",
                            }
                        }
                    },
                },
                "r2": {"X": {"p3": {}}},
            }
        ),
        encoding="utf-8",
    )
    (build / "travel_times.json").write_text(
        json.dumps(
            {
                "r1": {
                    "A": {"A": 0, "B": 1, "C": 5},
                    "B": {"A": 2, "B": 0, "C": 1},
                    "C": {"A": 4, "B": 3, "C": 0},
                },
                "r2": {
                    "S": {"S": 0, "X": 5, "Y": 1},
                    "X": {"S": 5, "X": 0, "Y": 1},
                    "Y": {"S": 2, "X": 1, "Y": 0},
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
