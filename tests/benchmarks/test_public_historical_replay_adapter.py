from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import public_historical_replay_adapter as replay


class FakeJointPolicy:
    def __init__(self, discrete: int = 24) -> None:
        self.discrete = discrete
        self.calls: list[np.ndarray] = []

    def predict_joint(self, observation: np.ndarray, *, deterministic: bool = True) -> SimpleNamespace:
        self.calls.append(np.asarray(observation, dtype=np.float32).reshape(-1))
        return SimpleNamespace(
            continuous=np.array([0.1, -0.2, 0.0, 0.75, -0.75], dtype=np.float32),
            discrete=self.discrete,
        )


def _rich_row() -> replay.ReplayInputRow:
    return replay.ReplayInputRow(
        source_dataset="synthetic",
        source_row_id="row-1",
        event_time="2026-01-01T08:15:00",
        service_time_seconds=3600.0,
        wait_time_seconds=240.0,
        route_distance_km=12.5,
        pickup_location_id="zone-a",
        dropoff_location_id="zone-b",
        demand_pressure=0.72,
        dispatch_pressure=0.61,
        route_congestion_proxy=0.44,
        route_disruption_proxy=0.31,
        fleet_pressure=0.68,
        lateness_risk=0.22,
        cost_proxy=35.0,
    )


class PublicHistoricalReplayAdapterTests(unittest.TestCase):
    def test_build_public_observation_returns_73_dims_and_feature_status(self) -> None:
        observation = replay.build_public_observation(_rich_row())

        self.assertEqual(len(observation.vector), 73)
        self.assertEqual(len(observation.status_by_feature), 73)
        self.assertEqual(len(observation.missingness_mask), 73)
        self.assertEqual(observation.status_by_feature["feasible_dispatch_opportunity"], "observed")
        self.assertEqual(observation.status_by_feature["bias"], "neutral")
        self.assertIn(observation.confidence, {"high", "medium"})
        self.assertLess(observation.missingness_rate, 0.75)

    def test_sparse_public_row_is_low_confidence_and_mostly_unavailable(self) -> None:
        sparse = replay.ReplayInputRow(source_dataset="olist", source_row_id="order-1")

        observation = replay.build_public_observation(sparse)

        self.assertEqual(len(observation.vector), 73)
        self.assertEqual(observation.confidence, "low")
        self.assertGreater(observation.missingness_rate, 0.80)
        self.assertEqual(observation.status_by_feature["route_disruption_pressure"], "unavailable")

    def test_fake_policy_prediction_is_decoded_and_bounded(self) -> None:
        prediction = replay.predict_public_row(FakeJointPolicy(discrete=24), _rich_row())

        self.assertEqual(len(prediction["continuous"]), 5)
        self.assertEqual(prediction["discrete"], 24)
        self.assertEqual(prediction["decoded_action"]["dispatch"], "dispatch")
        self.assertEqual(prediction["decoded_action"]["route"], "shortest")
        self.assertEqual(prediction["decoded_action"]["mode"], "secondary_fleet")
        self.assertEqual(prediction["decoded_action"]["reorder"], "none")

    def test_no_policy_dry_run_keeps_predictions_empty(self) -> None:
        report = replay.build_replay_report([_rich_row()], policy=None, source_name="synthetic")

        self.assertFalse(report["report_metadata"]["policy_loaded"])
        self.assertIsNone(report["rows"][0]["prediction"])
        self.assertEqual(report["summary"]["row_count"], 1)
        self.assertEqual(report["summary"]["prediction_count"], 0)

    def test_policy_report_summarizes_action_24_and_32_rates(self) -> None:
        report = replay.build_replay_report(
            [_rich_row(), _rich_row()],
            policy=FakeJointPolicy(discrete=32),
            source_name="synthetic",
        )

        self.assertTrue(report["report_metadata"]["policy_loaded"])
        self.assertEqual(report["summary"]["prediction_count"], 2)
        self.assertEqual(report["summary"]["action_distribution"]["32"], 2)
        self.assertEqual(report["summary"]["action_24_rate"], 0.0)
        self.assertEqual(report["summary"]["action_32_rate"], 1.0)
        self.assertEqual(report["summary"]["dispatch_rate"], 1.0)

    def test_write_report_rejects_protected_output_paths(self) -> None:
        protected_output = Path("models/registry/public_replay_should_not_write.json")

        with patch.object(Path, "write_text", side_effect=AssertionError("write_text should not be reached")):
            with self.assertRaises(ValueError):
                replay.write_report_fresh(protected_output, {"decision": {"classification": "SHOULD_NOT_WRITE"}})

        self.assertFalse(protected_output.exists())

    def test_cli_no_policy_jsonl_writes_fresh_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rows.jsonl"
            output_path = root / "report.json"
            input_path.write_text(json.dumps(_rich_row().as_dict()) + "\n", encoding="utf-8")

            exit_code = replay.main(["--input-jsonl", str(input_path), "--output", str(output_path)])

            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertFalse(report["report_metadata"]["policy_loaded"])
        self.assertEqual(report["summary"]["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
