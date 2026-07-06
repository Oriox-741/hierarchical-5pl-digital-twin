from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import company_data_replay_validator as replay


class CompanyDataReplayValidatorTests(unittest.TestCase):
    def test_valid_synthetic_replay_answers_action_watch_questions(self) -> None:
        report = replay.build_replay_validation_report(replay.built_in_replay_fixture("valid_minimal"))

        self.assertEqual(report["decision"]["classification"], "COMPANY_REPLAY_SCHEMA_READY")
        self.assertEqual(report["action_watch_summary"]["action_24_count"], 1)
        self.assertEqual(report["action_watch_summary"]["action_32_count"], 1)
        self.assertEqual(report["fleet_economics"]["secondary_cost_premium_mean"], 14.0)
        self.assertEqual(report["inventory_reorder"]["none_reorder_stockout_rate"], 0.0)

    def test_missing_secondary_cost_blocks_secondary_fleet_validation(self) -> None:
        report = replay.build_replay_validation_report(replay.built_in_replay_fixture("missing_secondary_cost"))

        self.assertEqual(report["decision"]["classification"], "COMPANY_REPLAY_SCHEMA_NEEDS_FIXES")
        self.assertIn("secondary_fleet_cost", report["missing_fields_by_table"]["costs"])

    def test_stockout_after_no_reorder_escalates_warning(self) -> None:
        report = replay.build_replay_validation_report(replay.built_in_replay_fixture("no_reorder_stockout"))

        self.assertEqual(report["decision"]["classification"], "COMPANY_REPLAY_SCHEMA_WARNINGS_ONLY")
        self.assertGreater(report["inventory_reorder"]["none_reorder_stockout_rate"], 0.0)

    def test_fixture_root_cli_is_not_reported_as_synthetic_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for table, rows in replay.built_in_replay_fixture("valid_minimal").items():
                with (root / f"{table}.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            output = root / "report.json"

            replay.main(["--fixture-root", str(root), "--output", str(output)])
            report = replay.json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(report["report_metadata"]["synthetic_only"])
        self.assertEqual(report["report_metadata"]["input_mode"], "caller_supplied_fixture_root")

    def test_write_report_rejects_protected_output_paths(self) -> None:
        protected_output = Path("models/registry/replay_report_test_should_not_write.json")

        with patch.object(Path, "write_text", side_effect=AssertionError("write_text should not be reached")):
            with self.assertRaises(ValueError):
                replay.write_report(protected_output, {"decision": {"classification": "SHOULD_NOT_WRITE"}})

        self.assertFalse(protected_output.exists())


if __name__ == "__main__":
    unittest.main()
