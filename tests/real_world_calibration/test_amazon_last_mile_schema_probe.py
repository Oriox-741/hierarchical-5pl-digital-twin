from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.real_world_calibration.amazon_last_mile_schema_probe import (
    probe_amazon_last_mile_schema,
)


class AmazonLastMileSchemaProbeTests(unittest.TestCase):
    def test_probe_reports_new_invalid_sequence_scores_file_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "new_invalid_sequence_scores.json").write_text(
                json.dumps({"RouteID_sample": 0.42}),
                encoding="utf-8",
            )

            summary = probe_amazon_last_mile_schema(root, max_routes=1)

            self.assertIn("new_invalid_sequence_scores.json", summary["files"])
            file_summary = summary["files"]["new_invalid_sequence_scores.json"]
            self.assertTrue(file_summary["present"])
            self.assertEqual(file_summary["shape"]["sample_top_level_keys"], ["RouteID_sample"])
            self.assertEqual(file_summary["shape"]["first_value_type"], "float")


if __name__ == "__main__":
    unittest.main()
