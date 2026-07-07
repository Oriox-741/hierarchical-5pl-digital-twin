from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class AblationCliDryRunTests(unittest.TestCase):
    def test_ablation_matrix_dry_run_all_exits_zero_without_private_artifacts(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_ablation_matrix.py", "--dry-run", "--all"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("DRY RUN", result.stdout)
        self.assertIn("neutral_ppo_eval", result.stdout)
        self.assertNotIn("No result generated", result.stderr)


if __name__ == "__main__":
    unittest.main()
