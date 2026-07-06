from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.full_completion_rule_based_benchmark import (
    BenchmarkChunk,
    aggregate_rule_chunks,
    build_chunk_plan,
    next_pending_chunk,
    write_chunk_artifacts,
)


class FullCompletionRuleBasedBenchmarkTests(unittest.TestCase):
    def test_direct_script_help_imports_repo_modules(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/full_completion_rule_based_benchmark.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resumable full rule-based", result.stdout)

    def test_next_pending_chunk_skips_completed_chunks_in_plan_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            plan = build_chunk_plan(["s2", "s1"], ["b2", "b1"])
            self.assertEqual(plan[0], BenchmarkChunk(baseline_name="b1", scenario_id="s1"))

            write_chunk_artifacts(
                output_dir=output_dir,
                chunk=plan[0],
                summary={"baseline_name": "b1", "scenario_id": "s1", "episodes": 1, "steps": 1},
                episode_rows=[{"baseline_name": "b1", "scenario_id": "s1", "episode_index": 0, "service_level": 1.0}],
            )

            self.assertEqual(next_pending_chunk(output_dir, plan), plan[1])

    def test_aggregate_refuses_incomplete_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            plan = build_chunk_plan(["s1", "s2"], ["b1"])
            write_chunk_artifacts(
                output_dir=output_dir,
                chunk=plan[0],
                summary={"baseline_name": "b1", "scenario_id": "s1", "episodes": 1, "steps": 1},
                episode_rows=[{"baseline_name": "b1", "scenario_id": "s1", "episode_index": 0, "service_level": 1.0}],
            )

            with self.assertRaises(RuntimeError):
                aggregate_rule_chunks(
                    output_dir=output_dir,
                    plan=plan,
                    episodes=1,
                    seed=42,
                    scenario_dir=Path("configs/eval_scenarios"),
                    hierarchical_summary_path=None,
                    hierarchical_episode_path=None,
                )

    def test_aggregate_writes_report_csv_jsonl_and_paired_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "rule"
            reference_dir = Path(tmp) / "reference"
            output_dir.mkdir()
            reference_dir.mkdir()
            plan = build_chunk_plan(["s1"], ["b1"])
            write_chunk_artifacts(
                output_dir=output_dir,
                chunk=plan[0],
                summary={
                    "baseline_name": "b1",
                    "scenario_id": "s1",
                    "episodes": 2,
                    "steps": 2,
                    "service_level": 0.8,
                    "mean_lateness": 0.1,
                    "dispatch_success_per_attempt": 0.9,
                    "hard_blockers": {},
                },
                episode_rows=[
                    {"baseline_name": "b1", "scenario_id": "s1", "episode_index": 0, "service_level": 0.7},
                    {"baseline_name": "b1", "scenario_id": "s1", "episode_index": 1, "service_level": 0.9},
                ],
            )
            (reference_dir / "scenario_summary.json").write_text(
                json.dumps({"scenarios": [{"scenario_id": "s1", "service_level": 0.75}]}),
                encoding="utf-8",
            )
            (reference_dir / "episode_metrics.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"scenario_id": "s1", "episode_index": 0, "service_level": 0.6}),
                        json.dumps({"scenario_id": "s1", "episode_index": 1, "service_level": 0.8}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = aggregate_rule_chunks(
                output_dir=output_dir,
                plan=plan,
                episodes=2,
                seed=42,
                scenario_dir=Path("configs/eval_scenarios"),
                hierarchical_summary_path=reference_dir / "scenario_summary.json",
                hierarchical_episode_path=reference_dir / "episode_metrics.jsonl",
            )

            self.assertEqual(report["decision"], "RULE_BASED_FULL_BENCHMARK_READY")
            self.assertEqual(len(report["comparison_to_hierarchical_v1"]), 1)
            self.assertAlmostEqual(
                report["comparison_to_hierarchical_v1"][0]["paired_service_delta_stats"]["mean"],
                0.1,
            )
            self.assertTrue((output_dir / "rule_based_full_report.json").exists())
            self.assertTrue((output_dir / "rule_based_full_summary.csv").exists())
            self.assertTrue((output_dir / "episode_metrics.jsonl").exists())
            self.assertTrue((output_dir / "rule_based_full_report.md").exists())


if __name__ == "__main__":
    unittest.main()
