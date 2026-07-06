from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import inventory_reorder_benchmark as bench


class InventoryReorderBenchmarkTests(unittest.TestCase):
    def test_policy_actions_are_bounded_to_action_space(self) -> None:
        obs = np.array([0.0, 10.0, 100.0, 0.0, 5.0], dtype=np.float32)
        low = np.zeros(3, dtype=np.float32)
        high = np.array([100.0, 200.0, 300.0], dtype=np.float32)

        for policy_name in bench.POLICY_NAMES:
            with self.subTest(policy_name=policy_name):
                action = bench.policy_action(policy_name, obs, low, high)

                self.assertEqual(action.shape, (3,))
                self.assertTrue(np.all(action >= low))
                self.assertTrue(np.all(action <= high))

    def test_project_inspired_policy_increases_order_when_inventory_is_low(self) -> None:
        low = np.zeros(3, dtype=np.float32)
        high = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        scarce_obs = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        stocked_obs = np.array([80.0, 80.0, 80.0], dtype=np.float32)

        scarce = bench.policy_action("project_heuristic_continuous_reorder", scarce_obs, low, high)
        stocked = bench.policy_action("project_heuristic_continuous_reorder", stocked_obs, low, high)

        self.assertGreater(float(scarce.mean()), float(stocked.mean()))

    def test_summarize_policy_run_reports_cost_and_service_proxies(self) -> None:
        summary = bench.summarize_policy_run(
            policy_name="base_stock_order_up_to",
            rewards=[-10.0, -5.0],
            inventories=[100.0, 80.0],
            backlogs=[0.0, 2.0],
            order_quantities=[10.0, 20.0],
        )

        self.assertEqual(summary["policy_name"], "base_stock_order_up_to")
        self.assertEqual(summary["total_cost"], 15.0)
        self.assertEqual(summary["no_backlog_step_rate"], 0.5)
        self.assertEqual(summary["mean_order_quantity"], 15.0)

    def test_write_outputs_refuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = {"decision": "INVENTORY_REORDER_BENCHMARK_READY", "policy_summaries": []}
            bench.write_outputs(output_dir, report)
            with self.assertRaises(FileExistsError):
                bench.write_outputs(output_dir, report)

            self.assertEqual(
                "INVENTORY_REORDER_BENCHMARK_READY",
                json.loads((output_dir / "inventory_benchmark_report.json").read_text(encoding="utf-8"))["decision"],
            )


if __name__ == "__main__":
    unittest.main()
