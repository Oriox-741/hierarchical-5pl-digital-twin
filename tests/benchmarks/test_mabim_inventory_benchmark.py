import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.mabim_inventory_benchmark import project_inspired_reorder, ss_policy, write_outputs, zero_reorder


class FakeMabimEnv:
    warehouse_count = 1

    def __init__(self, in_stock, in_transit, demand_mean):
        self._in_stock = np.array([in_stock], dtype=float)
        self._in_transit = np.array([in_transit], dtype=float)
        self._demand_mean = np.array([demand_mean], dtype=float)

    def get_sku_list(self):
        return list(range(self._demand_mean.shape[1]))

    def get_in_stock(self):
        return self._in_stock

    def get_in_transit(self):
        return self._in_transit

    def get_demand_mean(self):
        return self._demand_mean


class MabimInventoryBenchmarkTests(unittest.TestCase):
    def test_zero_reorder_matches_env_shape(self):
        env = FakeMabimEnv([1, 2], [0, 0], [5, 5])

        action = zero_reorder(env)

        self.assertEqual(action.shape, (1, 2))
        self.assertTrue(np.all(action == 0.0))

    def test_ss_policy_orders_only_below_reorder_point(self):
        env = FakeMabimEnv([1.0, 8.0], [0.0, 0.0], [2.0, 2.0])

        action = ss_policy(env, reorder_point=1.0, order_up_to=2.5)

        self.assertGreater(action[0, 0], 0.0)
        self.assertEqual(action[0, 1], 0.0)

    def test_project_inspired_reorder_is_finite_and_non_negative(self):
        env = FakeMabimEnv([0.5, 20.0], [0.0, 0.0], [1.0, 5.0])

        action = project_inspired_reorder(env)

        self.assertEqual(action.shape, (1, 2))
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue((action >= 0.0).all())

    def test_write_outputs_refuses_overwrite(self):
        report = {
            "decision": "MABIM_INVENTORY_BENCHMARK_READY",
            "policy_summaries": [{"policy_name": "p", "steps": 1}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_outputs(output_dir, report)

            with self.assertRaises(FileExistsError):
                write_outputs(output_dir, report)


if __name__ == "__main__":
    unittest.main()
