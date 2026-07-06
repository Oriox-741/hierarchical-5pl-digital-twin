from __future__ import annotations

import time
import unittest

from src.learn.pretrain_mdp_sanity_check import assert_all_checks_pass, run_all_checks


class PretrainMdpSanityCheckTest(unittest.TestCase):
    def test_pretrain_mdp_sanity_checks_pass_quickly(self) -> None:
        started = time.perf_counter()
        assert_all_checks_pass()
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 30.0)

    def test_each_sanity_check_reports_a_passing_result(self) -> None:
        results = run_all_checks()

        self.assertTrue(results)
        self.assertTrue(all(result.passed for result in results))


if __name__ == "__main__":
    unittest.main()
