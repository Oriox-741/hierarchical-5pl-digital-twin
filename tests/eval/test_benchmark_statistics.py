from __future__ import annotations

import unittest

from src.eval.benchmark_statistics import (
    bootstrap_mean_ci,
    exact_sign_test_two_sided,
    summary_stats,
)


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_summary_stats_reports_core_moments_and_quantiles(self) -> None:
        stats = summary_stats([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["sd"], 1.2909944487)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertAlmostEqual(stats["q25"], 1.75)
        self.assertAlmostEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["q75"], 3.25)

    def test_summary_stats_rejects_empty_or_non_finite_samples(self) -> None:
        with self.assertRaises(ValueError):
            summary_stats([])
        with self.assertRaises(ValueError):
            summary_stats([float("nan")])

    def test_bootstrap_mean_ci_is_deterministic_and_bounded(self) -> None:
        interval = bootstrap_mean_ci([1.0, 2.0, 3.0], iterations=100, seed=7)
        repeat = bootstrap_mean_ci([1.0, 2.0, 3.0], iterations=100, seed=7)

        self.assertEqual(interval, repeat)
        self.assertEqual(interval["sample_size"], 3)
        self.assertAlmostEqual(interval["mean"], 2.0)
        self.assertLessEqual(interval["lower"], interval["mean"])
        self.assertGreaterEqual(interval["upper"], interval["mean"])

    def test_exact_sign_test_ignores_zero_deltas(self) -> None:
        result = exact_sign_test_two_sided([1.0, 2.0, 0.0, -1.0, 3.0])

        self.assertEqual(result["positive"], 3)
        self.assertEqual(result["negative"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertAlmostEqual(result["p_value"], 0.625)


if __name__ == "__main__":
    unittest.main()
