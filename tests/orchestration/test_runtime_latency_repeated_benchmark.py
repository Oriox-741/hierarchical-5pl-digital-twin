from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.runtime_latency_repeated_benchmark import (
    aggregate_repeated_latency_reports,
    write_repeated_latency_report,
)


class RuntimeLatencyRepeatedBenchmarkTests(unittest.TestCase):
    def test_aggregate_repeated_latency_reports(self) -> None:
        reports = [
            _report(seed=42, mean=1.0, p95=2.0, p99=3.0, max_ms=4.0),
            _report(seed=43, mean=2.0, p95=3.0, p99=4.0, max_ms=5.0),
        ]

        aggregate = aggregate_repeated_latency_reports(reports)

        self.assertEqual(aggregate["decision"], "RUNTIME_LATENCY_REPEATED_BENCHMARK_READY")
        self.assertEqual(aggregate["run_count"], 2)
        self.assertAlmostEqual(aggregate["aggregate_latency_ms"]["mean_ms"]["mean"], 1.5)
        self.assertEqual(aggregate["aggregate_latency_ms"]["p99_ms"]["max"], 4.0)

    def test_writer_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            write_repeated_latency_report(output, {"decision": "ok"})
            with self.assertRaises(FileExistsError):
                write_repeated_latency_report(output, {"decision": "ok"})


def _report(*, seed: int, mean: float, p95: float, p99: float, max_ms: float) -> dict:
    return {
        "decision": "RUNTIME_LATENCY_BENCHMARK_READY",
        "runtime_metadata": {"seed": seed, "iterations": 1000},
        "load_latency_ms": 10.0,
        "latency_ms": {
            "count": 1000,
            "mean_ms": mean,
            "p50_ms": mean,
            "p95_ms": p95,
            "p99_ms": p99,
            "max_ms": max_ms,
            "min_ms": 0.1,
        },
        "prediction_validation": {"valid_predictions": 1000},
    }


if __name__ == "__main__":
    unittest.main()
