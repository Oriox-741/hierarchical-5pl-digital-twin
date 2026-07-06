from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import ortools_vrptw_smoke_benchmark as smoke


class ORToolsVRPTWSmokeBenchmarkTests(unittest.TestCase):
    def test_smoke_data_has_expected_shape(self) -> None:
        data = smoke.build_smoke_vrptw_data()

        self.assertEqual(data["depot"], 0)
        self.assertEqual(data["num_vehicles"], 1)
        self.assertEqual(len(data["time_matrix"]), len(data["time_windows"]))
        self.assertGreater(len(data["time_matrix"]), 3)

    def test_nearest_neighbor_baseline_visits_every_customer_once(self) -> None:
        data = smoke.build_smoke_vrptw_data()

        baseline = smoke.nearest_neighbor_baseline(data)

        self.assertEqual(baseline["route"][0], data["depot"])
        self.assertEqual(baseline["route"][-1], data["depot"])
        self.assertEqual(sorted(baseline["route"][1:-1]), [1, 2, 3, 4, 5])
        self.assertGreater(baseline["travel_time"], 0)

    def test_ortools_solves_smoke_instance(self) -> None:
        data = smoke.build_smoke_vrptw_data()

        solution = smoke.solve_vrptw_smoke(data, time_limit_seconds=1)

        self.assertEqual(solution["status"], "FEASIBLE")
        self.assertEqual(sorted(solution["served_nodes"]), [1, 2, 3, 4, 5])
        self.assertGreater(solution["objective"], 0)

    def test_report_builder_compares_solver_to_rule_baseline(self) -> None:
        report = smoke.build_ortools_smoke_report(time_limit_seconds=1)

        self.assertEqual(report["decision"], "ORTOOLS_VRPTW_SMOKE_BENCHMARK_READY")
        self.assertEqual(report["ortools_solution"]["status"], "FEASIBLE")
        self.assertLessEqual(
            report["ortools_solution"]["objective"],
            report["nearest_neighbor_baseline"]["travel_time"],
        )

    def test_output_guard_and_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ortools.json"
            report = {"decision": "ORTOOLS_VRPTW_SMOKE_BENCHMARK_READY"}

            smoke.write_ortools_smoke_report(output, report)

            self.assertEqual(
                "ORTOOLS_VRPTW_SMOKE_BENCHMARK_READY",
                json.loads(output.read_text(encoding="utf-8"))["decision"],
            )
            with self.assertRaises(FileExistsError):
                smoke.ensure_fresh_output_file(output)

        with self.assertRaises(ValueError):
            smoke.ensure_fresh_output_file(Path("models/checkpoints/bad.json"))
        with self.assertRaises(ValueError):
            smoke.ensure_fresh_output_file(Path.cwd() / "models" / "checkpoints" / "bad.json")


if __name__ == "__main__":
    unittest.main()
