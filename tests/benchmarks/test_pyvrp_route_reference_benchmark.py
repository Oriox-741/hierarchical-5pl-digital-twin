from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import pyvrp_route_reference_benchmark as bench


class PyVRPRouteReferenceBenchmarkTests(unittest.TestCase):
    def test_select_instances_prefers_bounded_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cvrp = root / "cvrplib"
            cvrp.mkdir()
            for name in ("A-n32-k5.vrp", "A-n33-k5.vrp", "A-n34-k5.vrp"):
                (cvrp / name).write_text("NAME : unit\n", encoding="utf-8")

            selected = bench.select_instances(root, max_cvrp=2, max_vrptw=0)

            self.assertEqual(2, len(selected))
            self.assertTrue(all(item.family == "cvrplib_cvrp" for item in selected))

    def test_result_row_classifies_success_and_timeout(self) -> None:
        row = bench.build_result_row(
            instance=bench.RouteInstance(path=Path("A.vrp"), family="cvrplib_cvrp"),
            objective=123.0,
            runtime_seconds=0.5,
            feasible=True,
            routes_used=3,
            error=None,
        )

        self.assertEqual(row["status"], "success")
        self.assertEqual(row["objective"], 123.0)

        blocked = bench.build_result_row(
            instance=bench.RouteInstance(path=Path("B.vrp"), family="cvrplib_cvrp"),
            objective=None,
            runtime_seconds=0.1,
            feasible=False,
            routes_used=None,
            error="boom",
        )
        self.assertEqual(blocked["status"], "blocked")

    def test_write_outputs_refuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = {"decision": "PYVRP_ROUTE_REFERENCE_READY", "results": []}
            bench.write_outputs(output_dir, report)
            with self.assertRaises(FileExistsError):
                bench.write_outputs(output_dir, report)

            self.assertEqual(
                "PYVRP_ROUTE_REFERENCE_READY",
                json.loads((output_dir / "pyvrp_benchmark_report.json").read_text(encoding="utf-8"))["decision"],
            )

    def test_parse_solomon_txt_extracts_vehicle_and_customer_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c101.txt"
            path.write_text(
                "\n".join(
                    [
                        "C101",
                        "",
                        "VEHICLE",
                        "NUMBER     CAPACITY",
                        "  25         200",
                        "",
                        "CUSTOMER",
                        "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME",
                        "    0      40         50          0          0       1236          0",
                        "    1      45         68         10        912        967         90",
                    ]
                ),
                encoding="utf-8",
            )

            parsed = bench.parse_solomon_txt(path)

            self.assertEqual(parsed.name, "C101")
            self.assertEqual(parsed.vehicle_count, 25)
            self.assertEqual(parsed.capacity, 200)
            self.assertEqual(parsed.customers[0].node_id, 0)
            self.assertEqual(parsed.customers[1].demand, 10)
            self.assertEqual(parsed.customers[1].ready_time, 912)

    def test_build_solomon_model_accepts_parsed_instance(self) -> None:
        parsed = bench.parse_solomon_txt(
            Path("data/public/route_benchmarks_20260613/pyvrp_extracted_vrptw/In/c101.txt")
        )

        model = bench.build_solomon_model(parsed)
        data = model.data()

        self.assertEqual(data.num_clients, len(parsed.customers) - 1)
        self.assertEqual(data.num_vehicle_types, 1)


if __name__ == "__main__":
    unittest.main()
