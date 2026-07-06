from __future__ import annotations

import unittest

from scripts.ortools_route_benchmark_suite import (
    parse_cvrplib_text,
    parse_solomon_vrptw_text,
    solve_cvrp_instance,
    solve_vrptw_instance,
)


SOLOMON_TINY = """C101

VEHICLE
NUMBER     CAPACITY
2          100

CUSTOMER
CUST NO.  XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME
0         0       0       0      0          1000     0
1         1       0       10     0          1000     0
2         2       0       10     0          1000     0
"""


CVRPLIB_TINY = """NAME : A-n3-k1
COMMENT : tiny
TYPE : CVRP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 0 0
2 1 0
3 2 0
DEMAND_SECTION
1 0
2 10
3 10
DEPOT_SECTION
1
-1
EOF
"""


class ORToolsRouteBenchmarkSuiteTests(unittest.TestCase):
    def test_parse_solomon_vrptw_text(self) -> None:
        instance = parse_solomon_vrptw_text(
            SOLOMON_TINY,
            name="C101",
            benchmark_family="solomon",
            source_url="fixture",
        )

        self.assertEqual(instance.name, "C101")
        self.assertEqual(instance.vehicle_count, 2)
        self.assertEqual(instance.capacity, 100)
        self.assertEqual(len(instance.nodes), 3)
        self.assertEqual(instance.nodes[0].ready_time, 0)

    def test_parse_cvrplib_text(self) -> None:
        instance = parse_cvrplib_text(CVRPLIB_TINY, source_url="fixture")

        self.assertEqual(instance.name, "A-n3-k1")
        self.assertEqual(instance.vehicle_count, 1)
        self.assertEqual(instance.capacity, 100)
        self.assertEqual(len(instance.nodes), 3)
        self.assertEqual(instance.nodes[2].demand, 10)

    def test_solve_tiny_vrptw_instance(self) -> None:
        instance = parse_solomon_vrptw_text(
            SOLOMON_TINY,
            name="C101",
            benchmark_family="solomon",
            source_url="fixture",
        )

        result = solve_vrptw_instance(instance, time_limit_seconds=1)

        self.assertEqual(result["solver_status"], "ROUTING_SUCCESS")
        self.assertGreater(result["objective_value"], 0)
        self.assertEqual(result["customer_count"], 2)

    def test_solve_tiny_cvrp_instance(self) -> None:
        instance = parse_cvrplib_text(CVRPLIB_TINY, source_url="fixture")

        result = solve_cvrp_instance(instance, time_limit_seconds=1)

        self.assertEqual(result["solver_status"], "ROUTING_SUCCESS")
        self.assertGreater(result["objective_value"], 0)
        self.assertEqual(result["customer_count"], 2)


if __name__ == "__main__":
    unittest.main()
