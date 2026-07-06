"""Small OR-Tools VRPTW smoke benchmark.

This is a classical-optimizer sanity check, not a production replacement. It
solves one tiny deterministic vehicle-routing-with-time-windows instance and
writes a JSON report to a fresh caller-provided path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ortools
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROTECTED_OUTPUT_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("models", "registry"),
    ("models", "production"),
    ("models", "baselines"),
    ("models", "checkpoints"),
    ("models", "eval"),
    ("db",),
)


def build_smoke_vrptw_data() -> dict[str, Any]:
    return {
        "time_matrix": [
            [0, 4, 8, 8, 7, 3],
            [4, 0, 5, 4, 8, 6],
            [8, 5, 0, 3, 6, 7],
            [8, 4, 3, 0, 5, 6],
            [7, 8, 6, 5, 0, 4],
            [3, 6, 7, 6, 4, 0],
        ],
        "time_windows": [
            (0, 100),
            (0, 40),
            (0, 40),
            (0, 40),
            (0, 40),
            (0, 40),
        ],
        "num_vehicles": 1,
        "depot": 0,
    }


def nearest_neighbor_baseline(data: Mapping[str, Any]) -> dict[str, Any]:
    matrix = data["time_matrix"]
    depot = int(data["depot"])
    unvisited = set(range(len(matrix)))
    unvisited.remove(depot)
    route = [depot]
    current = depot
    travel_time = 0
    while unvisited:
        next_node = min(unvisited, key=lambda node: (int(matrix[current][node]), node))
        travel_time += int(matrix[current][next_node])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    travel_time += int(matrix[current][depot])
    route.append(depot)
    return {"route": route, "travel_time": travel_time}


def solve_vrptw_smoke(data: Mapping[str, Any], *, time_limit_seconds: int = 1) -> dict[str, Any]:
    matrix = data["time_matrix"]
    manager = pywrapcp.RoutingIndexManager(len(matrix), int(data["num_vehicles"]), int(data["depot"]))
    routing = pywrapcp.RoutingModel(manager)

    def transit_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    routing.AddDimension(
        transit_callback_index,
        30,
        120,
        False,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    for location_index, time_window in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(location_index)
        time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
    for vehicle_id in range(int(data["num_vehicles"])):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)
        depot_window = data["time_windows"][int(data["depot"])]
        time_dimension.CumulVar(start_index).SetRange(int(depot_window[0]), int(depot_window[1]))
        time_dimension.CumulVar(end_index).SetRange(int(depot_window[0]), 120)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = max(1, int(time_limit_seconds))

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return {"status": "INFEASIBLE"}

    routes: list[dict[str, Any]] = []
    served_nodes: list[int] = []
    for vehicle_id in range(int(data["num_vehicles"])):
        index = routing.Start(vehicle_id)
        route_nodes: list[int] = []
        arrival_times: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            if node != int(data["depot"]):
                served_nodes.append(node)
            arrival_times.append(int(solution.Value(time_dimension.CumulVar(index))))
            index = solution.Value(routing.NextVar(index))
        route_nodes.append(manager.IndexToNode(index))
        arrival_times.append(int(solution.Value(time_dimension.CumulVar(index))))
        routes.append({"vehicle_id": vehicle_id, "route": route_nodes, "arrival_times": arrival_times})

    return {
        "status": "FEASIBLE",
        "objective": int(solution.ObjectiveValue()),
        "routes": routes,
        "served_nodes": sorted(served_nodes),
    }


def build_ortools_smoke_report(*, time_limit_seconds: int = 1) -> dict[str, Any]:
    data = build_smoke_vrptw_data()
    baseline = nearest_neighbor_baseline(data)
    solution = solve_vrptw_smoke(data, time_limit_seconds=time_limit_seconds)
    decision = (
        "ORTOOLS_VRPTW_SMOKE_BENCHMARK_READY"
        if solution.get("status") == "FEASIBLE"
        else "ORTOOLS_VRPTW_SMOKE_BENCHMARK_BLOCKED"
    )
    return {
        "decision": decision,
        "benchmark_metadata": {
            "benchmark": "ortools_vrptw_smoke",
            "ortools_version": getattr(ortools, "__version__", "unknown"),
            "time_limit_seconds": int(time_limit_seconds),
            "node_count": len(data["time_matrix"]),
            "vehicle_count": int(data["num_vehicles"]),
        },
        "nearest_neighbor_baseline": baseline,
        "ortools_solution": solution,
        "interpretation": {
            "scope": "tiny deterministic VRPTW smoke benchmark",
            "not_a_full_production_benchmark": True,
            "use": "confirms an exact/classical solver path is available for small route instances",
        },
    }


def write_ortools_smoke_report(output: Path, report: Mapping[str, Any]) -> None:
    ensure_fresh_output_file(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def ensure_fresh_output_file(output: Path) -> None:
    output = Path(output)
    _reject_protected_output_path(output)
    if output.exists():
        raise FileExistsError(f"output file already exists: {output}")


def _reject_protected_output_path(output: Path) -> None:
    candidate = Path(output).resolve(strict=False)
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        protected_root = REPO_ROOT.joinpath(*prefix).resolve(strict=False)
        if candidate == protected_root or protected_root in candidate.parents:
            raise ValueError(f"refusing to write benchmark output under protected path: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny OR-Tools VRPTW smoke benchmark.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-limit-seconds", type=int, default=1)
    args = parser.parse_args(argv)

    report = build_ortools_smoke_report(time_limit_seconds=args.time_limit_seconds)
    write_ortools_smoke_report(Path(args.output), report)
    print(report["decision"])
    print(json.dumps(report["ortools_solution"], sort_keys=True))
    return 0 if report["decision"] == "ORTOOLS_VRPTW_SMOKE_BENCHMARK_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
