"""Expanded OR-Tools route benchmark suite for public VRP instances."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


SOLOMON_ZIP_URL = "https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip"
HOMBERGER_200_ZIP_URL = (
    "https://www.sintef.no/globalassets/project/top/vrptw/homberger/200/"
    "homberger_200_customer_instances.zip"
)
CVRPLIB_INSTANCE_URLS = {
    "A-n32-k5": "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/4",
    "A-n33-k5": "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/5",
    "A-n33-k6": "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/7",
    "A-n34-k5": "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/8",
    "A-n36-k5": "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/9",
}
SOLOMON_SELECTION = ("C101", "C201", "R101", "R201", "RC101", "RC201")
HOMBERGER_SELECTION = ("c1_2_1", "c2_2_1", "r1_2_1")


@dataclass(frozen=True, slots=True)
class RouteNode:
    node_id: int
    x: float
    y: float
    demand: int
    ready_time: int = 0
    due_time: int = 1_000_000
    service_time: int = 0


@dataclass(frozen=True, slots=True)
class RouteInstance:
    name: str
    benchmark_family: str
    vehicle_count: int
    capacity: int
    nodes: tuple[RouteNode, ...]
    source_url: str


def parse_solomon_vrptw_text(
    text: str,
    *,
    name: str,
    benchmark_family: str,
    source_url: str,
) -> RouteInstance:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    vehicle_count: int | None = None
    capacity: int | None = None
    nodes: list[RouteNode] = []
    for index, line in enumerate(lines):
        if re.match(r"^NUMBER\s+CAPACITY", line, flags=re.IGNORECASE):
            if index + 1 >= len(lines):
                break
            values = _numbers(lines[index + 1])
            if len(values) >= 2:
                vehicle_count = int(values[0])
                capacity = int(values[1])
            continue
        values = _numbers(line)
        if len(values) >= 7 and values[0].is_integer():
            nodes.append(
                RouteNode(
                    node_id=int(values[0]),
                    x=float(values[1]),
                    y=float(values[2]),
                    demand=int(values[3]),
                    ready_time=int(values[4]),
                    due_time=int(values[5]),
                    service_time=int(values[6]),
                )
            )
    if vehicle_count is None or capacity is None:
        raise ValueError(f"could not parse vehicle header for {name}")
    if len(nodes) < 2:
        raise ValueError(f"expected at least depot plus one customer for {name}")
    return RouteInstance(
        name=name,
        benchmark_family=benchmark_family,
        vehicle_count=vehicle_count,
        capacity=capacity,
        nodes=tuple(nodes),
        source_url=source_url,
    )


def parse_cvrplib_text(text: str, *, source_url: str) -> RouteInstance:
    metadata: dict[str, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depot_id = 1
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper in {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}:
            section = upper
            continue
        if upper == "EOF":
            break
        if section is None and ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().upper()] = value.strip()
            continue
        if section == "NODE_COORD_SECTION":
            values = line.split()
            if len(values) >= 3:
                coords[int(values[0])] = (float(values[1]), float(values[2]))
        elif section == "DEMAND_SECTION":
            values = line.split()
            if len(values) >= 2:
                demands[int(values[0])] = int(float(values[1]))
        elif section == "DEPOT_SECTION":
            value = int(line.split()[0])
            if value != -1:
                depot_id = value

    name = metadata.get("NAME")
    capacity = int(float(metadata.get("CAPACITY", "0")))
    if not name or capacity <= 0 or not coords:
        raise ValueError("invalid CVRPLIB instance")
    vehicle_match = re.search(r"k(\d+)", name, flags=re.IGNORECASE)
    vehicle_count = int(vehicle_match.group(1)) if vehicle_match else max(1, math.ceil(sum(demands.values()) / capacity))

    ordered_ids = [depot_id] + sorted(node_id for node_id in coords if node_id != depot_id)
    nodes = []
    for new_index, original_id in enumerate(ordered_ids):
        x, y = coords[original_id]
        nodes.append(RouteNode(node_id=new_index, x=x, y=y, demand=int(demands.get(original_id, 0))))
    return RouteInstance(
        name=name,
        benchmark_family="cvrplib_cvrp",
        vehicle_count=vehicle_count,
        capacity=capacity,
        nodes=tuple(nodes),
        source_url=source_url,
    )


def solve_vrptw_instance(instance: RouteInstance, *, time_limit_seconds: int = 5) -> dict[str, Any]:
    manager = pywrapcp.RoutingIndexManager(len(instance.nodes), instance.vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)
    distance_matrix = _distance_matrix(instance.nodes)

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node] + int(instance.nodes[from_node].service_time)

    transit_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    _add_capacity_dimension(routing, manager, instance)
    horizon = max(node.due_time for node in instance.nodes) + max(10_000, len(instance.nodes) * 100)
    routing.AddDimension(transit_index, horizon, horizon, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")
    for node_index, node in enumerate(instance.nodes):
        index = manager.NodeToIndex(node_index)
        time_dimension.CumulVar(index).SetRange(int(node.ready_time), int(node.due_time))
    depot = instance.nodes[0]
    for vehicle_id in range(instance.vehicle_count):
        time_dimension.CumulVar(routing.Start(vehicle_id)).SetRange(depot.ready_time, depot.due_time)
        time_dimension.CumulVar(routing.End(vehicle_id)).SetRange(depot.ready_time, horizon)

    return _solve_routing(instance, routing, manager, distance_matrix, time_limit_seconds=time_limit_seconds)


def solve_cvrp_instance(instance: RouteInstance, *, time_limit_seconds: int = 5) -> dict[str, Any]:
    manager = pywrapcp.RoutingIndexManager(len(instance.nodes), instance.vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)
    distance_matrix = _distance_matrix(instance.nodes)

    def distance_callback(from_index: int, to_index: int) -> int:
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    _add_capacity_dimension(routing, manager, instance)
    return _solve_routing(instance, routing, manager, distance_matrix, time_limit_seconds=time_limit_seconds)


def run_ortools_route_benchmark_suite(
    *,
    output_dir: Path,
    data_dir: Path,
    time_limit_seconds: int = 5,
) -> dict[str, Any]:
    _prepare_output_dir(output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    instances = _download_and_parse_instances(data_dir)
    results = []
    for instance in instances:
        if instance.benchmark_family in {"solomon_vrptw", "homberger_200_vrptw"}:
            result = solve_vrptw_instance(instance, time_limit_seconds=time_limit_seconds)
        else:
            result = solve_cvrp_instance(instance, time_limit_seconds=time_limit_seconds)
        results.append(result)
    report = {
        "decision": _decision(results),
        "metadata": {
            "benchmark": "ortools_route_benchmark_suite",
            "solver": "Google OR-Tools RoutingModel",
            "ortools_version": _ortools_version(),
            "time_limit_seconds_per_instance": int(time_limit_seconds),
            "instance_count": len(results),
            "source_caveat": "Route-only classical optimization benchmark; not full 5PL digital twin.",
        },
        "sources": {
            "solomon_100": SOLOMON_ZIP_URL,
            "homberger_200": HOMBERGER_200_ZIP_URL,
            "cvrplib_instances": CVRPLIB_INSTANCE_URLS,
        },
        "results": results,
        "family_counts": dict(Counter(row["benchmark_family"] for row in results)),
        "success_counts": dict(
            Counter(row["benchmark_family"] for row in results if row["solver_status"] == "ROUTING_SUCCESS")
        ),
    }
    _write_json(output_dir / "ortools_route_benchmark_report.json", report)
    _write_summary_csv(output_dir / "ortools_route_benchmark_summary.csv", results)
    return report


def _download_and_parse_instances(data_dir: Path) -> list[RouteInstance]:
    solomon_zip = _download(SOLOMON_ZIP_URL, data_dir / "solomon-100.zip")
    homberger_zip = _download(HOMBERGER_200_ZIP_URL, data_dir / "homberger_200_customer_instances.zip")
    solomon_texts = _extract_zip_members(solomon_zip, SOLOMON_SELECTION)
    homberger_texts = _extract_zip_members(homberger_zip, HOMBERGER_SELECTION)

    instances: list[RouteInstance] = []
    for name in SOLOMON_SELECTION:
        instances.append(
            parse_solomon_vrptw_text(
                solomon_texts[name],
                name=name,
                benchmark_family="solomon_vrptw",
                source_url=SOLOMON_ZIP_URL,
            )
        )
    for name in HOMBERGER_SELECTION:
        instances.append(
            parse_solomon_vrptw_text(
                homberger_texts[name],
                name=name,
                benchmark_family="homberger_200_vrptw",
                source_url=HOMBERGER_200_ZIP_URL,
            )
        )
    cvrp_dir = data_dir / "cvrplib"
    cvrp_dir.mkdir(exist_ok=True)
    for name, url in CVRPLIB_INSTANCE_URLS.items():
        path = _download(url, cvrp_dir / f"{name}.vrp")
        instances.append(parse_cvrplib_text(path.read_text(encoding="utf-8"), source_url=url))
    return instances


def _solve_routing(
    instance: RouteInstance,
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    distance_matrix: Sequence[Sequence[int]],
    *,
    time_limit_seconds: int,
) -> dict[str, Any]:
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_seconds))

    start = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    elapsed = time.perf_counter() - start
    status = _routing_status_name(routing.status())
    result = {
        "instance_name": instance.name,
        "benchmark_family": instance.benchmark_family,
        "customer_count": len(instance.nodes) - 1,
        "vehicle_count": instance.vehicle_count,
        "capacity": instance.capacity,
        "solver_status": status,
        "solve_time_seconds": elapsed,
        "source_url": instance.source_url,
    }
    if solution is None:
        result.update({"objective_value": None, "vehicles_used": None, "total_distance": None, "routes": []})
        return result

    routes = []
    total_distance = 0
    vehicles_used = 0
    for vehicle_id in range(instance.vehicle_count):
        index = routing.Start(vehicle_id)
        route_nodes = []
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_nodes.append(node_index)
            previous = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous, index, vehicle_id)
        route_nodes.append(manager.IndexToNode(index))
        if len(route_nodes) > 2:
            vehicles_used += 1
            total_distance += route_distance
            routes.append({"vehicle_id": vehicle_id, "node_count": len(route_nodes), "distance": route_distance})
    result.update(
        {
            "objective_value": int(solution.ObjectiveValue()),
            "vehicles_used": vehicles_used,
            "total_distance": int(total_distance),
            "routes": routes,
        }
    )
    return result


def _add_capacity_dimension(
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    instance: RouteInstance,
) -> None:
    def demand_callback(from_index: int) -> int:
        return int(instance.nodes[manager.IndexToNode(from_index)].demand)

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [int(instance.capacity)] * instance.vehicle_count,
        True,
        "Capacity",
    )


def _distance_matrix(nodes: Sequence[RouteNode]) -> list[list[int]]:
    matrix = []
    for left in nodes:
        row = []
        for right in nodes:
            row.append(int(round(math.hypot(left.x - right.x, left.y - right.y))))
        matrix.append(row)
    return matrix


def _download(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "CODEX-PROJE-benchmark-suite/1.0"})
    with urlopen(request, timeout=120) as response, path.open("wb") as handle:
        handle.write(response.read())
    return path


def _extract_zip_members(zip_path: Path, names: Sequence[str]) -> dict[str, str]:
    wanted = {name.lower(): name for name in names}
    result: dict[str, str] = {}
    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            stem = Path(member).stem.lower()
            if stem in wanted and not member.endswith("/"):
                canonical = wanted[stem]
                result[canonical] = archive.read(member).decode("utf-8", errors="replace")
    missing = sorted(set(names) - set(result))
    if missing:
        raise FileNotFoundError(f"missing zip members in {zip_path}: {', '.join(missing)}")
    return result


def _numbers(line: str) -> list[float]:
    return [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", line)]


def _decision(results: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(row.get("benchmark_family") for row in results)
    required_counts = {
        "solomon_vrptw": 6,
        "homberger_200_vrptw": 3,
        "cvrplib_cvrp": 5,
    }
    for family, required in required_counts.items():
        if counts.get(family, 0) < required:
            return "OR_TOOLS_ROUTE_BENCHMARK_BLOCKED"
    return "OR_TOOLS_ROUTE_BENCHMARK_READY"


def _routing_status_name(status: int) -> str:
    status_map = {
        0: "ROUTING_NOT_SOLVED",
        1: "ROUTING_SUCCESS",
        2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
        3: "ROUTING_FAIL",
        4: "ROUTING_FAIL_TIMEOUT",
        5: "ROUTING_INVALID",
        6: "ROUTING_INFEASIBLE",
        7: "ROUTING_OPTIMAL",
    }
    return status_map.get(status, f"ROUTING_STATUS_{status}")


def _ortools_version() -> str:
    try:
        import ortools

        return str(ortools.__version__)
    except Exception:
        return "unknown"


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ortools_route_benchmark_report.json", "ortools_route_benchmark_summary.csv"):
        path = output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing OR-Tools benchmark output: {path}")


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "instance_name",
        "benchmark_family",
        "customer_count",
        "vehicle_count",
        "capacity",
        "solver_status",
        "objective_value",
        "vehicles_used",
        "total_distance",
        "solve_time_seconds",
        "source_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run expanded OR-Tools route benchmarks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/public/route_benchmarks_20260613"))
    parser.add_argument("--time-limit-seconds", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_ortools_route_benchmark_suite(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        time_limit_seconds=args.time_limit_seconds,
    )
    print(report["decision"])
    print(json.dumps(report["family_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
