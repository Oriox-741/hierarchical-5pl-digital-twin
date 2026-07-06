"""PyVRP route-reference benchmark over existing public VRP instances."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True, slots=True)
class RouteInstance:
    path: Path
    family: str


@dataclass(frozen=True, slots=True)
class SolomonCustomer:
    node_id: int
    x: float
    y: float
    demand: int
    ready_time: int
    due_date: int
    service_time: int


@dataclass(frozen=True, slots=True)
class SolomonInstance:
    name: str
    vehicle_count: int
    capacity: int
    customers: tuple[SolomonCustomer, ...]


def select_instances(data_dir: Path, *, max_cvrp: int = 5, max_vrptw: int = 3) -> list[RouteInstance]:
    data_dir = Path(data_dir)
    selected: list[RouteInstance] = []
    cvrp_dir = data_dir / "cvrplib"
    if cvrp_dir.exists():
        for path in sorted(cvrp_dir.glob("*.vrp"))[: int(max_cvrp)]:
            selected.append(RouteInstance(path=path, family="cvrplib_cvrp"))

    vrptw_candidates = _vrptw_files(data_dir)
    for path in vrptw_candidates[: int(max_vrptw)]:
        selected.append(RouteInstance(path=path, family="vrptw_reference"))
    return selected


def parse_solomon_txt(path: Path) -> SolomonInstance:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if not non_empty:
        raise ValueError(f"empty Solomon instance: {path}")
    name = non_empty[0]

    vehicle_count: int | None = None
    capacity: int | None = None
    for idx, line in enumerate(lines):
        if line.strip().upper() == "VEHICLE":
            for candidate in lines[idx + 1 : idx + 5]:
                parts = candidate.split()
                if len(parts) >= 2 and all(part.lstrip("-").isdigit() for part in parts[:2]):
                    vehicle_count = int(parts[0])
                    capacity = int(parts[1])
                    break
            break
    if vehicle_count is None or capacity is None:
        raise ValueError(f"could not parse vehicle count/capacity from {path}")

    customers: list[SolomonCustomer] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7 or not parts[0].lstrip("-").isdigit():
            continue
        customers.append(
            SolomonCustomer(
                node_id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                demand=int(float(parts[3])),
                ready_time=int(float(parts[4])),
                due_date=int(float(parts[5])),
                service_time=int(float(parts[6])),
            )
        )
    if len(customers) < 2 or customers[0].node_id != 0:
        raise ValueError(f"expected depot plus customers in {path}")
    return SolomonInstance(name=name, vehicle_count=vehicle_count, capacity=capacity, customers=tuple(customers))


def build_solomon_model(instance: SolomonInstance) -> Any:
    from pyvrp import Model

    model = Model()
    depot_src = instance.customers[0]
    depot = model.add_depot(
        x=depot_src.x,
        y=depot_src.y,
        tw_early=depot_src.ready_time,
        tw_late=depot_src.due_date,
        service_duration=depot_src.service_time,
        name=str(depot_src.node_id),
    )
    clients = [
        model.add_client(
            x=customer.x,
            y=customer.y,
            delivery=customer.demand,
            service_duration=customer.service_time,
            tw_early=customer.ready_time,
            tw_late=customer.due_date,
            name=str(customer.node_id),
        )
        for customer in instance.customers[1:]
    ]
    model.add_vehicle_type(
        num_available=instance.vehicle_count,
        capacity=instance.capacity,
        start_depot=depot,
        end_depot=depot,
        tw_early=depot_src.ready_time,
        tw_late=depot_src.due_date,
    )
    locations = [depot, *clients]
    source_customers = list(instance.customers)
    for frm_idx, frm in enumerate(locations):
        for to_idx, to in enumerate(locations):
            if frm_idx == to_idx:
                continue
            src = source_customers[frm_idx]
            dst = source_customers[to_idx]
            distance = int(round(math.hypot(src.x - dst.x, src.y - dst.y)))
            model.add_edge(frm, to, distance=distance, duration=distance)
    return model


def build_result_row(
    *,
    instance: RouteInstance,
    objective: float | None,
    runtime_seconds: float,
    feasible: bool,
    routes_used: int | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "instance": instance.path.name,
        "path": str(instance.path),
        "family": instance.family,
        "status": "success" if error is None and feasible else "blocked",
        "objective": objective,
        "runtime_seconds": runtime_seconds,
        "feasible": feasible,
        "routes_used": routes_used,
        "error": error,
    }


def run_pyvrp_benchmark(
    *,
    data_dir: Path,
    time_limit_seconds: float = 2.0,
    max_cvrp: int = 5,
    max_vrptw: int = 3,
) -> dict[str, Any]:
    instances = select_instances(data_dir, max_cvrp=max_cvrp, max_vrptw=max_vrptw)
    if not instances:
        return {
            "benchmark_metadata": {
                "benchmark": "pyvrp_route_reference",
                "data_dir": str(data_dir),
            },
            "decision": "PYVRP_ROUTE_REFERENCE_BLOCKED",
            "blocker": "no readable CVRP/VRPTW instances found",
            "results": [],
        }

    try:
        import pyvrp
        from pyvrp.stop import MaxRuntime
    except Exception as exc:  # pragma: no cover - dependency-path guard
        return {
            "benchmark_metadata": {
                "benchmark": "pyvrp_route_reference",
                "data_dir": str(data_dir),
            },
            "decision": "PYVRP_ROUTE_REFERENCE_BLOCKED",
            "blocker": f"pyvrp_import_failed: {exc}",
            "results": [],
        }

    rows = []
    for instance in instances:
        started = time.perf_counter()
        try:
            if instance.family == "vrptw_reference":
                problem = build_solomon_model(parse_solomon_txt(instance.path)).data()
            else:
                problem = pyvrp.read(instance.path)
            result = pyvrp.solve(problem, stop=MaxRuntime(float(time_limit_seconds)), seed=42)
            runtime = time.perf_counter() - started
            solution = getattr(result, "best", None)
            feasible = bool(result.is_feasible()) if hasattr(result, "is_feasible") else solution is not None
            objective = _call_numeric(result, "cost")
            if objective is None and solution is not None:
                objective = _call_numeric(solution, "cost")
            routes_used = _routes_used(solution)
            rows.append(
                build_result_row(
                    instance=instance,
                    objective=objective,
                    runtime_seconds=runtime,
                    feasible=feasible,
                    routes_used=routes_used,
                    error=None,
                )
            )
        except Exception as exc:
            rows.append(
                build_result_row(
                    instance=instance,
                    objective=None,
                    runtime_seconds=time.perf_counter() - started,
                    feasible=False,
                    routes_used=None,
                    error=type(exc).__name__ + ": " + str(exc),
                )
            )

    successful = [row for row in rows if row["status"] == "success"]
    decision = "PYVRP_ROUTE_REFERENCE_READY" if successful else "PYVRP_ROUTE_REFERENCE_BLOCKED"
    return {
        "benchmark_metadata": {
            "benchmark": "pyvrp_route_reference",
            "data_dir": str(data_dir),
            "time_limit_seconds": float(time_limit_seconds),
            "instance_count": len(instances),
            "successful_instances": len(successful),
            "route_only_caveat": "PyVRP validates route solver references only, not full 5PL dispatch/fleet/reorder control.",
        },
        "decision": decision,
        "results": rows,
    }


def write_outputs(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "pyvrp_benchmark_report.json"
    csv_path = output_dir / "pyvrp_benchmark_summary.csv"
    if report_path.exists() or csv_path.exists():
        raise FileExistsError(f"refusing to overwrite PyVRP benchmark outputs in {output_dir}")
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "instance",
        "family",
        "status",
        "objective",
        "runtime_seconds",
        "feasible",
        "routes_used",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report.get("results", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field) for field in fields})


def _vrptw_files(data_dir: Path) -> list[Path]:
    extracted_root = data_dir / "pyvrp_extracted_vrptw"
    candidates = list(extracted_root.rglob("*.txt")) if extracted_root.exists() else []
    if not candidates:
        for archive in sorted(data_dir.glob("*.zip")):
            try:
                with zipfile.ZipFile(archive) as zf:
                    small_members = [
                        member
                        for member in zf.namelist()
                        if member.lower().endswith(".txt")
                        and not member.endswith("/")
                        and ("r1" in member.lower() or "c1" in member.lower() or "rc1" in member.lower())
                    ][:6]
                    for member in small_members:
                        zf.extract(member, extracted_root)
            except zipfile.BadZipFile:
                continue
        candidates = list(extracted_root.rglob("*.txt")) if extracted_root.exists() else []
    return sorted(candidates)


def _call_numeric(obj: Any, name: str) -> float | None:
    attr = getattr(obj, name, None)
    if not callable(attr):
        return None
    try:
        value = float(attr())
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _routes_used(solution: Any) -> int | None:
    if solution is None:
        return None
    routes = getattr(solution, "routes", None)
    if callable(routes):
        try:
            return len(routes())
        except Exception:
            return None
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PyVRP route-reference benchmark.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/public/route_benchmarks_20260613"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=2.0)
    parser.add_argument("--max-cvrp", type=int, default=5)
    parser.add_argument("--max-vrptw", type=int, default=3)
    args = parser.parse_args(argv)
    report = run_pyvrp_benchmark(
        data_dir=args.data_dir,
        time_limit_seconds=args.time_limit_seconds,
        max_cvrp=args.max_cvrp,
        max_vrptw=args.max_vrptw,
    )
    write_outputs(args.output_dir, report)
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
