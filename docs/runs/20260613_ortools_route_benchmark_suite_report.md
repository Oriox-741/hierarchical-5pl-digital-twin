# OR-Tools Route Benchmark Suite Report

Final classification: `OR_TOOLS_ROUTE_BENCHMARK_READY`

Instance counts: `{'cvrplib_cvrp': 5, 'homberger_200_vrptw': 3, 'solomon_vrptw': 6}`.
Successful solve counts within 5s: `{'cvrplib_cvrp': 5, 'homberger_200_vrptw': 2, 'solomon_vrptw': 4}`.

Outputs:
- `reports/benchmarks/full_completion_20260613/ortools_route_benchmarks/ortools_route_benchmark_report.json`
- `reports/benchmarks/full_completion_20260613/ortools_route_benchmarks/ortools_route_benchmark_summary.csv`

## Instance Results

| Instance | Family | Customers | Status | Objective | Vehicles used | Solve seconds |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `C101` | `solomon_vrptw` | 100 | `ROUTING_SUCCESS` | 9829 | 10 | 5.008 |
| `C201` | `solomon_vrptw` | 100 | `ROUTING_SUCCESS` | 9590 | 3 | 5.000 |
| `R101` | `solomon_vrptw` | 100 | `ROUTING_FAIL_TIMEOUT` |  |  | 5.001 |
| `R201` | `solomon_vrptw` | 100 | `ROUTING_SUCCESS` | 2217 | 7 | 5.000 |
| `RC101` | `solomon_vrptw` | 100 | `ROUTING_FAIL_TIMEOUT` |  |  | 5.001 |
| `RC201` | `solomon_vrptw` | 100 | `ROUTING_SUCCESS` | 2280 | 9 | 5.011 |
| `c1_2_1` | `homberger_200_vrptw` | 200 | `ROUTING_SUCCESS` | 20939 | 21 | 5.000 |
| `c2_2_1` | `homberger_200_vrptw` | 200 | `ROUTING_SUCCESS` | 20005 | 7 | 5.001 |
| `r1_2_1` | `homberger_200_vrptw` | 200 | `ROUTING_FAIL_TIMEOUT` |  |  | 5.004 |
| `A-n32-k5` | `cvrplib_cvrp` | 31 | `ROUTING_SUCCESS` | 784 | 5 | 5.000 |
| `A-n33-k5` | `cvrplib_cvrp` | 32 | `ROUTING_SUCCESS` | 661 | 5 | 5.000 |
| `A-n33-k6` | `cvrplib_cvrp` | 32 | `ROUTING_SUCCESS` | 742 | 6 | 5.001 |
| `A-n34-k5` | `cvrplib_cvrp` | 33 | `ROUTING_SUCCESS` | 786 | 5 | 5.001 |
| `A-n36-k5` | `cvrplib_cvrp` | 35 | `ROUTING_SUCCESS` | 799 | 5 | 5.002 |

Caveat: this is a route-only classical optimization benchmark. It does not validate dispatch, fleet-mode, reorder, inventory, or company-cost decisions.
