# OpenMines Isolated Dispatch Report - 2026-06-14

## Classification

`OPENMINES_ISOLATED_BLOCKED_NO_COMPATIBLE_PYTHON`

JSON report:

- `reports/benchmarks/fleet_dispatch_upgrade_20260614/openmines_isolated/openmines_isolated_report.json`

## Finding

Only Python 3.12.9 is available through `python` and `py -0p`. `conda`, `mamba`, `uv`, and `virtualenv` are not on PATH.

OpenMines remains blocked because the package pins an older dependency stack including `numpy==1.25.0`; the previous install attempt failed under Python 3.12.

## Safe Next Step

Use an isolated Python 3.10 or 3.11 environment/container and run non-LLM dispatchers against `north_pit_mine_short.json`. Do not install OpenMines into the production Python 3.12 workspace.

