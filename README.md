# Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration

This repository is a sanitized research-artifact snapshot for a Python-based fifth-party logistics (5PL) digital twin. It supports thesis and reproducibility review for simulator-grounded autonomous supply-chain orchestration.

## What This Project Does

The project models a 5PL operating environment as a digital twin and evaluates a joint decision policy inside that simulator. The current promoted research artifact uses:

- a 73-feature observation contract;
- a PPO continuous-control branch with 5 continuous controls;
- a hierarchical DQN tactical branch with 48 discrete actions;
- a `torch_joint` runtime path;
- the `physical_reality_v5_route_candidate_visibility` simulator contract;
- protected exact-resume training evidence and governance reports.

## Prospective Advisor Quick Read

For a concise research overview, see [ADVISOR_QUICK_READ.md](ADVISOR_QUICK_READ.md).

For the planned ablation protocol, see [docs/thesis/ablation_protocol.md](docs/thesis/ablation_protocol.md).

## Architecture Boundary

This repository does not claim live deployment into a TMS, WMS, ERP, carrier marketplace, or production operations stack. Autonomy is bounded to simulator/runtime decisioning and offline evidence. Public replay artifacts are descriptive proxy evidence only; they are not causal off-policy evaluation and do not prove real-world superiority.

## Repository Layout

- `src/`: simulator, policy runtime, learning, evaluation, and orchestration code.
- `scripts/`: safe analysis, reporting, public-proxy, benchmark, and dashboard utilities.
- `configs/`: simulator, curriculum, reward, and evaluation configuration files.
- `tests/`: unit and integration tests available in the source tree.
- `docs/`: selected runbooks, release notes, plans, and bounded claim documentation.
- `reports/`: selected lightweight summaries, thesis handoff metadata, dashboard assets, and final benchmark scorecard artifacts.

## Safe Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m py_compile (Get-ChildItem -Recurse src,scripts,tests -Filter *.py | ForEach-Object FullName)
```

The compile command above is intentionally lightweight. Some tests need optional research dependencies or local artifacts and should be selected deliberately.

## What Is Intentionally Excluded

This sanitized repository excludes model checkpoints, production model binaries, registry state, baseline artifacts, evaluation episode bodies, databases, raw public/private datasets, environment files, DevSpace/tunnel material, and local automation state. See `.gitignore`, `EXCLUDED_AMBIGUOUS_FILES.md`, and `GITHUB_REPO_PREP_REPORT.md` for details.

## Lightweight Reproduction Checks

Recommended safe checks are:

```powershell
python -m py_compile (Get-ChildItem -Recurse src,scripts,tests -Filter *.py | ForEach-Object FullName)
python scripts\production_artifact_health_report.py --help
python scripts\monitoring_report_generator.py --help
python scripts\company_data_intake_validator.py --help
```

Do not run training, offline evaluation gates, registry activation, production promotion, dataset downloads, or dashboard servers unless a separate approval explicitly authorizes that action.

## Thesis and Citation Note

Official project title: **Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration**. The repository is a thesis-supporting software and evidence artifact. It should be cited using `CITATION.cff` once the human owner confirms final publication metadata.
