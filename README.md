# Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration

A fifth-party logistics (5PL) network is modelled as a discrete-event digital
twin, and a joint decision policy is trained inside it: a PPO branch drives five
continuous controls while a hierarchical DQN branch selects among 48 discrete
tactical actions, with a hard safety-projection layer filtering every command
before it reaches the simulator. Components are compared against rule-based
baselines under an equal evaluation budget, using paired per-episode deltas with
bootstrap confidence intervals and an exact sign test. The claim is deliberately
bounded to the simulator; no live deployment or causal real-world superiority is
asserted.

This repository is a sanitized research-artifact snapshot for a Python-based fifth-party logistics (5PL) digital twin. It supports thesis and reproducibility review for simulator-grounded autonomous supply-chain orchestration.

## What This Project Does

The project models a 5PL operating environment as a digital twin and evaluates a joint decision policy inside that simulator. The current promoted research artifact uses:

- a 73-feature observation contract;
- a PPO continuous-control branch with 5 continuous controls;
- a hierarchical DQN tactical branch with 48 discrete actions;
- a `torch_joint` runtime path;
- the `physical_reality_v5_route_candidate_visibility` simulator contract;
- protected exact-resume training evidence and governance reports.

## Why This Might Interest You

If you are reading this as an engineering artifact rather than as thesis
support, these are the parts with the most transferable content. Every claim
below points at the file that implements it.

**The safety layer is a shield, not a penalty term.**
[`src/act/safety_projector.py`](src/act/safety_projector.py) sits between the
policy and the SimPy simulation and rewrites or blocks commands that leave a
declared envelope — clamping speed multipliers, forcing `DISPATCH` to `HOLD`
above a capacity threshold, raising reorder fractions under backlog pressure —
and returns the reasons alongside the corrected action. Both the continuous and
the discrete branch pass through it (`project_continuous`, `project_discrete`).
This places the mechanism in the same family as the shielding approach of
Alshiekh et al. (2018), *Safe Reinforcement Learning via Shielding*, AAAI, where
a shield corrects unsafe actions before execution rather than relying on the
reward signal to discourage them. Its behaviour is designed to be measured
rather than assumed: see
[`configs/ablation/safety_projection_diagnostic.json`](configs/ablation/safety_projection_diagnostic.json),
which records projection and block rates while safety stays enabled.

**Reward credit is tied to realised effect, not to the action label.**
In [`src/act/env_5pl.py`](src/act/env_5pl.py), `_dqn_local_reward_components`
gates dispatch credit on work actually happening: the dispatch signal requires
both the `dispatch` label *and* a non-zero count of dispatched or successfully
dispatched orders. The intent is pinned down by executable checks in
[`src/learn/pretrain_mdp_sanity_check.py`](src/learn/pretrain_mdp_sanity_check.py),
which holds 39 deterministic assertions about the reward contract. Two are
directly about label-versus-effect:

- `_check_dqn_dispatch_progress_credit_requires_successful_dispatch` issues the
  identical `dispatch` action twice and asserts that dispatch progress credit is
  exactly `0.0` when no orders moved and positive when they did.
- `_check_hold_route_labels_are_neutral` asserts that `shortest`,
  `low_congestion` and `high_resilience` route labels produce identical local
  reward while the policy is holding, so a route label earns nothing on its own.

Companion checks extend the same principle to other controls —
`_check_speed_credit_requires_actual_movement`,
`_check_global_flow_credit_requires_real_pressure_reduction`,
`_check_emergency_credit_requires_useful_override_units`. The suite is enforced
as a test in
[`tests/learn/test_pretrain_mdp_sanity_check.py`](tests/learn/test_pretrain_mdp_sanity_check.py).

**The comparison protocol is fixed before the numbers.**
[`src/eval/benchmark_statistics.py`](src/eval/benchmark_statistics.py) provides
`bootstrap_mean_ci` (1000 resamples, `alpha=0.05`, fixed `seed=42`) and
`exact_sign_test_two_sided`, an exact two-sided binomial sign test over paired
deltas. They are applied to paired per-episode deltas against an equal-budget
reference in
[`scripts/full_completion_rule_based_benchmark.py`](scripts/full_completion_rule_based_benchmark.py)
(`paired_service_delta_bootstrap_ci`, `paired_service_delta_sign_test`), with
the equal-budget pairing itself in
[`src/eval/rule_based_baseline_arena.py`](src/eval/rule_based_baseline_arena.py)
(`_compare_to_hierarchical_equal_budget`). The controls that make the budget
equal — same scenario configs, same episode count, same seed policy,
deterministic inference — are written down in
[`docs/thesis/ablation_protocol.md`](docs/thesis/ablation_protocol.md) under
*Required Controls*. Statistics helpers have their own tests in
[`tests/eval/test_benchmark_statistics.py`](tests/eval/test_benchmark_statistics.py).

**Off-policy evaluation limits are stated, not skirted.**
[`MODEL_CARD.md`](MODEL_CARD.md) records under *Limitations* that public replay
data is descriptive proxy evidence and lacks the state, action propensities,
rewards and operational contracts required for causal off-policy evaluation.
[`ADVISOR_QUICK_READ.md`](ADVISOR_QUICK_READ.md) lists causal off-policy
evaluation from public replay among the things explicitly not claimed, and
[`docs/thesis/ablation_protocol.md`](docs/thesis/ablation_protocol.md) closes
with a claim boundary limiting ablation evidence to component-level
interpretation inside the simulator.

Three caveats worth knowing before you dig in: no ablation results are claimed
unless a generated report is present, several evaluation paths need artifacts
that are deliberately excluded from this snapshot (see *What Is Intentionally
Excluded*), and known defects are tracked in
[`docs/governance/known_issues.md`](docs/governance/known_issues.md).

## Prospective Advisor Quick Read

For a concise research overview, see [ADVISOR_QUICK_READ.md](ADVISOR_QUICK_READ.md).

For the planned ablation protocol, see [docs/thesis/ablation_protocol.md](docs/thesis/ablation_protocol.md).

For professor-facing positioning and public/private scope, see [docs/thesis/research_positioning.md](docs/thesis/research_positioning.md) and [docs/thesis/public_private_boundary.md](docs/thesis/public_private_boundary.md).

## Architecture Boundary

This repository does not claim live deployment into a TMS, WMS, ERP, carrier marketplace, or production operations stack. Autonomy is bounded to simulator/runtime decisioning and offline evidence. Public replay artifacts are descriptive proxy evidence only; they are not causal off-policy evaluation and do not prove real-world superiority.

## Repository Layout

- `src/`: simulator, policy runtime, learning, evaluation, and orchestration code.
- `scripts/`: safe analysis, reporting, public-proxy, benchmark, and dashboard utilities.
- `configs/`: simulator, curriculum, reward, and evaluation configuration files.
- `tests/`: unit and integration tests available in the source tree.
- `docs/`: selected runbooks, release notes, plans, and bounded claim documentation.
- `reports/`: selected lightweight summaries, thesis handoff metadata, dashboard assets, and final benchmark scorecard artifacts.

### Governance Documents

The root directory carries five short documents that define what may be claimed,
what may be run, and how the artifact should be cited.

| File | What it is for |
| --- | --- |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Model identity, intended and out-of-scope use, stated limitations, and ablation status. Read this before interpreting any result. |
| [`GOVERNANCE.md`](GOVERNANCE.md) | Protected-artifact policy and the registry/production boundary. Defines which operations need separate approval and which claims are off limits. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Safe verification path, what is excluded from the snapshot, exact-resume evidence pointers, and the list of operations not to run by default. |
| [`SECURITY.md`](SECURITY.md) | Private vulnerability reporting, the sanitization boundary, and secret-handling rules. |
| [`CITATION.cff`](CITATION.cff) | Machine-readable citation metadata used by GitHub and reference managers. |

## Requirements

**Python 3.11 or newer.** Two independent constraints set that floor:

- The source imports runtime APIs added in 3.11: `enum.StrEnum`
  (`src/act/discrete_action_mapper.py`), `datetime.UTC`
  (`src/learn/joint_metrics.py`), and `typing.Self` (`src/sense/db_pool.py`).
- Pinned dependencies require it: `numpy==2.4.4` and `pandas==3.0.2` each
  declare `Requires-Python >=3.11`.

Continuous integration builds against Python 3.12 on Windows
(`.github/workflows/ci.yml`), so 3.12 is the version with automated coverage.

## Safe Quickstart

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
find src scripts tests -name '*.py' -exec python -m py_compile {} +
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m py_compile (Get-ChildItem -Recurse src,scripts,tests -Filter *.py | ForEach-Object FullName)
```

Both compile commands cover the same set of source files. If you prefer one
invocation that is identical on every platform, `python -m compileall -q src
scripts tests` does the same job.

The compile step is intentionally lightweight. Some tests need optional research dependencies or local artifacts and should be selected deliberately.

## What Is Intentionally Excluded

This sanitized repository excludes model checkpoints, production model binaries, registry state, baseline artifacts, evaluation episode bodies, databases, raw public/private datasets, environment files, DevSpace/tunnel material, and local automation state. See `.gitignore`, `EXCLUDED_AMBIGUOUS_FILES.md`, and `GITHUB_REPO_PREP_REPORT.md` for details.

## Lightweight Reproduction Checks

Recommended safe checks are shown below. Run them from an activated virtual
environment, where `python` resolves to the interpreter inside `.venv`.

### Linux and macOS

```bash
find src scripts tests -name '*.py' -exec python -m py_compile {} +
python scripts/production_artifact_health_report.py --help
python scripts/monitoring_report_generator.py --help
python scripts/company_data_intake_validator.py --help
```

### Windows (PowerShell)

```powershell
python -m py_compile (Get-ChildItem -Recurse src,scripts,tests -Filter *.py | ForEach-Object FullName)
python scripts\production_artifact_health_report.py --help
python scripts\monitoring_report_generator.py --help
python scripts\company_data_intake_validator.py --help
```

Each `--help` call prints the argument contract and exits without touching
models, registry state, or data.

Do not run training, offline evaluation gates, registry activation, production promotion, dataset downloads, or dashboard servers unless a separate approval explicitly authorizes that action.

The repository ships `start_control_room.bat` and `start_control_room.ps1` as
Windows launchers for the control room dashboard. No shell-script equivalent is
provided and none is required: both scripts locate an interpreter and then
invoke `python scripts/control_room_server.py`, which is the same command on
Linux and macOS. Launching it remains subject to the approval rule above.

## Thesis and Citation Note

Official project title: **Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration**. The repository is a thesis-supporting software and evidence artifact. It should be cited using `CITATION.cff` once the human owner confirms final publication metadata.
