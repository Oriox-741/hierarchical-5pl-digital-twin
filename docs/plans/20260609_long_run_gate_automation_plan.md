# Long-Run Gate Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only long-run gate tooling so future 3M, 5M, 10M, or 100M experiments cannot proceed without scenario pass gates, hard-blocker gates, production comparison, previous-gate comparison, and anti-overdispatch checks.

**Architecture:** Implement a pure summary-file checker that reads existing `scenario_summary.json` outputs and emits a gate report to stdout. Keep it separate from offline evaluation, training, registry, checkpoint, production, baseline, and DB code paths. The checker compares a candidate eval against production and an optional previous gate, returning a non-zero exit code on fatal gate failures and warnings for residual watch regressions.

**Tech Stack:** Python `unittest`, JSON summary artifacts from `src.eval.real_world_scenario_arena.write_evaluation_outputs`, `src.eval.scenario_metrics.GLOBAL_FATAL_HARD_BLOCKER_FIELDS`, no PyTorch dependency, no model loading, no file writes by default.

---

## Source Evidence

Reference plan:

`docs/plans/20260609_nextgen_longrun_redesign_plan.md`

Current production eval:

`models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json`

Observed production status:

- 8/8 scenarios PASS.
- Hard-blocker failure lists empty.
- Contract `physical_reality_v5_route_candidate_visibility`.
- Observation dim `73`.

Stopped next-gen 1M eval:

`models/eval/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios/scenario_summary.json`

Observed next-gen 1M status:

- 6/8 scenarios PASS.
- Failed scenarios: `premium_sla_pressure`, `route_disruption_congestion`.
- Hard blockers zero.
- Failure class: overtrained dispatch aggression.

Next-gen 300k intermediate eval:

`models/eval/joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios/scenario_summary.json`

Observed next-gen 300k status:

- 2/8 scenarios PASS.
- Failed scenarios: baseline, demand spike, high holding cost, lead time, mixed stress, route disruption.
- Hard blockers zero.
- Failure class: under-serving and lateness.

## Scenario Summary Schema

Top-level keys observed:

- `checkpoint`
- `contract`
- `observation_dim`
- `scenarios`

Each `scenarios` entry is a dictionary containing gate-relevant fields:

- Identity and verdicts:
  - `scenario_id`
  - `verdict`
  - `scenario_threshold_verdict`
  - `scenario_threshold_failures`
  - `hard_blocker_verdict`
  - `hard_blocker_failures`
- Core metrics:
  - `service_level`
  - `true_lateness_pressure`
  - `dispatch_success_per_attempt`
  - `dispatch_rate`
  - `hold_rate`
  - `route_failures`
  - `no_vehicle_available`
  - `already_assigned_context_rows`
- Distribution and action quality:
  - `route_distribution`
  - `route_distribution_successful_dispatch`
  - `route_distribution_failed_dispatch`
  - `route_distribution_hold`
  - `fleet_distribution`
  - `reorder_mode_distribution`
  - `top_action_ids`
  - `top_failed_noop_actions`
  - `action_failed_noop_by_id`
  - `action_no_current_work_by_id`
  - `action_no_unassigned_by_id`
  - `action_dispatch_success_ratio_by_id`
- Premium/route/mixed watches:
  - `premium_missed_useful_dispatch_rate`
  - `premium_missed_useful_reorder_rate`
  - `premium_hold_under_useful_dispatch_opportunity`
  - `high_resilience_selected_when_not_best_count`
  - `mixed_success_route_failure_steps`
  - `mixed_route_overconservative_candidate_count`
- Global fatal hard blockers from `src.eval.scenario_metrics.GLOBAL_FATAL_HARD_BLOCKER_FIELDS`:
  - `fake_dispatch_credit`
  - `customer_revisited`
  - `route_failure_positive_dispatch_credit`
  - `nan_inf_detected`
  - `dqn_local_negative_positive_train_rows`
  - `no_current_work_dqn_delivery_credit`
  - `hold_delivery_credit_leak`
  - `action8_route_or_delivery_credit_leak`
  - `unsafe_24_25_candidate_credit`
  - `no_work_positive_dqn_local`
  - `emergency_zero_useful_positive_credit`

## Gate Semantics

The checker must produce:

- `decision`: `PASS` or `FAIL`
- `fatal_failures`: list of strings
- `warnings`: list of strings
- `candidate_label`
- `candidate_checkpoint`
- `production_checkpoint`
- `previous_checkpoint`, when provided
- per-scenario comparison rows

### Fatal Failure Rules

Fail the gate when any of these are true:

1. Any candidate scenario has `scenario_threshold_verdict != "PASS"`.
2. Any candidate scenario has non-empty `scenario_threshold_failures`.
3. Any candidate scenario has `hard_blocker_verdict != "PASS"`.
4. Any candidate scenario has non-empty `hard_blocker_failures`.
5. Any candidate scenario has a non-zero value for any global fatal hard-blocker field.
6. Candidate `premium_sla_pressure.dispatch_success_per_attempt` is below production by more than `0.005`.
7. Candidate `route_disruption_congestion.dispatch_success_per_attempt` is below production by more than `0.005`.
8. Candidate `baseline_normal.service_level` is below production by more than `0.005`.
9. Candidate has a no-current-work/no-unassigned top-action explosion:
   - For any scenario, `sum(action_no_current_work_by_id) + sum(action_no_unassigned_by_id)` is greater than `max(production_total * 2.0, production_total + 50)`.
   - Or any action id has `action_no_current_work_by_id[action] + action_no_unassigned_by_id[action] >= 100` and the same action appears in `top_action_ids` with percentage at least `0.10`.

### Warning Rules

Warn, but do not fail, when any of these are true and no fatal rule already covers the same scenario:

1. Candidate service is lower than production by more than `0.005` in any non-baseline scenario.
2. Candidate lateness is higher than production by more than `0.005` in any scenario.
3. Candidate dispatch rate is higher than production by more than `0.15`.
4. Candidate route disruption `high_resilience_selected_when_not_best_count` is higher than production.
5. Candidate `mixed_success_route_failure_steps` is higher than production.
6. Candidate mixed stress service is lower than production by more than `0.005`.
7. Candidate demand spike lateness is higher than production.
8. Candidate top action concentration exceeds production's top action percentage by more than `0.15`.
9. Candidate previous-gate service regresses by more than `0.005`, when previous gate is provided.
10. Candidate previous-gate dispatch success regresses by more than `0.005`, when previous gate is provided.

## Files To Create

- Create: `src/eval/long_run_gate.py`
  - Pure functions for loading summaries, validating schema, computing fatal failures, computing warnings, and building report dictionaries.
- Create: `src/eval/check_long_run_gate.py`
  - CLI wrapper around `src.eval.long_run_gate`.
  - Reads existing JSON files.
  - Prints JSON report to stdout.
  - Exits `0` on PASS, `2` on FAIL.
  - Does not write files unless a future explicitly approved `--output-json` flag is added.
- Create: `tests/eval/test_long_run_gate.py`
  - Unit tests with temporary JSON summaries.
  - No training, no evaluation, no model loading.

## Task 1: Add Test Fixtures And Schema Loader Tests

**Files:**
- Create: `tests/eval/test_long_run_gate.py`
- Future create: `src/eval/long_run_gate.py`

- [ ] **Step 1: Write failing tests for summary loading**

Create `tests/eval/test_long_run_gate.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.eval.long_run_gate import load_summary


def _scenario(
    scenario_id: str,
    *,
    verdict: str = "PASS",
    threshold: str = "PASS",
    hard: str = "PASS",
    service: float = 0.95,
    lateness: float = 0.0,
    dispatch_success: float = 1.0,
    dispatch_rate: float = 0.5,
    hard_blocker_failures: list[str] | None = None,
    scenario_threshold_failures: list[str] | None = None,
    action_no_current_work_by_id: dict[str, int] | None = None,
    action_no_unassigned_by_id: dict[str, int] | None = None,
    top_action_ids: list[dict[str, float | int]] | None = None,
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "verdict": verdict,
        "scenario_threshold_verdict": threshold,
        "scenario_threshold_failures": scenario_threshold_failures or [],
        "hard_blocker_verdict": hard,
        "hard_blocker_failures": hard_blocker_failures or [],
        "service_level": service,
        "true_lateness_pressure": lateness,
        "dispatch_success_per_attempt": dispatch_success,
        "dispatch_rate": dispatch_rate,
        "hold_rate": 1.0 - dispatch_rate,
        "fake_dispatch_credit": 0,
        "customer_revisited": 0,
        "route_failure_positive_dispatch_credit": 0,
        "nan_inf_detected": 0,
        "dqn_local_negative_positive_train_rows": 0,
        "no_current_work_dqn_delivery_credit": 0,
        "hold_delivery_credit_leak": 0,
        "action8_route_or_delivery_credit_leak": 0,
        "unsafe_24_25_candidate_credit": 0,
        "no_work_positive_dqn_local": 0,
        "emergency_zero_useful_positive_credit": 0,
        "action_no_current_work_by_id": action_no_current_work_by_id or {},
        "action_no_unassigned_by_id": action_no_unassigned_by_id or {},
        "top_action_ids": top_action_ids or [],
        "high_resilience_selected_when_not_best_count": 0,
        "mixed_success_route_failure_steps": 0,
    }


def _summary(path: Path, scenarios: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint": "models/checkpoints/candidate.pt",
                "contract": "physical_reality_v5_route_candidate_visibility",
                "observation_dim": 73,
                "scenarios": scenarios,
            }
        ),
        encoding="utf-8",
    )


class LongRunGateLoaderTests(unittest.TestCase):
    def test_load_summary_indexes_scenarios_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario_summary.json"
            _summary(path, [_scenario("baseline_normal"), _scenario("premium_sla_pressure")])

            loaded = load_summary(path)

            self.assertEqual(loaded.checkpoint, "models/checkpoints/candidate.pt")
            self.assertEqual(loaded.contract, "physical_reality_v5_route_candidate_visibility")
            self.assertEqual(loaded.observation_dim, 73)
            self.assertEqual(sorted(loaded.scenarios), ["baseline_normal", "premium_sla_pressure"])

    def test_load_summary_rejects_missing_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario_summary.json"
            path.write_text(json.dumps({"checkpoint": "x"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scenarios"):
                load_summary(path)
```

- [ ] **Step 2: Run RED verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- FAIL with `ModuleNotFoundError: No module named 'src.eval.long_run_gate'`.

- [ ] **Step 3: Implement minimal loader**

Create `src/eval/long_run_gate.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EvalSummary:
    path: Path
    checkpoint: str
    contract: str
    observation_dim: int
    scenarios: dict[str, Mapping[str, Any]]


def load_summary(path: Path) -> EvalSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("scenario_summary.json must contain a scenarios list.")
    indexed: dict[str, Mapping[str, Any]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario entries must be mappings.")
        scenario_id = scenario.get("scenario_id")
        if not scenario_id:
            raise ValueError("scenario entry missing scenario_id.")
        indexed[str(scenario_id)] = scenario
    return EvalSummary(
        path=path,
        checkpoint=str(payload.get("checkpoint", "")),
        contract=str(payload.get("contract", "")),
        observation_dim=int(payload.get("observation_dim", 0)),
        scenarios=indexed,
    )
```

- [ ] **Step 4: Run GREEN verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- PASS.

## Task 2: Add Fatal Gate Rules

**Files:**
- Modify: `tests/eval/test_long_run_gate.py`
- Modify: `src/eval/long_run_gate.py`

- [ ] **Step 1: Write failing fatal-rule tests**

Append tests:

```python
from src.eval.long_run_gate import evaluate_gate


class LongRunGateFatalTests(unittest.TestCase):
    def _write_pair(
        self,
        tmp: str,
        candidate_scenarios: list[dict[str, object]],
        production_scenarios: list[dict[str, object]] | None = None,
    ) -> tuple[Path, Path]:
        root = Path(tmp)
        candidate = root / "candidate.json"
        production = root / "production.json"
        _summary(candidate, candidate_scenarios)
        _summary(production, production_scenarios or candidate_scenarios)
        return candidate, production

    def test_gate_fails_on_any_scenario_threshold_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, production = self._write_pair(
                tmp,
                [
                    _scenario(
                        "baseline_normal",
                        verdict="FAIL",
                        threshold="FAIL",
                        service=0.80,
                        scenario_threshold_failures=["service_level 0.800 < 0.930"],
                    )
                ],
                [_scenario("baseline_normal", service=0.95)],
            )

            report = evaluate_gate(candidate_summary=candidate, production_summary=production)

            self.assertEqual(report["decision"], "FAIL")
            self.assertTrue(any("scenario threshold" in item for item in report["fatal_failures"]))

    def test_gate_fails_on_hard_blocker_field_even_if_verdict_string_is_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = _scenario("baseline_normal")
            bad["fake_dispatch_credit"] = 1
            candidate, production = self._write_pair(tmp, [bad], [_scenario("baseline_normal")])

            report = evaluate_gate(candidate_summary=candidate, production_summary=production)

            self.assertEqual(report["decision"], "FAIL")
            self.assertTrue(any("fake_dispatch_credit" in item for item in report["fatal_failures"]))

    def test_gate_fails_on_premium_dispatch_success_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, production = self._write_pair(
                tmp,
                [_scenario("premium_sla_pressure", dispatch_success=0.70)],
                [_scenario("premium_sla_pressure", dispatch_success=0.90)],
            )

            report = evaluate_gate(candidate_summary=candidate, production_summary=production)

            self.assertEqual(report["decision"], "FAIL")
            self.assertTrue(any("premium_sla_pressure dispatch_success" in item for item in report["fatal_failures"]))

    def test_gate_fails_on_route_dispatch_success_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, production = self._write_pair(
                tmp,
                [_scenario("route_disruption_congestion", dispatch_success=0.70)],
                [_scenario("route_disruption_congestion", dispatch_success=0.90)],
            )

            report = evaluate_gate(candidate_summary=candidate, production_summary=production)

            self.assertEqual(report["decision"], "FAIL")
            self.assertTrue(any("route_disruption_congestion dispatch_success" in item for item in report["fatal_failures"]))

    def test_gate_fails_on_baseline_service_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, production = self._write_pair(
                tmp,
                [_scenario("baseline_normal", service=0.90)],
                [_scenario("baseline_normal", service=0.94)],
            )

            report = evaluate_gate(candidate_summary=candidate, production_summary=production)

            self.assertEqual(report["decision"], "FAIL")
            self.assertTrue(any("baseline_normal service" in item for item in report["fatal_failures"]))

    def test_gate_fails_on_no_current_work_top_action_explosion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, production = self._write_pair(
                tmp,
                [
                    _scenario(
                        "premium_sla_pressure",
                        action_no_current_work_by_id={"39": 120},
                        action_no_unassigned_by_id={"39": 120},
                        top_action_ids=[{"action_id": 39, "count": 308, "percentage": 0.35}],
                    )
                ],
                [
                    _scenario(
                        "premium_sla_pressure",
                        action_no_current_work_by_id={"39": 10},
                        action_no_unassigned_by_id={"39": 10},
                        top_action_ids=[{"action_id": 39, "count": 20, "percentage": 0.02}],
                    )
                ],
            )

            report = evaluate_gate(candidate_summary=candidate, production_summary=production)

            self.assertEqual(report["decision"], "FAIL")
            self.assertTrue(any("no-current-work/no-unassigned" in item for item in report["fatal_failures"]))
```

- [ ] **Step 2: Run RED verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- FAIL because `evaluate_gate` is not defined.

- [ ] **Step 3: Implement fatal rules**

Add to `src/eval/long_run_gate.py`:

```python
from src.eval.scenario_metrics import GLOBAL_FATAL_HARD_BLOCKER_FIELDS


DISPATCH_SUCCESS_REGRESSION_TOLERANCE = 0.005
SERVICE_REGRESSION_TOLERANCE = 0.005
NO_WORK_TOTAL_MULTIPLIER = 2.0
NO_WORK_TOTAL_ABSOLUTE_DELTA = 50
NO_WORK_ACTION_MIN_COUNT = 100
NO_WORK_ACTION_TOP_PERCENTAGE = 0.10


def _float(scenario: Mapping[str, Any], key: str) -> float:
    try:
        return float(scenario.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _sum_mapping(scenario: Mapping[str, Any], key: str) -> int:
    raw = scenario.get(key, {})
    if not isinstance(raw, Mapping):
        return 0
    return sum(int(value) for value in raw.values())


def _action_no_work_total(scenario: Mapping[str, Any], action_id: str) -> int:
    current = scenario.get("action_no_current_work_by_id", {})
    unassigned = scenario.get("action_no_unassigned_by_id", {})
    current_count = int(current.get(action_id, 0)) if isinstance(current, Mapping) else 0
    unassigned_count = int(unassigned.get(action_id, 0)) if isinstance(unassigned, Mapping) else 0
    return current_count + unassigned_count


def _top_action_percentage(scenario: Mapping[str, Any], action_id: str) -> float:
    raw = scenario.get("top_action_ids", [])
    if not isinstance(raw, list):
        return 0.0
    for item in raw:
        if isinstance(item, Mapping) and str(item.get("action_id")) == str(action_id):
            return _float(item, "percentage")
    return 0.0


def _fatal_failures(candidate: EvalSummary, production: EvalSummary) -> list[str]:
    failures: list[str] = []
    for scenario_id, scenario in candidate.scenarios.items():
        if scenario.get("scenario_threshold_verdict") != "PASS":
            failures.append(f"{scenario_id}: scenario threshold verdict is {scenario.get('scenario_threshold_verdict')!r}.")
        for item in scenario.get("scenario_threshold_failures", []) or []:
            failures.append(f"{scenario_id}: scenario threshold failure: {item}")
        if scenario.get("hard_blocker_verdict") != "PASS":
            failures.append(f"{scenario_id}: hard blocker verdict is {scenario.get('hard_blocker_verdict')!r}.")
        for item in scenario.get("hard_blocker_failures", []) or []:
            failures.append(f"{scenario_id}: hard blocker failure: {item}")
        for key, _label in GLOBAL_FATAL_HARD_BLOCKER_FIELDS:
            value = _float(scenario, key)
            if value > 0:
                failures.append(f"{scenario_id}: global hard blocker {key}={value:g}.")

    for scenario_id in ("premium_sla_pressure", "route_disruption_congestion"):
        candidate_scenario = candidate.scenarios.get(scenario_id)
        production_scenario = production.scenarios.get(scenario_id)
        if candidate_scenario is None or production_scenario is None:
            continue
        candidate_success = _float(candidate_scenario, "dispatch_success_per_attempt")
        production_success = _float(production_scenario, "dispatch_success_per_attempt")
        if candidate_success + DISPATCH_SUCCESS_REGRESSION_TOLERANCE < production_success:
            failures.append(
                f"{scenario_id} dispatch_success regressed from {production_success:.3f} to {candidate_success:.3f}."
            )

    candidate_baseline = candidate.scenarios.get("baseline_normal")
    production_baseline = production.scenarios.get("baseline_normal")
    if candidate_baseline is not None and production_baseline is not None:
        candidate_service = _float(candidate_baseline, "service_level")
        production_service = _float(production_baseline, "service_level")
        if candidate_service + SERVICE_REGRESSION_TOLERANCE < production_service:
            failures.append(
                f"baseline_normal service regressed from {production_service:.3f} to {candidate_service:.3f}."
            )

    for scenario_id, candidate_scenario in candidate.scenarios.items():
        production_scenario = production.scenarios.get(scenario_id, {})
        candidate_total = _sum_mapping(candidate_scenario, "action_no_current_work_by_id") + _sum_mapping(
            candidate_scenario, "action_no_unassigned_by_id"
        )
        production_total = _sum_mapping(production_scenario, "action_no_current_work_by_id") + _sum_mapping(
            production_scenario, "action_no_unassigned_by_id"
        )
        if candidate_total > max(
            int(production_total * NO_WORK_TOTAL_MULTIPLIER),
            production_total + NO_WORK_TOTAL_ABSOLUTE_DELTA,
        ):
            failures.append(
                f"{scenario_id}: no-current-work/no-unassigned total exploded from {production_total} to {candidate_total}."
            )
        action_ids = set()
        for key in ("action_no_current_work_by_id", "action_no_unassigned_by_id"):
            raw = candidate_scenario.get(key, {})
            if isinstance(raw, Mapping):
                action_ids.update(str(action_id) for action_id in raw)
        for action_id in sorted(action_ids):
            action_total = _action_no_work_total(candidate_scenario, action_id)
            top_pct = _top_action_percentage(candidate_scenario, action_id)
            if action_total >= NO_WORK_ACTION_MIN_COUNT and top_pct >= NO_WORK_ACTION_TOP_PERCENTAGE:
                failures.append(
                    f"{scenario_id}: no-current-work/no-unassigned top-action explosion on action {action_id}: "
                    f"count={action_total}, top_percentage={top_pct:.3f}."
                )
    return failures


def evaluate_gate(
    *,
    candidate_summary: Path,
    production_summary: Path,
    previous_summary: Path | None = None,
) -> dict[str, Any]:
    candidate = load_summary(candidate_summary)
    production = load_summary(production_summary)
    previous = load_summary(previous_summary) if previous_summary is not None else None
    fatal_failures = _fatal_failures(candidate, production)
    return {
        "decision": "FAIL" if fatal_failures else "PASS",
        "candidate_checkpoint": candidate.checkpoint,
        "production_checkpoint": production.checkpoint,
        "previous_checkpoint": previous.checkpoint if previous is not None else None,
        "fatal_failures": fatal_failures,
        "warnings": [],
    }
```

- [ ] **Step 4: Run GREEN verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- PASS.

## Task 3: Add Warning Rules And Previous-Gate Comparison

**Files:**
- Modify: `tests/eval/test_long_run_gate.py`
- Modify: `src/eval/long_run_gate.py`

- [ ] **Step 1: Write warning tests**

Append:

```python
class LongRunGateWarningTests(unittest.TestCase):
    def test_gate_warns_on_nonbaseline_service_regression_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.json"
            production = root / "production.json"
            _summary(candidate, [_scenario("mixed_stress", service=0.94)])
            _summary(production, [_scenario("mixed_stress", service=0.96)])

            report = evaluate_gate(candidate_summary=candidate, production_summary=production)

            self.assertEqual(report["decision"], "PASS")
            self.assertTrue(any("mixed_stress service" in item for item in report["warnings"]))

    def test_gate_warns_on_previous_gate_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.json"
            production = root / "production.json"
            previous = root / "previous.json"
            _summary(candidate, [_scenario("mixed_stress", service=0.94, dispatch_success=0.95)])
            _summary(production, [_scenario("mixed_stress", service=0.94, dispatch_success=0.95)])
            _summary(previous, [_scenario("mixed_stress", service=0.97, dispatch_success=0.99)])

            report = evaluate_gate(
                candidate_summary=candidate,
                production_summary=production,
                previous_summary=previous,
            )

            self.assertEqual(report["decision"], "PASS")
            self.assertTrue(any("previous gate service" in item for item in report["warnings"]))
            self.assertTrue(any("previous gate dispatch_success" in item for item in report["warnings"]))
```

- [ ] **Step 2: Run RED verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- FAIL because warnings are not implemented.

- [ ] **Step 3: Implement warning rules**

Add warning constants and `_warnings()` to `src/eval/long_run_gate.py`, then call it from `evaluate_gate`:

```python
DISPATCH_RATE_WARNING_DELTA = 0.15
TOP_ACTION_WARNING_DELTA = 0.15


def _max_top_action_percentage(scenario: Mapping[str, Any]) -> float:
    raw = scenario.get("top_action_ids", [])
    if not isinstance(raw, list):
        return 0.0
    values = [_float(item, "percentage") for item in raw if isinstance(item, Mapping)]
    return max(values, default=0.0)


def _warnings(candidate: EvalSummary, production: EvalSummary, previous: EvalSummary | None) -> list[str]:
    warnings: list[str] = []
    for scenario_id, candidate_scenario in candidate.scenarios.items():
        production_scenario = production.scenarios.get(scenario_id)
        if production_scenario is None:
            warnings.append(f"{scenario_id}: missing production comparison scenario.")
            continue
        candidate_service = _float(candidate_scenario, "service_level")
        production_service = _float(production_scenario, "service_level")
        candidate_lateness = _float(candidate_scenario, "true_lateness_pressure")
        production_lateness = _float(production_scenario, "true_lateness_pressure")
        candidate_dispatch_rate = _float(candidate_scenario, "dispatch_rate")
        production_dispatch_rate = _float(production_scenario, "dispatch_rate")
        if scenario_id != "baseline_normal" and candidate_service + SERVICE_REGRESSION_TOLERANCE < production_service:
            warnings.append(
                f"{scenario_id} service is below production: {candidate_service:.3f} vs {production_service:.3f}."
            )
        if candidate_lateness > production_lateness + SERVICE_REGRESSION_TOLERANCE:
            warnings.append(
                f"{scenario_id} lateness is above production: {candidate_lateness:.3f} vs {production_lateness:.3f}."
            )
        if candidate_dispatch_rate > production_dispatch_rate + DISPATCH_RATE_WARNING_DELTA:
            warnings.append(
                f"{scenario_id} dispatch_rate is above production: {candidate_dispatch_rate:.3f} vs {production_dispatch_rate:.3f}."
            )
        if _max_top_action_percentage(candidate_scenario) > _max_top_action_percentage(production_scenario) + TOP_ACTION_WARNING_DELTA:
            warnings.append(f"{scenario_id} top action concentration increased over production.")
        if scenario_id == "route_disruption_congestion":
            candidate_hr = _float(candidate_scenario, "high_resilience_selected_when_not_best_count")
            production_hr = _float(production_scenario, "high_resilience_selected_when_not_best_count")
            if candidate_hr > production_hr:
                warnings.append(
                    f"{scenario_id} high_resilience_selected_when_not_best_count increased: "
                    f"{candidate_hr:g} vs {production_hr:g}."
                )
        candidate_mixed_failure = _float(candidate_scenario, "mixed_success_route_failure_steps")
        production_mixed_failure = _float(production_scenario, "mixed_success_route_failure_steps")
        if candidate_mixed_failure > production_mixed_failure:
            warnings.append(
                f"{scenario_id} mixed_success_route_failure_steps increased: "
                f"{candidate_mixed_failure:g} vs {production_mixed_failure:g}."
            )
        if previous is not None:
            previous_scenario = previous.scenarios.get(scenario_id)
            if previous_scenario is not None:
                previous_service = _float(previous_scenario, "service_level")
                previous_dispatch_success = _float(previous_scenario, "dispatch_success_per_attempt")
                candidate_dispatch_success = _float(candidate_scenario, "dispatch_success_per_attempt")
                if candidate_service + SERVICE_REGRESSION_TOLERANCE < previous_service:
                    warnings.append(
                        f"{scenario_id} previous gate service regressed from {previous_service:.3f} to {candidate_service:.3f}."
                    )
                if candidate_dispatch_success + DISPATCH_SUCCESS_REGRESSION_TOLERANCE < previous_dispatch_success:
                    warnings.append(
                        f"{scenario_id} previous gate dispatch_success regressed from "
                        f"{previous_dispatch_success:.3f} to {candidate_dispatch_success:.3f}."
                    )
    return warnings
```

Update `evaluate_gate`:

```python
warnings = _warnings(candidate, production, previous)
...
"warnings": warnings,
```

- [ ] **Step 4: Run GREEN verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- PASS.

## Task 4: Add CLI Wrapper

**Files:**
- Modify: `tests/eval/test_long_run_gate.py`
- Create: `src/eval/check_long_run_gate.py`

- [ ] **Step 1: Write failing CLI tests**

Append:

```python
from src.eval.check_long_run_gate import main as gate_main
from unittest.mock import patch


class LongRunGateCliTests(unittest.TestCase):
    def test_cli_prints_json_and_returns_zero_on_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.json"
            production = root / "production.json"
            _summary(candidate, [_scenario("baseline_normal", service=0.95)])
            _summary(production, [_scenario("baseline_normal", service=0.95)])

            with patch("builtins.print") as mocked_print:
                exit_code = gate_main(
                    [
                        "--candidate-summary",
                        str(candidate),
                        "--production-summary",
                        str(production),
                        "--candidate-label",
                        "candidate",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(mocked_print.call_args.args[0])
            self.assertEqual(payload["decision"], "PASS")
            self.assertEqual(payload["candidate_label"], "candidate")

    def test_cli_returns_two_on_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.json"
            production = root / "production.json"
            _summary(candidate, [_scenario("baseline_normal", service=0.80)])
            _summary(production, [_scenario("baseline_normal", service=0.95)])

            with patch("builtins.print"):
                exit_code = gate_main(
                    [
                        "--candidate-summary",
                        str(candidate),
                        "--production-summary",
                        str(production),
                    ]
                )

            self.assertEqual(exit_code, 2)
```

- [ ] **Step 2: Run RED verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- FAIL because `src.eval.check_long_run_gate` does not exist.

- [ ] **Step 3: Implement CLI**

Create `src/eval/check_long_run_gate.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.eval.long_run_gate import evaluate_gate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only long-run eval gate checker.")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--production-summary", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path, default=None)
    parser.add_argument("--candidate-label", default="candidate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_gate(
        candidate_summary=args.candidate_summary,
        production_summary=args.production_summary,
        previous_summary=args.previous_summary,
    )
    report["candidate_label"] = str(args.candidate_label)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run GREEN verification**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- PASS.

## Task 5: Run Real Artifact Read-Only Sanity Checks

**Files:**
- Read: production, next-gen 1M, and next-gen 300k `scenario_summary.json` files.
- No writes.

- [ ] **Step 1: Run checker against known failing next-gen 1M**

Run:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models/eval/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609_offline_scenarios/scenario_summary.json `
  --production-summary models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json `
  --candidate-label nextgen_1m
```

Expected:

- Exit code `2`.
- JSON `decision` is `FAIL`.
- Fatal failures include scenario threshold failures for `premium_sla_pressure` and `route_disruption_congestion`.
- Fatal failures include dispatch-success regressions versus production for premium and route.
- Fatal failures include no-current-work/no-unassigned explosion for premium or route if the rule thresholds are met.

- [ ] **Step 2: Run checker against known failing next-gen 300k**

Run:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models/eval/joint_torch_v5_prod_balanced_retention_nextgen_300k_20260609_offline_scenarios/scenario_summary.json `
  --production-summary models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json `
  --candidate-label nextgen_300k
```

Expected:

- Exit code `2`.
- JSON `decision` is `FAIL`.
- Fatal failures include threshold failures for baseline, route disruption, demand spike, lead time, high holding cost, and mixed stress.
- Fatal failures include baseline service regression versus production.

- [ ] **Step 3: Run checker against production as candidate**

Run:

```powershell
python -m src.eval.check_long_run_gate `
  --candidate-summary models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json `
  --production-summary models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios/scenario_summary.json `
  --candidate-label production_self_check
```

Expected:

- Exit code `0`.
- JSON `decision` is `PASS`.
- `fatal_failures` is empty.

## Task 6: Final Verification

**Files:**
- Compile new Python files only.
- Run focused unit tests only.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m unittest tests.eval.test_long_run_gate -v
```

Expected:

- PASS.

- [ ] **Step 2: Compile changed Python files**

Run:

```powershell
python -m py_compile src\eval\long_run_gate.py src\eval\check_long_run_gate.py tests\eval\test_long_run_gate.py
```

Expected:

- Exit code `0`.

- [ ] **Step 3: Confirm no prohibited process is running**

Run:

```powershell
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -match 'python|py' -and (
    ($_.CommandLine -like '*train_joint_torch*') -or
    ($_.CommandLine -like '*evaluate_real_world_scenarios*')
  )
} | Measure-Object | Select-Object -ExpandProperty Count
```

Expected:

- `0`.

## Non-Goals

This automation must not:

- Run training.
- Run offline evaluation.
- Update registry.
- Mutate `active_models.json`.
- Append `models.jsonl`.
- Mutate production.
- Mutate baselines.
- Mutate DB rows.
- Mutate checkpoints.
- Edit eval outputs.
- Start 3M, 5M, 10M, or 100M branches.

## Final Classification After Implementation

If the plan is implemented and all verification passes:

`LONG_RUN_GATE_CHECKER_READY`

For this plan-only readiness step:

`LONG_RUN_GATE_AUTOMATION_PLAN_READY`

