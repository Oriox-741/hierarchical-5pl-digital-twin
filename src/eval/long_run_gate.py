"""Read-only long-run evaluation gate checker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.scenario_metrics import GLOBAL_FATAL_HARD_BLOCKER_FIELDS


DISPATCH_SUCCESS_REGRESSION_TOLERANCE = 0.005
SERVICE_REGRESSION_TOLERANCE = 0.005
NO_WORK_TOTAL_MULTIPLIER = 2.0
NO_WORK_TOTAL_ABSOLUTE_DELTA = 50
NO_WORK_ACTION_MIN_COUNT = 100
NO_WORK_ACTION_TOP_PERCENTAGE = 0.10
DISPATCH_RATE_WARNING_DELTA = 0.15
TOP_ACTION_WARNING_DELTA = 0.15


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
    fatal_scenarios = _fatal_scenario_ids(fatal_failures)
    warnings = _warnings(candidate, production, previous, fatal_scenarios)
    return {
        "decision": "FAIL" if fatal_failures else "PASS",
        "candidate_checkpoint": candidate.checkpoint,
        "production_checkpoint": production.checkpoint,
        "previous_checkpoint": previous.checkpoint if previous is not None else None,
        "fatal_failures": fatal_failures,
        "warnings": warnings,
        "scenario_comparisons": _scenario_comparisons(candidate, production, previous),
    }


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


def _max_top_action_percentage(scenario: Mapping[str, Any]) -> float:
    raw = scenario.get("top_action_ids", [])
    if not isinstance(raw, list):
        return 0.0
    values = [_float(item, "percentage") for item in raw if isinstance(item, Mapping)]
    return max(values, default=0.0)


def _fatal_failures(candidate: EvalSummary, production: EvalSummary) -> list[str]:
    failures: list[str] = []
    for scenario_id, scenario in candidate.scenarios.items():
        if scenario.get("scenario_threshold_verdict") != "PASS":
            failures.append(
                f"{scenario_id}: scenario threshold verdict is {scenario.get('scenario_threshold_verdict')!r}."
            )
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
                f"{scenario_id}: no-current-work/no-unassigned total exploded from "
                f"{production_total} to {candidate_total}."
            )
        action_ids = set()
        for key in ("action_no_current_work_by_id", "action_no_unassigned_by_id"):
            raw = candidate_scenario.get(key, {})
            if isinstance(raw, Mapping):
                action_ids.update(str(action_id) for action_id in raw)
        for action_id in sorted(action_ids):
            action_total = _action_no_work_total(candidate_scenario, action_id)
            production_action_total = _action_no_work_total(production_scenario, action_id)
            top_pct = _top_action_percentage(candidate_scenario, action_id)
            action_threshold = max(
                int(production_action_total * NO_WORK_TOTAL_MULTIPLIER),
                production_action_total + NO_WORK_TOTAL_ABSOLUTE_DELTA,
            )
            if (
                action_total >= NO_WORK_ACTION_MIN_COUNT
                and action_total > action_threshold
                and top_pct >= NO_WORK_ACTION_TOP_PERCENTAGE
            ):
                failures.append(
                    f"{scenario_id}: no-current-work/no-unassigned top-action explosion on action {action_id}: "
                    f"count={action_total}, production_count={production_action_total}, top_percentage={top_pct:.3f}."
                )
    return failures


def _warnings(
    candidate: EvalSummary,
    production: EvalSummary,
    previous: EvalSummary | None,
    fatal_scenarios: set[str],
) -> list[str]:
    warnings: list[str] = []
    for scenario_id, candidate_scenario in candidate.scenarios.items():
        if scenario_id in fatal_scenarios:
            continue
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
                f"{scenario_id} dispatch_rate is above production: "
                f"{candidate_dispatch_rate:.3f} vs {production_dispatch_rate:.3f}."
            )
        if _max_top_action_percentage(candidate_scenario) > (
            _max_top_action_percentage(production_scenario) + TOP_ACTION_WARNING_DELTA
        ):
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
                        f"{scenario_id} previous gate service regressed from "
                        f"{previous_service:.3f} to {candidate_service:.3f}."
                    )
                if candidate_dispatch_success + DISPATCH_SUCCESS_REGRESSION_TOLERANCE < previous_dispatch_success:
                    warnings.append(
                        f"{scenario_id} previous gate dispatch_success regressed from "
                        f"{previous_dispatch_success:.3f} to {candidate_dispatch_success:.3f}."
                    )
    return warnings


def _scenario_comparisons(
    candidate: EvalSummary,
    production: EvalSummary,
    previous: EvalSummary | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id in sorted(candidate.scenarios):
        candidate_scenario = candidate.scenarios[scenario_id]
        production_scenario = production.scenarios.get(scenario_id, {})
        previous_scenario = previous.scenarios.get(scenario_id, {}) if previous is not None else {}
        rows.append(
            {
                "scenario_id": scenario_id,
                "candidate_verdict": candidate_scenario.get("verdict"),
                "candidate_threshold_verdict": candidate_scenario.get("scenario_threshold_verdict"),
                "candidate_hard_blocker_verdict": candidate_scenario.get("hard_blocker_verdict"),
                "candidate_service_level": _float(candidate_scenario, "service_level"),
                "production_service_level": _float(production_scenario, "service_level"),
                "previous_service_level": _float(previous_scenario, "service_level") if previous else None,
                "candidate_lateness": _float(candidate_scenario, "true_lateness_pressure"),
                "production_lateness": _float(production_scenario, "true_lateness_pressure"),
                "previous_lateness": _float(previous_scenario, "true_lateness_pressure") if previous else None,
                "candidate_dispatch_success": _float(candidate_scenario, "dispatch_success_per_attempt"),
                "production_dispatch_success": _float(production_scenario, "dispatch_success_per_attempt"),
                "previous_dispatch_success": (
                    _float(previous_scenario, "dispatch_success_per_attempt") if previous else None
                ),
                "candidate_dispatch_rate": _float(candidate_scenario, "dispatch_rate"),
                "production_dispatch_rate": _float(production_scenario, "dispatch_rate"),
                "previous_dispatch_rate": _float(previous_scenario, "dispatch_rate") if previous else None,
                "candidate_no_work_unassigned_total": _sum_mapping(
                    candidate_scenario, "action_no_current_work_by_id"
                )
                + _sum_mapping(candidate_scenario, "action_no_unassigned_by_id"),
                "production_no_work_unassigned_total": _sum_mapping(
                    production_scenario, "action_no_current_work_by_id"
                )
                + _sum_mapping(production_scenario, "action_no_unassigned_by_id"),
            }
        )
    return rows


def _fatal_scenario_ids(fatal_failures: list[str]) -> set[str]:
    scenario_ids: set[str] = set()
    for failure in fatal_failures:
        prefix = failure.split(":", 1)[0]
        if prefix:
            scenario_ids.add(prefix)
    return scenario_ids
