from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.eval.check_long_run_gate import main as gate_main
from src.eval.long_run_gate import evaluate_gate, load_summary


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
            self.assertTrue(
                any("route_disruption_congestion dispatch_success" in item for item in report["fatal_failures"])
            )

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

    def test_gate_does_not_fail_self_check_for_existing_production_action_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scenario = _scenario(
                "route_disruption_congestion",
                action_no_current_work_by_id={"41": 68},
                action_no_unassigned_by_id={"41": 67},
                top_action_ids=[{"action_id": 41, "count": 140, "percentage": 0.162}],
            )
            candidate, production = self._write_pair(tmp, [scenario], [scenario])

            report = evaluate_gate(candidate_summary=candidate, production_summary=production)

            self.assertEqual(report["decision"], "PASS")
            self.assertEqual(report["fatal_failures"], [])


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
