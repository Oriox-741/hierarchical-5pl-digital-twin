from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import benchmark_runtime_latency as latency


class RuntimeLatencyBenchmarkTests(unittest.TestCase):
    def test_compute_latency_stats_ms(self) -> None:
        stats = latency.compute_latency_stats_ms([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["min_ms"], 1.0)
        self.assertEqual(stats["max_ms"], 4.0)
        self.assertEqual(stats["mean_ms"], 2.5)
        self.assertEqual(stats["p50_ms"], 2.5)
        self.assertGreaterEqual(stats["p95_ms"], stats["p50_ms"])

    def test_validate_prediction_accepts_legal_action(self) -> None:
        action = latency.validate_prediction(_Action(np.zeros(5, dtype=np.float32), 47))

        self.assertEqual(action["continuous_dim"], 5)
        self.assertEqual(action["discrete"], 47)

    def test_validate_prediction_rejects_illegal_action(self) -> None:
        with self.assertRaises(ValueError):
            latency.validate_prediction(_Action(np.zeros(4, dtype=np.float32), 0))
        with self.assertRaises(ValueError):
            latency.validate_prediction(_Action(np.zeros(5, dtype=np.float32), 48))
        with self.assertRaises(ValueError):
            latency.validate_prediction(_Action(np.array([float("nan")] * 5, dtype=np.float32), 0))

    def test_build_report_with_injected_policy_loader(self) -> None:
        report = latency.build_runtime_latency_report(
            checkpoint_path=Path("fake.pt"),
            iterations=5,
            seed=123,
            device="cpu",
            policy_loader=lambda *_args, **_kwargs: _Policy(),
            timer=_IncrementingTimer(),
        )

        self.assertEqual(report["decision"], "RUNTIME_LATENCY_BENCHMARK_READY")
        self.assertEqual(report["prediction_validation"]["valid_predictions"], 5)
        self.assertEqual(report["latency_ms"]["count"], 5)
        self.assertEqual(report["runtime_metadata"]["obs_dim"], 73)
        self.assertEqual(report["runtime_metadata"]["discrete_action_count"], 48)

    def test_cli_writes_only_requested_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "latency.json"
            before = _relative_files(root)

            report = latency.write_runtime_latency_report(
                output,
                {
                    "decision": "RUNTIME_LATENCY_BENCHMARK_READY",
                    "runtime_metadata": {},
                    "latency_ms": {"count": 1},
                    "prediction_validation": {"valid_predictions": 1},
                },
            )

            self.assertEqual(report["decision"], "RUNTIME_LATENCY_BENCHMARK_READY")
            self.assertEqual(before | {Path("latency.json")}, _relative_files(root))
            self.assertEqual(
                "RUNTIME_LATENCY_BENCHMARK_READY",
                json.loads(output.read_text(encoding="utf-8"))["decision"],
            )

    def test_output_guard_rejects_existing_or_protected_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "exists.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                latency.ensure_fresh_output_file(output)

        with self.assertRaises(ValueError):
            latency.ensure_fresh_output_file(Path("models/production/bad.json"))
        with self.assertRaises(ValueError):
            latency.ensure_fresh_output_file(Path.cwd() / "models" / "production" / "bad.json")


class _Action:
    def __init__(self, continuous: np.ndarray, discrete: int) -> None:
        self.continuous = continuous
        self.discrete = discrete


class _Policy:
    checkpoint = {
        "global_step": 1_000_000,
        "dqn_architecture": "hierarchical_v1",
        "hierarchical_init_method": "flat_teacher_distillation_v1",
    }

    def predict_joint(self, observation, *, deterministic: bool = True):
        return _Action(np.zeros(5, dtype=np.float32), 24)


class _IncrementingTimer:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _relative_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


if __name__ == "__main__":
    unittest.main()
