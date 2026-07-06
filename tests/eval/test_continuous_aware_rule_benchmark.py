from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import continuous_aware_rule_baseline_benchmark as bench


class ContinuousAwareRuleBenchmarkTests(unittest.TestCase):
    def test_build_chunk_plan_crosses_modes_baselines_and_scenarios(self) -> None:
        plan = bench.build_chunk_plan(["s2", "s1"], ["b1"], ["heuristic_continuous", "neutral_continuous"])

        self.assertEqual(4, len(plan))
        self.assertEqual(
            bench.ContinuousBenchmarkChunk(
                continuous_mode="heuristic_continuous",
                baseline_name="b1",
                scenario_id="s1",
            ),
            plan[0],
        )

    def test_write_and_aggregate_mode_chunks_to_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "continuous"
            output_dir.mkdir()
            plan = bench.build_chunk_plan(["s1"], ["b1"], ["neutral_continuous", "heuristic_continuous"])
            for chunk in plan:
                bench.write_chunk_artifacts(
                    output_dir=output_dir,
                    chunk=chunk,
                    summary={
                        "continuous_mode": chunk.continuous_mode,
                        "baseline_name": chunk.baseline_name,
                        "scenario_id": chunk.scenario_id,
                        "episodes": 1,
                        "steps": 1,
                        "service_level": 1.0,
                        "mean_lateness": 0.0,
                        "dispatch_success_per_attempt": 1.0,
                        "hard_blockers": {},
                    },
                    episode_rows=[
                        {
                            "continuous_mode": chunk.continuous_mode,
                            "baseline_name": chunk.baseline_name,
                            "scenario_id": chunk.scenario_id,
                            "episode_index": 0,
                            "service_level": 1.0,
                        }
                    ],
                )

            report = bench.aggregate_continuous_chunks(
                output_dir=output_dir,
                plan=plan,
                episodes=1,
                seed=42,
                scenario_dir=Path("configs/eval_scenarios"),
            )

            self.assertEqual("CONTINUOUS_AWARE_RULE_BASELINE_BENCHMARK_READY", report["decision"])
            self.assertEqual(2, report["benchmark_metadata"]["continuous_mode_count"])
            self.assertEqual(2, report["benchmark_metadata"]["episode_rows"])
            self.assertTrue((output_dir / "continuous_aware_rule_baseline_report.json").exists())
            self.assertTrue((output_dir / "continuous_aware_rule_baseline_summary.csv").exists())
            self.assertTrue((output_dir / "episode_metrics.jsonl").exists())

    def test_aggregate_refuses_existing_final_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "continuous"
            output_dir.mkdir()
            (output_dir / "continuous_aware_rule_baseline_report.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                bench.aggregate_continuous_chunks(
                    output_dir=output_dir,
                    plan=[],
                    episodes=1,
                    seed=42,
                    scenario_dir=Path("configs/eval_scenarios"),
                )

    def test_ppo_continuous_provider_uses_policy_observation(self) -> None:
        provider = bench.build_ppo_continuous_provider(
            checkpoint_path=Path("fake.pt"),
            device="cpu",
            policy_loader=lambda *_args, **_kwargs: _FakePolicy(),
        )

        action = provider(np.zeros(73, dtype=np.float32))

        self.assertEqual(action.shape, (5,))
        self.assertAlmostEqual(float(action[0]), 0.25)

    def test_ppo_continuous_provider_rejects_missing_observation(self) -> None:
        provider = bench.build_ppo_continuous_provider(
            checkpoint_path=Path("fake.pt"),
            device="cpu",
            policy_loader=lambda *_args, **_kwargs: _FakePolicy(),
        )

        with self.assertRaises(ValueError):
            provider(None)


class _FakeAction:
    def __init__(self) -> None:
        self.continuous = np.array([0.25, 0.0, 0.0, -0.25, -0.5], dtype=np.float32)
        self.discrete = 24


class _FakePolicy:
    checkpoint = {
        "global_step": 1_000_000,
        "dqn_architecture": "hierarchical_v1",
        "hierarchical_init_method": "flat_teacher_distillation_v1",
    }

    def predict_joint(self, observation, *, deterministic: bool = True):
        self.observation = np.asarray(observation)
        return _FakeAction()


if __name__ == "__main__":
    unittest.main()
